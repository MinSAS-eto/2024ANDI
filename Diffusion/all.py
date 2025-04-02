import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
import argparse
from tqdm import tqdm, trange

# 保留原CNN-BiLSTM模型的组件
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        attention = self.attn_dropout(attention)
        
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)
        context = self.output_dropout(context)
        return context, attention

class MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MultiScaleCNNBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k-1)//2))
            self.layernorms.append(nn.LayerNorm(out_channels))
        self.activation = nn.Mish()
    
    def forward(self, x):
        outs = []
        for conv, ln in zip(self.convs, self.layernorms):
            conv_out = conv(x)
            conv_out = conv_out.transpose(1, 2)
            conv_out = ln(conv_out)
            conv_out = conv_out.transpose(1, 2)
            conv_out = self.activation(conv_out)
            outs.append(conv_out)
        out = torch.cat(outs, dim=1)
        return out

class ResidualCNNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualCNNBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln = nn.LayerNorm(channels)
        self.activation = nn.Mish()
    
    def forward(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.ln(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        return x + out


# 修改模型：CNN-BiLSTM用于预测扩散指数
class CNNBiLSTMExponentPredictor(nn.Module):
    def __init__(self,
                 input_size=1,
                 conv_channels=256,
                 kernel_sizes=[3,5,7],
                 num_residual_blocks=1,
                 lstm_hidden_size=256,
                 lstm_layers=3,
                 bidirectional=True,
                 dropout_rate=0.3,
                 sequence_length=100):
        super(CNNBiLSTMExponentPredictor, self).__init__()
        
        self.mish = nn.Mish()
        self.input_size = input_size
        self.sequence_length = sequence_length
        num_kernels = len(kernel_sizes)
        
        # 多尺度CNN模块
        self.multi_scale_cnn = MultiScaleCNNBlock(in_channels=input_size, out_channels=conv_channels, kernel_sizes=kernel_sizes)
        self.residual_conv = nn.Conv1d(input_size, conv_channels * num_kernels, kernel_size=1)
        self.fusion_conv = nn.Conv1d(conv_channels * num_kernels, conv_channels, kernel_size=1)
        
        # 层归一化
        self.layernorm_after_fusion = nn.LayerNorm(conv_channels)
        
        # 残差CNN块
        self.residual_blocks = nn.ModuleList([
            ResidualCNNBlock(conv_channels, kernel_size=3) for _ in range(num_residual_blocks)
        ])
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.num_directions = 2 if bidirectional else 1
        hidden_dim = lstm_hidden_size * self.num_directions
        
        # 注意力机制
        self.attention = MultiHeadSelfAttention(hidden_size=hidden_dim, num_heads=2, dropout_rate=dropout_rate * 0.6)
        self.layernorm_post_attn = nn.LayerNorm(hidden_dim)
        
        # 特征处理层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.features_norm = nn.LayerNorm(hidden_dim // 2)
        
        # 输出层 - 修改为预测单个扩散指数
        self.fc_exponent = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Mish()
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'fc' in name or 'linear' in name or 'conv' in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def extract_features(self, x):
        """提取特征，用于后续预测"""
        # 确保x有正确的形状 [B, T, C]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        
        batch_size, seq_len, channels = x.shape
        
        # 检查序列长度是否符合预期
        if seq_len != self.sequence_length:
            if seq_len > self.sequence_length:
                x = x[:, :self.sequence_length, :]  # 裁剪
            else:
                # 填充（用零填充）
                padding = torch.zeros(batch_size, self.sequence_length - seq_len, channels, device=x.device)
                x = torch.cat([x, padding], dim=1)
            seq_len = self.sequence_length
        
        # 准备CNN输入 [B, C, T]
        x_cnn = x.transpose(1, 2)  # [B, C, T]
        
        # 1. 多尺度CNN特征提取
        cnn_features = self.multi_scale_cnn(x_cnn)  # [B, conv_channels * num_kernels, T]
        residual = self.residual_conv(x_cnn)  # [B, conv_channels * num_kernels, T]
        cnn_features = cnn_features + residual
        fused = self.fusion_conv(cnn_features)  # [B, conv_channels, T]
        fused = self.mish(fused)
        
        # 转换为 [B, T, C] 用于LayerNorm
        fused = fused.transpose(1, 2)  # [B, T, conv_channels]
        
        # 应用LayerNorm
        fused = self.layernorm_after_fusion(fused)
        fused = self.activation(fused)
        
        # 保存CNN全局信息用于跨模块残差
        cnn_avg = fused.mean(dim=1)  # [B, conv_channels]
        
        # 用残差CNN块处理
        fused = fused.transpose(1, 2)  # 转回 [B, C, T]
        for block in self.residual_blocks:
            fused = block(fused)
        fused = fused.transpose(1, 2)  # 转回 [B, T, C]
        
        # 2. BiLSTM层
        lstm_out, _ = self.lstm(fused)  # [B, T, hidden_dim]
        
        # 3. 多头注意力
        attn_out, _ = self.attention(lstm_out)  # [B, T, hidden_dim]
        
        # 应用层归一化和残差连接
        attn_out = self.layernorm_post_attn(attn_out + lstm_out)
        
        # 全局池化以获得序列级特征
        global_max_pool = torch.max(attn_out, dim=1)[0]
        global_avg_pool = torch.mean(attn_out, dim=1)
        global_features = torch.cat([global_max_pool, global_avg_pool], dim=1)
        
        # 最后的特征处理
        features = self.fc1(global_features)  # [B, hidden_dim // 2]
        features = self.features_norm(features)
        features = self.activation(features)
        features = self.dropout(features)
        
        return features
    
    def forward(self, x):
        """前向传播，预测扩散指数"""
        features = self.extract_features(x)
        # 预测扩散指数
        exponent = self.fc_exponent(features)  # [B, 1]
        return exponent.squeeze(-1)  # [B]


# 轨迹数据集
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, exponents, model_ids, segment_length=100, prediction_length=100, normalize=True):
        self.trajectories = trajectories
        self.exponents = exponents
        self.model_ids = model_ids
        self.segment_length = segment_length
        self.prediction_length = prediction_length
        self.normalize = normalize
        
        # 标准化轨迹数据
        if normalize:
            self._normalize_trajectories()
    
    def _normalize_trajectories(self):
        """标准化轨迹数据"""
        for i in range(len(self.trajectories)):
            traj = self.trajectories[i]
            traj_mean = traj.mean()
            traj_std = traj.std()
            if traj_std > 0:
                self.trajectories[i] = (traj - traj_mean) / traj_std
            else:
                self.trajectories[i] = traj - traj_mean
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        exponent = self.exponents[idx]
        model_id = self.model_ids[idx]
        
        # 对于预测任务，我们不需要划分为segment和prediction部分
        # 直接返回整个轨迹及其对应的扩散指数
        return torch.FloatTensor(trajectory), torch.FloatTensor([exponent]), model_id


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="异常扩散指数预测")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--segment_length", type=int, default=100, help="输入段长度")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--num_samples", type=int, default=50, help="每个模型的样本数")
    parser.add_argument("--traj_length", type=int, default=200, help="轨迹长度")
    return parser.parse_args()


# 可视化训练损失
def plot_losses(train_losses, val_losses=None, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    if val_losses:
        plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()


# 可视化预测结果
def plot_predictions(true_values, predicted_values, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predicted_values, alpha=0.6)
    
    # 添加对角线，表示完美预测
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实扩散指数')
    plt.ylabel('预测扩散指数')
    plt.title('扩散指数预测vs真实值')
    plt.grid(True)
    
    # 计算相关系数
    correlation = np.corrcoef(true_values, predicted_values)[0, 1]
    mse = np.mean((np.array(true_values) - np.array(predicted_values)) ** 2)
    mae = np.mean(np.abs(np.array(true_values) - np.array(predicted_values)))
    
    plt.text(0.05, 0.95, f'相关系数: {correlation:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


# 获取当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 示例加载ANDI数据函数
def load_andi_data(N=50, Length=200):
    """
    加载ANDI数据集
    注意：这是一个示例函数，需要根据实际数据加载逻辑修改
    """
    # 这里应该是实际的数据加载逻辑
    # 示例实现，生成随机数据
    trajectories = [np.random.randn(Length) for _ in range(N * 5)]  # 5种模型，每种N个样本
    exponents = np.random.uniform(0.5, 2.0, size=N * 5)  # 随机扩散指数
    model_ids = np.repeat(np.arange(5), N)  # 模型ID: 0,0,...0,1,1,...1,2,2,...
    
    return trajectories, exponents, model_ids


# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建模型保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载ANDI数据
    tqdm.write("正在加载ANDI数据...")
    trajectories, exponents, model_ids = load_andi_data(N=args.num_samples, Length=args.traj_length)
    tqdm.write(f"已加载 {len(trajectories)} 条轨迹，平均扩散指数: {np.mean(exponents):.4f}")
    
    # 创建数据集
    tqdm.write("正在创建轨迹数据集...")
    dataset = TrajectoryDataset(
        trajectories=trajectories,
        exponents=exponents,
        model_ids=model_ids,
        segment_length=args.segment_length,
        normalize=True
    )
    tqdm.write(f"数据集大小: {len(dataset)} 个样本")
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型
    tqdm.write(f"正在初始化模型，设备: {args.device}...")
    model = CNNBiLSTMExponentPredictor(
        input_size=1,
        sequence_length=args.segment_length
    ).to(args.device)
    
    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    tqdm.write("开始训练...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 使用trange作为整个训练过程的进度条
    pbar = trange(1, args.epochs + 1, desc="训练进度")
    
    for epoch in pbar:
        # 训练阶段
        model.train()
        epoch_losses = []
        
        # 设置训练进度条
        train_pbar = tqdm(train_loader, desc=f"训练", leave=False)
        for batch in train_pbar:
            # 提取数据并移动到设备
            trajectories, exponents, _ = batch
            trajectories = trajectories.to(args.device)
            exponents = exponents.to(args.device).squeeze()
            
            # 前向传播
            optimizer.zero_grad()
            predicted_exponents = model(trajectories)
            
            # 计算损失
            loss = criterion(predicted_exponents, exponents)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 记录损失
            epoch_losses.append(loss.item())
            current_lr = get_lr(optimizer)
            
            # 更新进度条信息
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        # 计算平均训练损失
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_epoch_losses = []
        all_true_exponents = []
        all_pred_exponents = []
        
        with torch.no_grad():
            # 设置验证进度条
            val_pbar = tqdm(val_loader, desc=f"验证", leave=False)
            for batch in val_pbar:
                # 提取数据并移动到设备
                trajectories, exponents, _ = batch
                trajectories = trajectories.to(args.device)
                exponents = exponents.to(args.device).squeeze()
                
                # 前向传播
                predicted_exponents = model(trajectories)
                
                # 计算验证损失
                val_loss = criterion(predicted_exponents, exponents)
                
                # 记录损失
                val_epoch_losses.append(val_loss.item())
                
                # 收集预测结果用于评估和可视化
                all_true_exponents.extend(exponents.cpu().numpy())
                all_pred_exponents.extend(predicted_exponents.cpu().numpy())
                
                # 更新进度条信息
                val_pbar.set_postfix({
                    'val_loss': f"{val_loss.item():.4f}"
                })
        
        # 计算平均验证损失
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 更新主进度条描述
        pbar.set_description(f"[训练] 轮次: {epoch}/{args.epochs}")
        pbar.set_postfix({
            '训练损失': f"{avg_train_loss:.4f}",
            '验证损失': f"{avg_val_loss:.4f}",
            'LR': f"{get_lr(optimizer):.6f}",
            '最佳损失': f"{best_val_loss:.4f}"
        })
        
        # 使用tqdm.write避免与进度条冲突
        tqdm.write(f"轮次 {epoch}/{args.epochs} - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, LR: {get_lr(optimizer):.6f}")
        
        # 每10轮可视化一次训练进度
        if epoch % 10 == 0:
            tqdm.write(f"生成第 {epoch} 轮的可视化...")
            plot_losses(
                train_losses, 
                val_losses, 
                save_path=os.path.join(args.checkpoint_dir, f"loss_epoch_{epoch}.png")
            )
            
            # 可视化预测结果
            plot_predictions(
                all_true_exponents,
                all_pred_exponents,
                save_path=os.path.join(args.checkpoint_dir, f"predictions_epoch_{epoch}.png")
            )
            
            tqdm.write(f"第 {epoch} 轮可视化已保存")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            tqdm.write(f"在第 {epoch} 轮保存了最佳模型，验证损失: {best_val_loss:.4f}")
        
        # 每20轮保存一次检查点
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
            tqdm.write(f"第 {epoch} 轮检查点已保存")
    
    # 训练结束，保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
    
    # 可视化最终的训练损失
    plot_losses(
        train_losses, 
        val_losses, 
        save_path=os.path.join(args.checkpoint_dir, "final_loss.png")
    )
    
    # 计算训练总时间
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    tqdm.write(f"训练完成！总时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    tqdm.write(f"最佳验证损失: {best_val_loss:.4f}")
    tqdm.write(f"模型已保存到: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
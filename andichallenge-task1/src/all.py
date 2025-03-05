import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from andi_datasets.datasets_challenge import challenge_theory_dataset
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1. 先写一个函数，只在这里调用一次 challenge_theory_dataset
###############################################################################
def load_andi_data(N, tasks):
    """
    返回 X, Y，其中 X 是 list/array of arrays (变长序列)，Y 是 list/array of labels
    """
    X1, Y1, _, _, _, _ = challenge_theory_dataset(N=N, tasks=tasks, dimensions=1)
    # 题目中维度固定为1，因此 X, Y 直接取 X1[0], Y1[0]
    X = X1[0]
    Y = Y1[0]
    return X, Y


###############################################################################
# 2. 对 X 做 fit，得到一个 StandardScaler
###############################################################################
def create_and_fit_scaler(X):
    """
    对所有序列 X 拼起来，然后 fit 一个 scaler 并返回。
    X: 形如 [array(seq1), array(seq2), ...]，seq_i 是长度不定的 1D 序列。
    """
    all_data = np.concatenate(X)    # 把每条序列连成一个长的一维数组
    all_data = all_data.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(all_data)
    return scaler


###############################################################################
# 3. 定义 Transform 用于数据标准化
###############################################################################
class StandardScalerTransform:
    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler

    def __call__(self, x):
        # 转成 numpy
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = np.array(x, dtype=np.float32)

        # reshape 成 [T, 1] 
        x_np = x_np.reshape(-1, 1)
        # 标准化
        x_scaled = self.scaler.transform(x_np)
        # reshape 回 [T]
        x_scaled = x_scaled.reshape(-1)

        return torch.tensor(x_scaled, dtype=torch.float32)

###############################################################################
# 4. 重新定义 AnDiDataset，不再在内部调用 challenge_theory_dataset，而是外部传入 X, Y
###############################################################################
class AnDiDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        """
        X: list of arrays (每条序列长度不定)
        Y: list/array of 对应标签
        transform: 如果需要对 x 做转换, 传入 transform
        """
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


###############################################################################
# 5. collate_fn 保持不变，只是对预处理后的 x 做 padding
###############################################################################
def collate_fn(batch, label_scaler=None):
    xs, ys = zip(*batch)
    xs_tensor = []
    lengths_x = []
    for x in xs:
        # x 如果是 numpy 或 list, 转成 torch
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)
        lengths_x.append(len(x))

    # padding
    padded_x = torch.nn.utils.rnn.pad_sequence(
        xs_tensor, batch_first=True, padding_value=0
    )
    
    # 标签标准化
    if label_scaler:
        ys_tensor = torch.tensor(label_scaler.transform(
            np.array(ys).reshape(-1, 1)).flatten(), dtype=torch.float32)
    else:
        ys_tensor = torch.tensor(ys, dtype=torch.float32)

    return padded_x, lengths_x, ys_tensor
class MultiScaleCNNBlock(nn.Module):
    """
    多尺度 CNN 模块：使用不同尺寸的卷积核提取多尺度特征，
    并在每个卷积后使用 LayerNorm（在时间步维度上标准化）。
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MultiScaleCNNBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        for k in kernel_sizes:
            # 使用 padding 保持序列长度不变
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k-1)//2))
            # LayerNorm 对于卷积输出先将 [B, C, T] 转为 [B, T, C] 后进行归一化
            self.layernorms.append(nn.LayerNorm(out_channels))
        self.activation = nn.Mish()
        
    def forward(self, x):
        # x: [B, in_channels, T]
        outs = []
        for conv, ln in zip(self.convs, self.layernorms):
            conv_out = conv(x)  # [B, out_channels, T]
            conv_out = conv_out.transpose(1, 2)  # [B, T, out_channels]
            conv_out = ln(conv_out)
            conv_out = conv_out.transpose(1, 2)  # [B, out_channels, T]
            conv_out = self.activation(conv_out)
            outs.append(conv_out)
        # 拼接多个尺度的特征，通道数变为 out_channels * len(kernel_sizes)
        out = torch.cat(outs, dim=1)
        return out

class ResidualCNNBlock(nn.Module):
    """
    一个简单的残差卷积块：
    对输入进行卷积、LayerNorm 和激活，然后与原始输入相加。
    """
    def __init__(self, channels, kernel_size=3):
        super(ResidualCNNBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln = nn.LayerNorm(channels)  # 后续需要转换维度为 [B, T, channels]
        self.activation = nn.Mish()
        
    def forward(self, x):
        # x: [B, channels, T]
        out = self.conv(x)  # [B, channels, T]
        out = out.transpose(1, 2)  # [B, T, channels]
        out = self.ln(out)
        out = out.transpose(1, 2)  # [B, channels, T]
        out = self.activation(out)
        return x + out  # 残差连接

class AttentionLayer(nn.Module):
    """
    简单的注意力机制，对 LSTM 输出每个时间步计算注意力得分，
    并利用 softmax 加权求和得到上下文向量。
    """
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, lstm_outputs, lengths):
        # lstm_outputs: [B, T, D]
        attn_scores = self.attention(lstm_outputs)  # [B, T, 1]
        attn_scores = attn_scores.squeeze(-1)  # [B, T]
        
        # 构造 mask，防止对 padding 部分计算注意力
        max_len = lstm_outputs.size(1)
        mask = torch.arange(max_len, device=lstm_outputs.device).unsqueeze(0) < torch.tensor(lengths, device=lstm_outputs.device).unsqueeze(1)
        attn_scores[~mask] = float('-inf')
        
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]
        attn_weights = attn_weights.unsqueeze(-1)  # [B, T, 1]
        context = torch.sum(lstm_outputs * attn_weights, dim=1)  # [B, D]
        return context

class ImprovedCNNBiLSTM(nn.Module):
    """
    改进版网络：
      1. 使用多尺度 CNN 提取特征，并添加残差连接及额外的 CNN 层加深网络；
      2. 使用 LayerNorm 对特征进行归一化；
      3. LSTM 层增加到 4 层，并在最后结合了来自 CNN 部分的全局特征进行残差融合；
      4. LSTM 层后添加注意力机制获得全局上下文向量，
         最后经过 dropout 和全连接层进行回归预测。
    
    输入: x -> [B, T] (T 为 padding 后的最大序列长度)
    """
    def __init__(self,
                 conv_channels=256,
                 lstm_hidden_size=512,
                 lstm_layers=4,   # 增加 LSTM 层数
                 bidirectional=True,
                 dropout_rate=0.5,
                 kernel_sizes=[3,5,7],
                 num_residual_blocks=1):  # 增加额外 CNN 残差块的数量
        super(ImprovedCNNBiLSTM, self).__init__()
        
        self.mish = nn.Mish()
        # 多尺度 CNN 模块：输入通道为1
        self.multi_scale_cnn = MultiScaleCNNBlock(in_channels=1, out_channels=conv_channels, kernel_sizes=kernel_sizes)
        
        # 残差连接：将原始输入经过 1×1 卷积映射到多尺度输出相同的通道数
        self.residual_conv = nn.Conv1d(1, conv_channels * len(kernel_sizes), kernel_size=1)
        
        # 融合层：1×1 卷积融合多尺度特征，输出通道数降为 conv_channels
        self.fusion_conv = nn.Conv1d(conv_channels * len(kernel_sizes), conv_channels, kernel_size=1)
        # LayerNorm 用于融合后的特征（需先转置到 [B, T, conv_channels]）
        self.layernorm_after_fusion = nn.LayerNorm(conv_channels)
        
        # 额外的 CNN 残差块
        self.num_residual_blocks = num_residual_blocks
        self.residual_blocks = nn.ModuleList([
            ResidualCNNBlock(conv_channels, kernel_size=3) for _ in range(num_residual_blocks)
        ])
        
        # LSTM 层：输入维度 = conv_channels，层数增加至 lstm_layers
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        num_directions = 2 if bidirectional else 1
        
        # 注意力层：对 LSTM 输出进行注意力加权
        self.attention_layer = AttentionLayer(lstm_hidden_size * num_directions)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size * num_directions, 1)
        
        # 跨模块残差连接：
        # 对 CNN 融合后的特征（全局平均池化）进行投影，以匹配 LSTM 输出的维度
        self.cnn_res_proj = nn.Linear(conv_channels, lstm_hidden_size * num_directions)
        
        self.init_weights()
    
    def init_weights(self):
        # 对卷积层和全连接层进行合适的初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, lengths):
        # 输入 x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        
        # 多尺度 CNN 特征提取
        cnn_features = self.multi_scale_cnn(x)  # [B, conv_channels * len(kernel_sizes), T]
        # 残差连接：原始输入经过 1×1 卷积映射
        residual = self.residual_conv(x)  # [B, conv_channels * len(kernel_sizes), T]
        cnn_features = cnn_features + residual
        
        # 融合多尺度特征
        fused = self.mish(self.fusion_conv(cnn_features))  # [B, conv_channels, T]
        
        # 转换维度以应用 LayerNorm：变为 [B, T, conv_channels]
        fused = fused.transpose(1, 2)
        fused = self.layernorm_after_fusion(fused)
        fused = self.mish(fused)  # [B, T, conv_channels]
        
        # 保存 CNN 融合特征的全局信息（全局平均池化），用于残差连接
        cnn_avg = fused.mean(dim=1)  # [B, conv_channels]
        
        # 将 fused 转换回 [B, conv_channels, T] 以便通过额外的 CNN 残差块
        fused = fused.transpose(1, 2)  # [B, conv_channels, T]
        for block in self.residual_blocks:
            fused = block(fused)  # [B, conv_channels, T]
        # 转换回 [B, T, conv_channels]，作为 LSTM 输入
        fused = fused.transpose(1, 2)  # [B, T, conv_channels]
        
        # 处理变长序列：pack 之后送入 LSTM
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=fused.device)
        packed_input = pack_padded_sequence(fused, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: [B, T, lstm_hidden_size * num_directions]
        
        # 注意力机制整合 LSTM 输出，得到上下文向量
        context = self.attention_layer(output, lengths)  # [B, lstm_hidden_size * num_directions]
        
        # 跨模块残差：将 CNN 全局特征投影后与上下文相加
        cnn_res = self.cnn_res_proj(cnn_avg)  # [B, lstm_hidden_size * num_directions]
        context = context + cnn_res
        
        context = self.dropout(context)
        out = self.fc(context)  # [B, 1]
        return out.squeeze(-1)  # 将形状从[B, 1]变为[B]

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 包装 dataloader 以显示训练进度
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, batch in pbar:
        x, lengths, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, lengths)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        # 在进度条中显示当前 batch 的 loss
        pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device, y_scaler=None):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            x, lengths, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x, lengths)
            
            # 计算归一化空间中的损失（用于显示在进度条）
            loss = criterion(outputs.squeeze(), y)
            
            # 收集预测和目标值，用于后续可能的逆变换
            predictions.append(outputs.cpu())
            targets.append(y.cpu())
            
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': loss.item()})
    
    # 如果提供了y_scaler，则在原始空间计算损失
    if y_scaler is not None:
        # 合并所有预测和目标
        all_preds = torch.cat(predictions)
        all_targets = torch.cat(targets)
        
        # 逆变换回原始空间
        orig_preds = torch.tensor(y_scaler.inverse_transform(
            all_preds.numpy().reshape(-1, 1))).squeeze()
        orig_targets = torch.tensor(y_scaler.inverse_transform(
            all_targets.numpy().reshape(-1, 1))).squeeze()
        
        # 计算原始空间的损失
        orig_loss = criterion(orig_preds, orig_targets)
        
        # 返回两种损失
        norm_loss = running_loss / len(dataloader.dataset)
        return norm_loss, orig_loss.item()
    
    # 否则仅返回归一化空间的损失
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

class TransformedSubset(torch.utils.data.Dataset):
    """
    包装一个 subset（通常由 random_split 返回的 Subset），
    在 __getitem__ 中对 (x, y) 施加 transform(x)。
    """
    def __init__(self, subset, transform=None):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def main():
    # 超参数设置
    N = 12000
    tasks = 1
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001

    # 1) 获取数据
    X, Y = load_andi_data(N, tasks)

    # 2) 用 X 来 fit scaler
    scaler = create_and_fit_scaler(X)
    standardize = StandardScalerTransform(scaler)
    y_scaler = StandardScaler()
    y_scaler.fit(np.array(Y).reshape(-1, 1))
    # 3) 构造原始 dataset（不带 transform）
    raw_dataset = AnDiDataset(X, Y, transform=None)

    # 4) random_split 切分成 train/val/test
    train_size = int(0.8 * len(raw_dataset))
    val_size = int(0.1 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size - val_size
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size]
    )

    # 5) 只在训练集上做「标准化 + 时间反转」，验证/测试只做「标准化」
    train_transform = standardize
    val_transform = standardize
    test_transform = standardize

    # 6) 用包装器给 subset 加上不同 transforms
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset   = TransformedSubset(val_subset,   transform=val_transform)
    test_dataset  = TransformedSubset(test_subset,  transform=test_transform)

    # 7) 构造 DataLoader，添加标签归一化
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=lambda batch: collate_fn(batch, y_scaler))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=lambda batch: collate_fn(batch, y_scaler))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, collate_fn=lambda batch: collate_fn(batch, y_scaler))

    # 8) 定义模型、损失、优化器
    # 使用改进后的模型 ImprovedCNNBiLSTM，注意参数可根据任务调优
    model = ImprovedCNNBiLSTM(
        conv_channels=256,         # 基础卷积通道数
        lstm_hidden_size=512,      # LSTM 隐状态维度
        lstm_layers=4,            # 使用4层LSTM
        bidirectional=True,
        dropout_rate=0.5,
        kernel_sizes=[3, 5, 7]    # 多尺度卷积核尺寸
    )
    criterion = nn.L1Loss()  # MAE损失
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    orig_val_losses = []  # 用于记录原始空间的验证损失
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 获取训练结果，并提取损失值
        train_result = train(model, train_loader, criterion, optimizer, device)
        train_loss = train_result[0] if isinstance(train_result, tuple) else train_result
        
        # 获取验证结果，并提取损失值
        val_result = evaluate(model, val_loader, criterion, device, y_scaler=y_scaler)
        val_loss = val_result[0] if isinstance(val_result, tuple) else val_result
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 如果有原始空间损失，记录并打印
        if isinstance(val_result, tuple) and len(val_result) > 1:
            orig_val_loss = val_result[1]
            orig_val_losses.append(orig_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Training Loss: {train_loss:.4f}, "
                  f"Validation Loss (Norm): {val_loss:.4f}, "
                  f"Validation Loss (Orig): {orig_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Training Loss: {train_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}")
        
        # Early Stopping 逻辑保持不变 (使用归一化损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger_times += 1
            print(f"patience {trigger_times}")
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # 绘制训练和验证 Loss 曲线 (归一化空间)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0, top=0.95) 
    plt.legend()
    plt.grid(True)
    plt.savefig("norm_loss_curve.png")
    plt.show()

    # 如果有原始空间损失，绘制原始空间损失曲线
    if orig_val_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(orig_val_losses) + 1), orig_val_losses, label='Validation Loss (Original Scale)')
        plt.title('Validation Loss in Original Scale')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("orig_loss_curve.png")
        plt.show()
    
    # 加载最优模型并评估
    best_model = ImprovedCNNBiLSTM(
        conv_channels=256,
        lstm_hidden_size=512,
        lstm_layers=4,
        bidirectional=True,
        dropout_rate=0.5,
        kernel_sizes=[3, 5, 7]
    )
    best_model.load_state_dict(torch.load("best_model.pt"))
    best_model.to(device)
    
    # 评估并显示两种空间的损失
    test_result = evaluate(best_model, test_loader, criterion, device, y_scaler=y_scaler)
    test_norm_loss = test_result[0] if isinstance(test_result, tuple) else test_result
    
    print(f"\n[Final Test] Normalized Loss: {test_norm_loss:.4f}")
    
    # 如果有原始空间测试损失，也显示出来
    if isinstance(test_result, tuple) and len(test_result) > 1:
        test_orig_loss = test_result[1]
        print(f"[Final Test] Original Scale Loss: {test_orig_loss:.4f}")

if __name__ == '__main__':
    main()
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import argparse
from sklearn.metrics import mean_squared_error, r2_score
import logging
from tqdm import tqdm

# 导入自定义模块
from dataset import load_andi_data, TrajectoryDataset, load_clean_and_noisy_data, split_dataset
from model import CNNBiLSTMDiffusionExponent, AnomalousDiffusionExponentModel, calculate_anomalous_diffusion_metrics

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log",encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='异常扩散指数预测训练脚本')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--n_trajectories', type=int, default=50, help='每个模型的轨迹数量')
    parser.add_argument('--trajectory_length', type=int, default=200, help='轨迹长度')
    parser.add_argument('--segment_length', type=int, default=100, help='输入段长度')
    parser.add_argument('--prediction_length', type=int, default=100, help='预测长度')
    parser.add_argument('--timesteps', type=int, default=500, help='扩散过程中的时间步数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    logger.info(f"模型保存在 {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"从 epoch {epoch} 加载模型，损失为 {loss:.4f}")
    return epoch, loss

def evaluate(model, dataloader, device):
    """评估模型在数据集上的性能"""
    model.model.eval()
    all_true_exponents = []
    all_pred_exponents = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            segments, _, true_exponents, _ = batch  # 忽略target和model_id
            segments = segments.to(device)
            true_exponents = true_exponents.to(device)
            
            # 预测扩散指数
            pred_exponents = model.predict_exponent(segments)
            
            # 收集真实和预测的指数值
            all_true_exponents.append(true_exponents.cpu().numpy())
            all_pred_exponents.append(pred_exponents.cpu().numpy())
            
            # 计算MSE损失
            loss = nn.MSELoss()(pred_exponents, true_exponents)
            total_loss += loss.item() * segments.size(0)
    
    # 组合所有批次的结果
    all_true_exponents = np.concatenate(all_true_exponents)
    all_pred_exponents = np.concatenate(all_pred_exponents)
    
    # 计算指标
    avg_loss = total_loss / len(dataloader.dataset)
    rmse = np.sqrt(mean_squared_error(all_true_exponents, all_pred_exponents))
    r2 = r2_score(all_true_exponents, all_pred_exponents)
    
    model.model.train()
    return avg_loss, rmse, r2, all_true_exponents, all_pred_exponents

def plot_results(true_exponents, pred_exponents, save_path):
    """绘制真实vs预测的指数图"""
    plt.figure(figsize=(10, 6))
    
    # 散点图：真实vs预测
    plt.scatter(true_exponents, pred_exponents, alpha=0.5)
    
    # 添加理想线 (y=x)
    min_val = min(np.min(true_exponents), np.min(pred_exponents))
    max_val = max(np.max(true_exponents), np.max(pred_exponents))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 标签和标题
    plt.xlabel('真实扩散指数')
    plt.ylabel('预测扩散指数')
    plt.title('异常扩散指数：真实 vs 预测')
    
    # 计算并显示统计信息
    rmse = np.sqrt(mean_squared_error(true_exponents, pred_exponents))
    r2 = r2_score(true_exponents, pred_exponents)
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"结果图保存在 {save_path}")

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    logger.info(f"使用设备: {args.device}")
    logger.info(f"参数: {vars(args)}")
    
    # 加载数据
    logger.info("加载ANDI数据集...")
    
    # 方式1：分别加载干净数据和噪声数据
    trajectories_clean, exponents_clean, model_ids_clean = load_andi_data(
        args.n_trajectories, args.trajectory_length, add_noise=False)

    trajectories_noisy, exponents_noisy, model_ids_noisy = load_andi_data(
        args.n_trajectories, args.trajectory_length, add_noise=True, noise_level=0.1)


    # 为扩散模型训练创建数据集（使用干净数据）
    train_dataset = TrajectoryDataset(
        trajectories=trajectories_clean, 
        exponents=exponents_clean,
        model_ids=model_ids_clean,
        segment_length=args.segment_length,
        prediction_length=args.prediction_length,
        normalize=True
    )

    # 使用新函数分割数据集
    train_dataset, val_dataset, test_dataset = split_dataset(
        train_dataset, 
        test_size=args.test_size, 
        val_size=args.val_size,
        seed=args.seed
    )

    # 在此处添加：为验证和测试创建噪声数据集
    noisy_dataset = TrajectoryDataset(
        trajectories=trajectories_noisy, 
        exponents=exponents_noisy,
        model_ids=model_ids_noisy,
        segment_length=args.segment_length,
        prediction_length=args.prediction_length,
        normalize=True
    )
    
    # 分割噪声数据集（使用相同的种子确保分割方式一致）
    _, val_dataset_noisy, test_dataset_noisy = split_dataset(
        noisy_dataset, 
        test_size=args.test_size, 
        val_size=args.val_size,
        seed=args.seed
    )
    
    # 对验证和测试集使用噪声数据
    val_dataset = val_dataset_noisy
    test_dataset = test_dataset_noisy

    logger.info(f"数据集分割: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    device = torch.device(args.device)
    model_backbone = CNNBiLSTMDiffusionExponent(
        input_size=1,
        conv_channels=128,
        kernel_sizes=[3, 5, 7],
        num_residual_blocks=2,
        lstm_hidden_size=128,
        lstm_layers=2,
        bidirectional=True,
        dropout_rate=0.3,
        time_embedding_dim=64,
        sequence_length=args.segment_length,
        exponent_range=(0.05, 2.0)  # 数据集的指数范围
    ).to(device)
    
    # 创建扩散模型包装器
    diffusion_model = AnomalousDiffusionExponentModel(
        model=model_backbone,
        beta_schedule='cosine',
        timesteps=args.timesteps,
        device=device
    )
    
    # 创建优化器
    optimizer = optim.AdamW(model_backbone.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 如果恢复训练，加载检查点
    if args.resume and args.checkpoint:
        logger.info(f"从检查点恢复训练: {args.checkpoint}")
        start_epoch, _ = load_checkpoint(diffusion_model, optimizer, args.checkpoint)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_r2': [],
        'lr': []
    }
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model_backbone.train()
        train_losses = []
        denoising_losses = []
        exponent_losses = []
        
        epoch_start_time = time.time()
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, batch in train_pbar:
            segments, targets, exponent_targets, _ = batch  # 忽略model_id
            segments = segments.to(device)
            targets = targets.to(device)
            exponent_targets = exponent_targets.to(device)
            
            optimizer.zero_grad()
            
            # 训练步骤
            total_loss, denoising_loss, exponent_loss = diffusion_model.train_step(segments, exponent_targets)
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_backbone.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses.append(total_loss.item())
            denoising_losses.append(denoising_loss.item())
            exponent_losses.append(exponent_loss.item())
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'denoise_loss': f"{denoising_loss.item():.4f}",
                'exp_loss': f"{exponent_loss.item():.4f}"
            })
        
        # 计算平均训练损失
        avg_train_loss = np.mean(train_losses)
        avg_denoising_loss = np.mean(denoising_losses)
        avg_exponent_loss = np.mean(exponent_losses)
        
        # 评估验证集
        val_loss, val_rmse, val_r2, _, _ = evaluate(diffusion_model, val_loader, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)
        
        # 输出信息
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Time: {epoch_time:.2f}s - "
                   f"Train Loss: {avg_train_loss:.4f} - "
                   f"Denoise Loss: {avg_denoising_loss:.4f} - "
                   f"Exponent Loss: {avg_exponent_loss:.4f} - "
                   f"Val Loss: {val_loss:.4f} - "
                   f"Val RMSE: {val_rmse:.4f} - "
                   f"Val R²: {val_r2:.4f} - "
                   f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                diffusion_model, 
                optimizer, 
                epoch, 
                val_loss, 
                os.path.join(args.save_dir, 'best_model.pth')
            )
            logger.info(f"保存了新的最佳模型，验证损失: {val_loss:.4f}")
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                diffusion_model, 
                optimizer, 
                epoch, 
                avg_train_loss, 
                os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth')
            )
    
    # 训练完成后保存最终模型
    save_checkpoint(
        diffusion_model, 
        optimizer, 
        args.epochs - 1, 
        avg_train_loss, 
        os.path.join(args.save_dir, 'final_model.pth')
    )
    
    logger.info("训练完成！")
    
    # 绘制训练历史
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['val_rmse'], label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Validation RMSE')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['val_r2'], label='R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.title('Validation R²')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    plt.close()
    
    # 评估测试集
    logger.info("在测试集上评估最佳模型...")
    
    # 加载最佳模型
    load_checkpoint(diffusion_model, None, os.path.join(args.save_dir, 'best_model.pth'))
    
    # 在测试集上评估
    test_loss, test_rmse, test_r2, true_exponents, pred_exponents = evaluate(diffusion_model, test_loader, device)
    
    logger.info(f"测试集结果: Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    # 绘制测试结果
    plot_results(true_exponents, pred_exponents, os.path.join(args.save_dir, 'test_results.png'))
    
    # 示例：生成具有特定扩散指数的轨迹
    logger.info("生成具有特定扩散指数的样本轨迹...")
    
    # 选择几个不同的目标指数进行演示
    target_exponents = [0.2, 0.5, 1.0, 1.5, 1.8]
    
    plt.figure(figsize=(15, 10))
    for i, target_exp in enumerate(target_exponents):
        # 生成具有目标扩散指数的轨迹
        target_exp_tensor = torch.tensor([target_exp], device=device).view(1, 1)
        sample_shape = (1, args.segment_length, 1)  # 单个样本
        
        generated_traj, achieved_exp = diffusion_model.sample(
            sample_shape, 
            guide_with_exponent=target_exp_tensor,
            num_steps=100  # 使用更少的步骤进行演示
        )
        
        # 转换为numpy并展平以便绘图
        generated_traj = generated_traj.cpu().numpy().squeeze()
        achieved_exp = achieved_exp.cpu().numpy().squeeze()
        
        # 绘制生成的轨迹
        plt.subplot(len(target_exponents), 1, i+1)
        plt.plot(generated_traj)
        plt.title(f'目标α={target_exp:.2f}, 实现α={achieved_exp:.2f}')
        plt.ylabel('位置')
        if i == len(target_exponents) - 1:
            plt.xlabel('时间步')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'generated_trajectories.png'))
    plt.close()

    logger.info("实验完成！")

if __name__ == "__main__":
    main()
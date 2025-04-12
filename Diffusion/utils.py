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
from dataset import load_andi_data, TrajectoryDataset, load_clean_and_noisy_data, split_dataset
from model import CNNBiLSTMDiffusionExponent, AnomalousDiffusionExponentModel, calculate_anomalous_diffusion_metrics, evaluate_with_denoising


logger = logging.getLogger(__name__)


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
    checkpoint = torch.load(checkpoint_path,weights_only=True)
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
            true_exponents_reshaped = true_exponents.view(pred_exponents.shape)
            loss = nn.MSELoss()(pred_exponents, true_exponents_reshaped)
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

def plot_comparison_results(true_exponents, direct_pred_exponents, denoised_pred_exponents, save_path):
    """绘制直接预测和去噪预测的比较图"""
    plt.figure(figsize=(15, 6))
    
    # 左图：散点图比较
    plt.subplot(1, 2, 1)
    plt.scatter(true_exponents, direct_pred_exponents, alpha=0.5, label='直接预测')
    plt.scatter(true_exponents, denoised_pred_exponents, alpha=0.5, label='去噪后预测')
    
    # 添加理想线 (y=x)
    min_val = min(np.min(true_exponents), np.min(direct_pred_exponents), np.min(denoised_pred_exponents))
    max_val = max(np.max(true_exponents), np.max(direct_pred_exponents), np.max(denoised_pred_exponents))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('真实扩散指数')
    plt.ylabel('预测扩散指数')
    plt.title('预测方法比较')
    plt.legend()
    
    # 右图：误差箱形图比较
    plt.subplot(1, 2, 2)
    direct_errors = np.abs(direct_pred_exponents - true_exponents)
    denoised_errors = np.abs(denoised_pred_exponents - true_exponents)
    
    box_data = [direct_errors, denoised_errors]
    plt.boxplot(box_data, labels=['直接预测', '去噪后预测'])
    plt.ylabel('绝对误差')
    plt.title('预测误差比较')
    
    # 添加统计数据
    direct_rmse = np.sqrt(mean_squared_error(true_exponents, direct_pred_exponents))
    direct_r2 = r2_score(true_exponents, direct_pred_exponents)
    denoised_rmse = np.sqrt(mean_squared_error(true_exponents, denoised_pred_exponents))
    denoised_r2 = r2_score(true_exponents, denoised_pred_exponents)
    
    plt.figtext(0.5, 0.01, 
                f'直接预测: RMSE={direct_rmse:.4f}, R²={direct_r2:.4f}\n去噪后预测: RMSE={denoised_rmse:.4f}, R²={denoised_r2:.4f}',
                ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"比较结果图保存在 {save_path}")

def visualize_denoising_examples(model, dataloader, device, save_dir, num_batches=2, samples_per_batch=3):
    """可视化一些去噪样本的效果"""
    model.model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:  
                break
                
            segments, _, true_exponents, _ = batch
            segments = segments.to(device)
            true_exponents = true_exponents.to(device)
            
            # 直接预测
            direct_pred = model.predict_exponent(segments)
            
            # 先去噪再预测
            denoised_pred, denoised_segments = model.predict_exponent_with_denoising(segments, denoise_steps=50)
            
            # 绘制样本去噪效果
            fig, axes = plt.subplots(samples_per_batch, 2, figsize=(12, 4*samples_per_batch))
            
            for j in range(min(samples_per_batch, segments.size(0))):
                # 原始噪声轨迹
                axes[j, 0].plot(segments[j].cpu().numpy())
                axes[j, 0].set_title(
                    f'噪声轨迹 (真实α={true_exponents[j].item():.2f}, 预测α={direct_pred[j].item():.2f})'
                )
                
                # 去噪后轨迹
                axes[j, 1].plot(denoised_segments[j].cpu().numpy())
                axes[j, 1].set_title(
                    f'去噪后轨迹 (预测α={denoised_pred[j].item():.2f})'
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'denoising_examples_batch_{i}.png'))
            plt.close()
    
    logger.info(f"去噪样本可视化已保存到 {save_dir}")
    
def save_test_dataset(dataset, filename):
        data_dict = {
            'trajectories': [],
            'exponents': [],
            'model_ids': []
        }
    
        # 收集所有数据
        for i in range(len(dataset)):
            segment, target, exponent, model_id = dataset[i]
            data_dict['trajectories'].append(segment.numpy())
            data_dict['exponents'].append(exponent.numpy())
            data_dict['model_ids'].append(model_id)
    
        # 转换为numpy数组
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])
    
        # 保存为numpy文件
        np.save(filename, data_dict)
        logger.info(f"测试数据已保存至: {filename}")

def load_saved_test_data(filepath, segment_length=200, prediction_length=0, normalize=True):
    """加载保存的测试数据集"""
    print(f"加载保存的测试数据: {filepath}")
    data_dict = np.load(filepath, allow_pickle=True).item()
    
    # 创建数据集对象
    test_dataset = TrajectoryDataset(
        trajectories=data_dict['trajectories'],
        exponents=data_dict['exponents'],
        model_ids=data_dict['model_ids'],
        segment_length=segment_length,
        prediction_length=prediction_length,
        normalize=normalize
    )
    
    return test_dataset
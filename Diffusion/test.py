import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from scipy.stats import gaussian_kde  # 核密度估计
from utils import load_saved_test_data

# 导入扩散模型所需的模块
from dataset import load_andi_data, TrajectoryDataset, split_dataset
from model import CNNBiLSTMDiffusionExponent, AnomalousDiffusionExponentModel, evaluate_with_denoising
from utils import plot_results, plot_comparison_results, visualize_denoising_examples, load_checkpoint, set_seed

def analyze_by_region(y_true, y_pred):
    """按真实α值区域分析错误"""
    print("\n===== 分区域性能分析 =====")
    # 定义α值范围
    ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    
    for min_val, max_val in ranges:
        # 创建掩码选择当前范围内的数据
        mask = (min_val <= y_true) & (y_true < max_val)
        count = np.sum(mask)
        
        # 跳过没有数据的范围
        if count == 0:
            continue
            
        # 计算该区域的错误指标
        region_mse = np.mean((y_pred[mask] - y_true[mask]) ** 2)
        region_mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
        region_r2 = 1 - np.sum((y_pred[mask] - y_true[mask]) ** 2) / np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
        
        # 打印结果
        print(f"α范围 {min_val:.1f}-{max_val:.1f}:")
        print(f"  • 样本数量: {count} (占比: {count/len(y_true):.1%})")
        print(f"  • MSE: {region_mse:.6f}")
        print(f"  • MAE: {region_mae:.6f}")
        print(f"  • R²: {region_r2:.6f}")

def analyze_outliers(y_true, y_pred, threshold=2.0):
    """识别和分析异常大的预测错误"""
    # 计算误差
    errors = y_pred - y_true
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    
    # 定义离群值（误差超过阈值标准差）
    outlier_mask = np.abs(errors - error_mean) > threshold * error_std
    outlier_count = np.sum(outlier_mask)
    
    print("\n===== 离群值分析 =====")
    print(f"误差标准差: {error_std:.6f}")
    print(f"离群值定义: |error - {error_mean:.6f}| > {threshold:.1f} * {error_std:.6f}")
    print(f"离群值数量: {outlier_count} (占比: {outlier_count/len(errors):.2%})")
    
    # 如果存在离群值，分析其特征
    if outlier_count > 0:
        outlier_true = y_true[outlier_mask]
        outlier_pred = y_pred[outlier_mask]
        outlier_err = errors[outlier_mask]
        
        print("\n离群值统计:")
        print(f"  • 真实α值: 最小={np.min(outlier_true):.2f}, 最大={np.max(outlier_true):.2f}, 平均={np.mean(outlier_true):.2f}")
        print(f"  • 预测α值: 最小={np.min(outlier_pred):.2f}, 最大={np.max(outlier_pred):.2f}, 平均={np.mean(outlier_pred):.2f}")
        print(f"  • 误差范围: 最小={np.min(outlier_err):.2f}, 最大={np.max(outlier_err):.2f}")
        
        # 分析离群值在各α范围的分布
        print("\n离群值在各α范围的分布:")
        ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
        for min_val, max_val in ranges:
            range_mask = (min_val <= outlier_true) & (outlier_true < max_val)
            range_count = np.sum(range_mask)
            if range_count > 0:
                total_in_range = np.sum((min_val <= y_true) & (y_true < max_val))
                print(f"  • α范围 {min_val:.1f}-{max_val:.1f}: {range_count} 个离群值 ({range_count/total_in_range:.1%} 的该范围样本)")
    
    return outlier_mask

def plot_error_distribution(y_true, y_pred):
    """绘制预测误差分布"""
    # 计算误差
    errors = y_pred - y_true
    
    # 创建新图形
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    
    # 绘制带密度曲线的误差直方图
    sns.histplot(errors, kde=True, bins=30, color='royalblue', alpha=0.7)
    
    # 添加零误差垂直线 - 改为英文
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    
    # 添加标记误差统计的垂直线 - 改为英文
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    plt.axvline(mean_err, color='g', linestyle='-', linewidth=2, label=f'Mean Error: {mean_err:.4f}')
    plt.axvline(mean_err + std_err, color='orange', linestyle=':', linewidth=1.5, label=f'+1σ: {mean_err+std_err:.4f}')
    plt.axvline(mean_err - std_err, color='orange', linestyle=':', linewidth=1.5, label=f'-1σ: {mean_err-std_err:.4f}')
    
    # 添加标签和标题 - 改为英文
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Diffusion Exponent Prediction Errors')
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # 添加误差统计文本框 - 改为英文
    textstr = '\n'.join((
        f'Mean = {mean_err:.4f}',
        f'Std Dev = {std_err:.4f}',
        f'Median = {np.median(errors):.4f}',
        f'Skewness = {np.mean(((errors - mean_err)/std_err)**3):.4f}',
        f'Kurtosis = {np.mean(((errors - mean_err)/std_err)**4) - 3:.4f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.03, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=props)
    
    # 保存和显示图形
    plt.tight_layout()
    plt.savefig('diffusion_alpha_error_dist.png', dpi=300, bbox_inches='tight')
    print("Error distribution plot saved as 'diffusion_alpha_error_dist.png'")
    plt.show()

def plot_ground_truth_alpha(alpha_values):
    """绘制真实alpha值的分布"""
    # 首先将列表转换为NumPy数组
    alpha_values = np.array(alpha_values)
    
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    
    # 绘制直方图并添加密度曲线
    sns.histplot(alpha_values, bins=20, kde=True, color='forestgreen', alpha=0.7)
    
    # 添加标签和标题 - 改为英文
    plt.xlabel('Diffusion Exponent (α)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ground Truth α Values in Test Set')
    
    # 计算并显示统计数据
    mean_alpha = np.mean(alpha_values)
    std_alpha = np.std(alpha_values)
    
    # 添加标记线 - 改为英文
    plt.axvline(mean_alpha, color='red', linestyle='-', linewidth=2, 
                label=f'Mean: {mean_alpha:.4f}')
    plt.axvline(mean_alpha + std_alpha, color='orange', linestyle=':', linewidth=1.5, 
                label=f'+1σ: {mean_alpha+std_alpha:.4f}')
    plt.axvline(mean_alpha - std_alpha, color='orange', linestyle=':', linewidth=1.5, 
                label=f'-1σ: {mean_alpha-std_alpha:.4f}')
    
    # 添加统计数据文本框 - 改为英文
    textstr = '\n'.join((
        f'Count = {len(alpha_values)}',
        f'Mean = {mean_alpha:.4f}',
        f'Std Dev = {std_alpha:.4f}',
        f'Min = {np.min(alpha_values):.4f}',
        f'Max = {np.max(alpha_values):.4f}',
        f'Median = {np.median(alpha_values):.4f}'))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.03, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=props)
    
    # 添加各α值区间的分布统计 - 改为英文
    ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    range_stats = []
    
    for min_val, max_val in ranges:
        count = np.sum((min_val <= alpha_values) & (alpha_values < max_val))
        if count > 0:
            percentage = count/len(alpha_values)*100
            range_stats.append(f"{min_val:.1f}-{max_val:.1f}: {count} ({percentage:.1f}%)")
    
    range_text = '\n'.join(range_stats)
    plt.text(0.75, 0.95, range_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=props)
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # 保存和显示图形
    plt.tight_layout()
    plt.savefig('diffusion_ground_truth_alpha_dist.png', dpi=300, bbox_inches='tight')
    print("Ground Truth α distribution saved as 'diffusion_ground_truth_alpha_dist.png'")
    
    plt.show()

def plot_by_model_type(y_true, y_pred, model_ids):
    """绘制不同模型类型的散点图比较"""
    # 模型名称映射
    model_names = {
        0: "AATM",
        1: "CTRW",  
        2: "FBM",
        3: "LW",
        4: "SBM"
    }
    
    # 分析不同模型类型的性能
    print("\n===== Performance Analysis by Diffusion Model Type =====")
    unique_models = np.unique(model_ids)
    
    # 准备画图
    plt.figure(figsize=(12, 10))
    
    # 对每种模型类型计算性能指标并绘制
    for model_id in unique_models:
        mask = model_ids == model_id
        model_true = y_true[mask]
        model_pred = y_pred[mask]
        
        # 计算性能指标
        mse = np.mean((model_pred - model_true) ** 2)
        mae = np.mean(np.abs(model_pred - model_true))
        
        # 打印结果
        model_name = model_names.get(model_id, f"Model{model_id}")
        print(f"{model_name} (n={np.sum(mask)}):")
        print(f"  • MSE: {mse:.6f}")
        print(f"  • MAE: {mae:.6f}")
        
        # 绘制散点
        plt.scatter(model_true, model_pred, 
                   label=f"{model_name} (n={np.sum(mask)})",
                   alpha=0.7, s=60)
    
    # 添加图例和标签 - 改为英文
    plt.xlabel('True α')
    plt.ylabel('Predicted α')
    plt.title('Diffusion Exponent Prediction Performance by Model Type')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 添加对角线（理想预测）
    min_val = min(np.min(y_true), np.min(y_pred)) * 0.9
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig('diffusion_prediction_by_model.png', dpi=300)
    plt.show()

def plot_direct_vs_denoised(true_values, direct_preds, denoised_preds):
    """比较直接预测和先去噪再预测的性能"""
    plt.figure(figsize=(14, 6))
    
    # 左图：散点图和拟合直线
    plt.subplot(1, 2, 1)
    plt.scatter(true_values, direct_preds, alpha=0.5, label='Direct Prediction', color='blue')
    plt.scatter(true_values, denoised_preds, alpha=0.5, label='After Denoising', color='red')
    
    # 添加理想线（y=x）
    min_val = min(np.min(true_values), np.min(direct_preds), np.min(denoised_preds)) * 0.9
    max_val = max(np.max(true_values), np.max(direct_preds), np.max(denoised_preds)) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    # 添加拟合线（直接预测） - 改为英文
    z_coef_direct = np.polyfit(true_values, direct_preds, 1)
    p_direct = np.poly1d(z_coef_direct)
    xx = np.linspace(min_val, max_val, 100)
    plt.plot(xx, p_direct(xx), 'b-', linewidth=1, alpha=0.7,
             label=f"Direct Fit (y={z_coef_direct[0]:.2f}x+{z_coef_direct[1]:.2f})")
    
    # 添加拟合线（去噪后预测） - 改为英文
    z_coef_denoised = np.polyfit(true_values, denoised_preds, 1)
    p_denoised = np.poly1d(z_coef_denoised)
    plt.plot(xx, p_denoised(xx), 'r-', linewidth=1, alpha=0.7,
             label=f"Denoised Fit (y={z_coef_denoised[0]:.2f}x+{z_coef_denoised[1]:.2f})")
    
    plt.xlabel('True α')
    plt.ylabel('Predicted α')
    plt.title('Direct Prediction vs. After Denoising')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 右图：误差箱线图
    plt.subplot(1, 2, 2)
    direct_errors = np.abs(direct_preds - true_values)
    denoised_errors = np.abs(denoised_preds - true_values)
    
    box_data = [direct_errors, denoised_errors]
    plt.boxplot(box_data, labels=['Direct Prediction', 'After Denoising'])
    plt.ylabel('Absolute Error')
    plt.title('Prediction Error Comparison')
    
    # 添加统计数据 - 改为英文
    direct_mse = np.mean((direct_preds - true_values) ** 2)
    direct_mae = np.mean(direct_errors)
    denoised_mse = np.mean((denoised_preds - true_values) ** 2)
    denoised_mae = np.mean(denoised_errors)
    
    stat_text = '\n'.join([
        f"Direct: MSE={direct_mse:.6f}, MAE={direct_mae:.6f}",
        f"Denoised: MSE={denoised_mse:.6f}, MAE={denoised_mae:.6f}",
        f"Improvement: {(1 - denoised_mse/direct_mse)*100:.2f}%"
    ])
    
    plt.figtext(0.5, 0.01, stat_text, ha="center", fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('diffusion_direct_vs_denoised.png', dpi=300)
    plt.show()

def create_density_scatter_plot(y_true, y_pred, filename='diffusion_density_scatter.png'):
    """创建密度散点图"""
    # 1) 为每个点计算2D核密度
    x = y_true
    y = y_pred
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # 每个点的密度值
    
    # 2) 按密度排序确保高密度点在顶部
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 14})
    sns.set_style("whitegrid")
    
    # 3) 创建基于密度的散点图
    scatter = plt.scatter(
        x, y, 
        c=z,            # 使用核密度值作为颜色
        cmap='jet',     # 色图选项: 'viridis', 'plasma', 'jet' 等
        s=50,           # 点大小
        alpha=0.8       # 透明度
    )
    
    # 添加颜色条 - 改为英文
    cbar = plt.colorbar(scatter)
    cbar.set_label('Probability Density')
    
    # 计算轴范围
    min_val = min(np.min(x), np.min(y)) * 0.9
    max_val = max(np.max(x), np.max(y)) * 1.1
    
    # 添加对角线（理想预测） - 改为英文
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Ideal Prediction")
    
    # 添加最佳拟合线 - 改为英文
    z_coef = np.polyfit(x, y, 1)
    p = np.poly1d(z_coef)
    xx = np.linspace(min_val, max_val, 200)
    plt.plot(xx, p(xx), "g-", linewidth=1.5, 
             label=f"Best Fit (y={z_coef[0]:.2f}x+{z_coef[1]:.2f})")
    
    # 添加±10%误差线 - 改为英文
    plt.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'r:', linewidth=1, alpha=0.6, label="+10%")
    plt.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'r:', linewidth=1, alpha=0.6, label="-10%")
    
    # 添加标签和标题 - 改为英文
    plt.xlabel('True Diffusion Exponent α')
    plt.ylabel('Predicted Diffusion Exponent α')
    plt.title('Diffusion Exponent Prediction Performance with Density Distribution')
    
    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # 添加性能指标文本框 - 改为英文
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    textstr = '\n'.join((
        f'MSE = {mse:.6f}',
        f'MAE = {mae:.6f}',
        f'RMSE = {rmse:.6f}',
        f'R² = {r2:.6f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Density scatter plot saved as '{filename}'")
    
    return mse, mae, rmse, r2

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='扩散模型测试脚本')
    parser.add_argument('--checkpoint', type=str, default='./Diffusion/checkpoints/best_model.pth', help='要测试的检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--n_trajectories', type=int, default=20, help='每种模型的轨迹数量')
    parser.add_argument('--trajectory_length', type=int, default=200, help='轨迹长度')
    parser.add_argument('--segment_length', type=int, default=200, help='输入段长度')
    parser.add_argument('--prediction_length', type=int, default=0, help='预测长度')
    parser.add_argument('--denoise_steps', type=int, default=100, help='去噪步骤数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--results_dir', type=str, default='./Diffusion/results', help='结果保存目录')
    parser.add_argument('--data_dir', type=str, default='./Diffusion/data', help='数据目录')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建结果保存目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    
    # 创建数据集
    test_data_path = os.path.join(args.data_dir, 'test_dataset', 'noisy_test_data.npy')
    if os.path.exists(test_data_path):
        test_dataset = load_saved_test_data(test_data_path)
        print(f"成功加载测试数据集: {len(test_dataset)} 个样本")
        
        # 提取模型ID
        model_ids = []
        for i in range(len(test_dataset)):
            _, _, _, model_id = test_dataset[i]
            model_ids.append(model_id.item())  # 将张量转换为标量
    else:
        # 如果找不到保存的数据，就使用原来的生成方法
        print("未找到保存的测试数据，重新生成...")
        trajectories_noisy, true_exponents, model_ids = load_andi_data(
            args.n_trajectories, args.trajectory_length, add_noise=True, noise_level=0.1)
        
        test_dataset = TrajectoryDataset(
        trajectories=trajectories_noisy,
        exponents=true_exponents,
        model_ids=model_ids,
        segment_length=args.segment_length,
        prediction_length=args.prediction_length,
        normalize=True
        )
        print(f"成功生成测试数据集: {len(test_dataset)} 个样本")

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 加载模型和检查点
    print(f"从 {args.checkpoint} 加载模型...")
    
    # 初始化模型结构
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
        exponent_range=(0.05, 2.0)
    ).to(device)
    
    # 创建扩散模型包装器
    diffusion_model = AnomalousDiffusionExponentModel(
        model=model_backbone,
        beta_schedule='cosine',
        timesteps=500,
        device=device
    )
    
    # 加载检查点
    _, _ = load_checkpoint(diffusion_model, None, args.checkpoint)
    
    # 评估模型 - 包括直接预测和去噪后预测的比较
    print("评估模型性能...")
    results = evaluate_with_denoising(diffusion_model, test_loader, device, denoise_steps=args.denoise_steps)
    
    # 提取结果
    direct_metrics = results['direct']  # (loss, rmse, r2, predictions)
    denoised_metrics = results['denoised']  # (loss, rmse, r2, predictions)
    true_exponents = results['true']
    
    # 打印直接预测结果
    direct_loss, direct_rmse, direct_r2, direct_preds = direct_metrics
    print("\n===== 直接预测性能 =====")
    print(f"MSE: {direct_loss:.6f}")
    print(f"RMSE: {direct_rmse:.6f}")
    print(f"R²: {direct_r2:.6f}")
    
    # 打印去噪后预测结果
    denoised_loss, denoised_rmse, denoised_r2, denoised_preds = denoised_metrics
    print("\n===== 去噪后预测性能 =====")
    print(f"MSE: {denoised_loss:.6f}")
    print(f"RMSE: {denoised_rmse:.6f}")
    print(f"R²: {denoised_r2:.6f}")
    
    print(f"\n去噪预测提升: {(1-denoised_loss/direct_loss)*100:.2f}%")
    
    # 绘制真实α值分布
    plot_ground_truth_alpha(true_exponents)
    
    # 绘制直接预测的密度散点图
    print("\n创建直接预测的密度散点图...")
    direct_file = os.path.join(args.results_dir, 'direct_prediction_density.png')
    create_density_scatter_plot(true_exponents, direct_preds, direct_file)
    
    # 绘制去噪后预测的密度散点图
    print("\n创建去噪后预测的密度散点图...")
    denoised_file = os.path.join(args.results_dir, 'denoised_prediction_density.png')
    create_density_scatter_plot(true_exponents, denoised_preds, denoised_file)
    
    # 绘制直接预测和去噪后预测的比较
    print("\n比较直接预测和去噪后预测...")
    plot_direct_vs_denoised(true_exponents, direct_preds, denoised_preds)
    
    # 按区域分析直接预测的性能
    print("\n按区域分析直接预测性能:")
    analyze_by_region(true_exponents, direct_preds)
    
    # 按区域分析去噪预测的性能
    print("\n按区域分析去噪后预测性能:")
    analyze_by_region(true_exponents, denoised_preds)
    
    # 分析直接预测的离群值
    print("\n分析直接预测的离群值:")
    analyze_outliers(true_exponents, direct_preds)
    
    # 分析去噪预测的离群值
    print("\n分析去噪后预测的离群值:")
    analyze_outliers(true_exponents, denoised_preds)
    
    # 按模型类型分析性能（如果有模型ID）
    if model_ids is not None and len(model_ids) > 0:
        print("\n按扩散模型类型分析直接预测性能:")
        model_ids_array = np.array([model_id for model_id in model_ids])
        plot_by_model_type(true_exponents, direct_preds, model_ids_array)
        
        print("\n按扩散模型类型分析去噪后预测性能:")
        plot_by_model_type(true_exponents, denoised_preds, model_ids_array)
    
    # 绘制预测误差分布
    print("\n绘制预测误差分布:")
    plot_error_distribution(true_exponents, direct_preds)
    plot_error_distribution(true_exponents, denoised_preds)
    
    # 可视化去噪效果
    print("\n可视化去噪效果...")
    visualize_denoising_examples(diffusion_model, test_loader, device, args.results_dir)
    
    print(f"\n测试完成！结果保存在 {args.results_dir} 目录")

if __name__ == "__main__":
    main()
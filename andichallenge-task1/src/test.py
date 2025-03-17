import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from scipy.stats import gaussian_kde  # For kernel density estimation

# Import necessary modules
from dataset import (
    AnDiDataset, 
    collate_fn,
    test_collate_fn,  # 添加逗号
    create_and_fit_scaler, 
    StandardScalerTransform, 
    load_andi_data,
)
from model import EquilenCNNBiLSTMAttention

def analyze_by_region(y_true, y_pred):
    """Analyze errors by region of true α values"""
    print("\n===== Regional Performance Analysis =====")
    # Define α value ranges for analysis
    ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    
    for min_val, max_val in ranges:
        # Create mask to select data in current range
        mask = (min_val <= y_true) & (y_true < max_val)
        count = np.sum(mask)
        
        # Skip ranges with no data
        if count == 0:
            continue
            
        # Calculate error metrics for this range
        region_mse = np.mean((y_pred[mask] - y_true[mask]) ** 2)
        region_mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
        region_r2 = 1 - np.sum((y_pred[mask] - y_true[mask]) ** 2) / np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
        
        # Print results
        print(f"α range {min_val:.1f}-{max_val:.1f}:")
        print(f"  • Sample count: {count} (ratio: {count/len(y_true):.1%})")
        print(f"  • MSE: {region_mse:.6f}")
        print(f"  • MAE: {region_mae:.6f}")
        print(f"  • R²: {region_r2:.6f}")

def analyze_outliers(y_true, y_pred, threshold=2.0):
    """Identify and analyze outliers with abnormally large prediction errors"""
    # Calculate errors
    errors = y_pred - y_true
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    
    # Define outliers (errors exceeding threshold standard deviations)
    outlier_mask = np.abs(errors - error_mean) > threshold * error_std
    outlier_count = np.sum(outlier_mask)
    
    print("\n===== Outlier Analysis =====")
    print(f"Error std dev: {error_std:.6f}")
    print(f"Outlier definition: |error - {error_mean:.6f}| > {threshold:.1f} * {error_std:.6f}")
    print(f"Outlier count: {outlier_count} (ratio: {outlier_count/len(errors):.2%})")
    
    # If outliers exist, analyze their characteristics
    if outlier_count > 0:
        outlier_true = y_true[outlier_mask]
        outlier_pred = y_pred[outlier_mask]
        outlier_err = errors[outlier_mask]
        
        print("\nOutlier statistics:")
        print(f"  • True α values: min={np.min(outlier_true):.2f}, max={np.max(outlier_true):.2f}, mean={np.mean(outlier_true):.2f}")
        print(f"  • Predicted α values: min={np.min(outlier_pred):.2f}, max={np.max(outlier_pred):.2f}, mean={np.mean(outlier_pred):.2f}")
        print(f"  • Error range: min={np.min(outlier_err):.2f}, max={np.max(outlier_err):.2f}")
        
        # Analyze outlier distribution across α ranges
        print("\nOutlier distribution by α range:")
        ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
        for min_val, max_val in ranges:
            range_mask = (min_val <= outlier_true) & (outlier_true < max_val)
            range_count = np.sum(range_mask)
            if range_count > 0:
                total_in_range = np.sum((min_val <= y_true) & (y_true < max_val))
                print(f"  • α range {min_val:.1f}-{max_val:.1f}: {range_count} outliers ({range_count/total_in_range:.1%} of samples in this range)")
    
    return outlier_mask

def plot_error_distribution(y_true, y_pred):
    """Plot distribution of prediction errors"""
    # Calculate errors
    errors = y_pred - y_true
    
    # Create new figure
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    
    # Plot error histogram with density curve
    sns.histplot(errors, kde=True, bins=30, color='royalblue', alpha=0.7)
    
    # Add vertical line at zero error
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    
    # Add vertical lines marking error statistics
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    plt.axvline(mean_err, color='g', linestyle='-', linewidth=2, label=f'Mean Error: {mean_err:.4f}')
    plt.axvline(mean_err + std_err, color='orange', linestyle=':', linewidth=1.5, label=f'+1σ: {mean_err+std_err:.4f}')
    plt.axvline(mean_err - std_err, color='orange', linestyle=':', linewidth=1.5, label=f'-1σ: {mean_err-std_err:.4f}')
    
    # Add labels and title
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Diffusion Exponent Prediction Error Distribution')
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Add error statistics text box
    textstr = '\n'.join((
        f'Mean = {mean_err:.4f}',
        f'Std Dev = {std_err:.4f}',
        f'Median = {np.median(errors):.4f}',
        f'Skewness = {np.mean(((errors - mean_err)/std_err)**3):.4f}',
        f'Kurtosis = {np.mean(((errors - mean_err)/std_err)**4) - 3:.4f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.03, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=props)
    
    # Save and display figure
    plt.tight_layout()
    plt.savefig('alpha_prediction_error_dist.png', dpi=300, bbox_inches='tight')
    print("Error distribution plot saved as 'alpha_prediction_error_dist.png'")
    plt.show()

def plot_ground_truth_alpha(alpha_values):
    """Plot distribution of ground truth alpha values"""
    # 首先将列表转换为NumPy数组
    alpha_values = np.array(alpha_values)
    
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    
    # 绘制直方图并添加密度曲线
    sns.histplot(alpha_values, bins=20, kde=True, color='forestgreen', alpha=0.7)
    
    # 添加标签和标题
    plt.xlabel('Diffusion Exponent (α)')
    plt.ylabel('Frequency')
    plt.title('Ground Truth α Value Distribution in Test Set')
    
    # 计算并显示统计数据
    mean_alpha = np.mean(alpha_values)
    std_alpha = np.std(alpha_values)
    
    # 添加标记线
    plt.axvline(mean_alpha, color='red', linestyle='-', linewidth=2, 
                label=f'Mean: {mean_alpha:.4f}')
    plt.axvline(mean_alpha + std_alpha, color='orange', linestyle=':', linewidth=1.5, 
                label=f'+1σ: {mean_alpha+std_alpha:.4f}')
    plt.axvline(mean_alpha - std_alpha, color='orange', linestyle=':', linewidth=1.5, 
                label=f'-1σ: {mean_alpha-std_alpha:.4f}')
    
    # 添加统计数据文本框
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
    
    # 添加各α值区间的分布统计
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
    plt.savefig('ground_truth_alpha_distribution.png', dpi=300, bbox_inches='tight')
    print("Ground truth α distribution saved as 'ground_truth_alpha_distribution.png'")
    
    # 可选：保存α值数据到文件以备后用
    np.save('test_alpha_values.npy', alpha_values)
    print(f"Saved {len(alpha_values)} test alpha values to 'test_alpha_values.npy'")
    
    plt.show()

# 加载保存的测试数据（包含模型ID）
def load_saved_test_data(filepath="test_dataset.npz"):
    print(f"加载保存的测试数据: {filepath}...")
    try:
        data = np.load(filepath, allow_pickle=True)
        X_test = data['features']
        Y_test = data['labels']
        model_ids = data['model_ids'] if 'model_ids' in data else None
        print(f"成功加载测试集，包含 {len(Y_test)} 个样本")
        if model_ids is not None:
            print(f"包含 {len(model_ids)} 个模型ID信息")
        return X_test, Y_test, model_ids
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return None, None, None
    
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
    print("\n===== 不同扩散模型类型性能分析 =====")
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
        model_name = model_names.get(model_id, f"模型{model_id}")
        print(f"{model_name} (n={np.sum(mask)}):")
        print(f"  • MSE: {mse:.6f}")
        print(f"  • MAE: {mae:.6f}")
        
        # 绘制散点
        plt.scatter(model_true, model_pred, 
                   label=f"{model_name} (n={np.sum(mask)})",
                   alpha=0.7, s=60)
    
    # 添加图例和标签
    plt.xlabel('True α')
    plt.ylabel('predicted α')
    plt.title('Diffusion Exponent Prediction Performance by Model Type')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 添加对角线（理想预测）
    min_val = min(np.min(y_true), np.min(y_pred)) * 0.9
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig('prediction_by_model.png', dpi=300)
    plt.show()

def plot_combined_models_hist2d(y_true, y_pred, model_ids):
    """
    将不同模型的 (真实值 vs 预测值) 绘制在同一个图中，使用 2D 直方图显示分布密度，
    并在每个子图上绘制:
      - 理想对角线 (y = x)
      - 分段平均曲线 (将 x 轴分为若干 bin, 计算每个 bin 中 y 的平均值并连线)
    
    参数：
    ----------
    y_true : array-like
        真实扩散指数 α 的数组
    y_pred : array-like
        预测扩散指数 α 的数组
    model_ids : array-like
        每个样本所属的模型类型 ID，需与下面的 model_names 对应
    """
    
    # 这里可根据需要自定义模型 ID 与名称的映射
    model_names = {
        0: "AATM",
        1: "CTRW",
        2: "FBM",
        3: "LW",
        4: "SBM"
    }
    
    # 获取所有出现的模型类型（去重）
    unique_models = np.unique(model_ids)
    
    # 使用深色背景风格（可根据个人喜好选择）
    plt.style.use('dark_background')
    
    # 创建子图：根据模型数自动生成列数，这里假设有 5 种模型
    fig, axes = plt.subplots(1, len(unique_models), 
                             figsize=(4 * len(unique_models), 4), 
                             sharex=True, sharey=True)
    # 如果只有一个模型，axes 可能不是数组，这里做一下兼容
    if len(unique_models) == 1:
        axes = [axes]
    
    # 为了统一颜色条，我们先保存最后一个 im 对象，再在循环外添加 colorbar
    im_list = []
    
    for i, model_id in enumerate(unique_models):
        ax = axes[i]
        
        # 筛选出当前模型对应的数据
        mask = (model_ids == model_id)
        x = y_true[mask]
        y = y_pred[mask]
        
        # 使用 2D 直方图来表示密度 (hist2d)
        # bins 可根据需要调整，range 设置到你想要观察的 α 范围
        # cmin=1 表示计数小于 1 的格子不显示颜色(可去掉)
        h = ax.hist2d(x, y, 
                      bins=50,          # 直方图网格数目，可根据数据量大小适当调整
                      range=[[0, 2], [0, 2]],  # 这里假设 α 范围在 [0, 2]
                      cmap='jet',       # 颜色映射，可换成你喜欢的
                      cmin=1)
        
        # 保存 im 对象以便后续统一加 colorbar
        im_list.append(h[3])  # h 的返回值: (counts, xedges, yedges, Image)
        
        # 绘制理想对角线 (y = x)
        ax.plot([0, 2], [0, 2], ls='--', color='white', linewidth=1)
        
        # 计算分段平均曲线（先将 x 轴分 bin，然后在每个 bin 中计算 y 的平均）
        bin_edges = np.linspace(0, 2, 20)  # 可根据需要调整分段数
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        digitized = np.digitize(x, bin_edges)
        bin_means = []
        for k in range(1, len(bin_edges)):
            # 找到第 k 个 bin 的所有 y
            y_in_bin = y[digitized == k]
            if len(y_in_bin) > 0:
                bin_means.append(np.mean(y_in_bin))
            else:
                bin_means.append(np.nan)
        
        # 在图上绘制这条平均曲线
        ax.plot(bin_centers, bin_means, color='white', linewidth=2)
        
        # 设置坐标范围 (如需自适应可去掉)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        
        # 设置标题
        model_name = model_names.get(model_id, f"模型{model_id}")
        ax.set_title(model_name, fontsize=14)
        
        # 只有第一个子图保留 y 轴标签
        if i == 0:
            ax.set_ylabel('Measured α', fontsize=12)
        ax.set_xlabel('Ground truth α', fontsize=12)
    
    # 统一添加颜色条（放在右侧）
    # 如果想让每个子图都有自己的颜色条，可以在循环内单独加
    cbar = fig.colorbar(im_list[-1], ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Counts', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('combined_models_hist2d.png', dpi=300)
    plt.show()

def main():
    batch_size = 32
    model_path = "cnn_lstm_attn_model.pt"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    
    # 只加载已保存的测试数据集，不重新生成数据
    test_data_file = "test_dataset.npz"
    if not os.path.exists(test_data_file):
        print(f"错误：未找到保存的测试数据文件 {test_data_file}。")
        print("请先运行包含数据生成和保存代码的脚本，或修改文件路径。")
        return  # 直接退出函数
    
    # 加载已保存的测试数据
    print(f"正在加载测试数据: {test_data_file}...")
    X_test, Y_test, model_ids_test = load_saved_test_data(test_data_file)
    
    if X_test is None or Y_test is None:
        print("错误：无法加载测试数据。")
        return  # 如果加载失败，直接退出
        
    print(f"成功加载测试数据集，包含 {len(Y_test)} 个样本")
    
    # 绘制真实α值分布直方图
    plot_ground_truth_alpha(Y_test)
    
    # Create and use scaler for input trajectories only
    scaler = create_and_fit_scaler(X_test)
    standardize = StandardScalerTransform(scaler)
    
    # Create test dataset with model_id support
    test_dataset = AnDiDataset(X_test, Y_test, 
                         model_ids=model_ids_test,
                         transform=standardize,
                         return_model_id=True) 
    
    # 使用 test_collate_fn 而不是 collate_fn
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=test_collate_fn  # 修改此处
    )
    
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EquilenCNNBiLSTMAttention(
        input_size=1, 
        conv_channels=128, 
        lstm_hidden_size=256,
        lstm_layers=2, 
        bidirectional=True, 
        dropout_rate=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run predictions and collect results
    print("Running predictions...")
    all_predictions = []
    all_targets = []
    all_model_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y, model_id = batch  # 注意这里解包三个元素
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_model_ids.append(model_id.cpu().numpy())
    
    # Combine all predictions and true values
    y_pred = np.concatenate(all_predictions).flatten()
    y_true = np.concatenate(all_targets).flatten()
    model_ids_array = np.concatenate(all_model_ids).flatten()

    plot_by_model_type(y_true, y_pred, model_ids_array)
    plot_combined_models_hist2d(y_true, y_pred, model_ids_array)

    
    # Calculate evaluation metrics
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    print(f"\n===== Overall Prediction Performance =====")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    
    # Perform regional error analysis
    analyze_by_region(y_true, y_pred)
    
    # Perform outlier analysis
    outlier_mask = analyze_outliers(y_true, y_pred, threshold=2.0)
    
    # ========== Create density scatter plot ==========
    # 1) Calculate 2D kernel density for each point
    x = y_true
    y = y_pred
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)  # Density value for each point
    
    # 2) Sort by density to ensure high density points are on top
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 14})
    sns.set_style("whitegrid")
    
    # 3) Create scatter plot with color based on density - MODIFIED: removed edgecolors
    scatter = plt.scatter(
        x, y, 
        c=z,            # Use kernel density value as color
        cmap='jet',     # Color map options: 'viridis', 'plasma', 'jet', etc.
        s=50,           # Point size
        alpha=0.8       # Transparency
        # Removed linewidths and edgecolors parameters
    )
    
    # 4) Removed outlier markers
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Probability Density')
    
    # Calculate axis range
    min_val = min(np.min(x), np.min(y)) * 0.9
    max_val = max(np.max(x), np.max(y)) * 1.1
    
    # Add diagonal line (ideal prediction)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Ideal Prediction")
    
    # Add best fit line
    z_coef = np.polyfit(x, y, 1)
    p = np.poly1d(z_coef)
    xx = np.linspace(min_val, max_val, 200)
    plt.plot(xx, p(xx), "g-", linewidth=1.5, 
             label=f"Best Fit (y={z_coef[0]:.2f}x+{z_coef[1]:.2f})")
    
    # Add ±10% error lines
    plt.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'r:', linewidth=1, alpha=0.6, label="+10%")
    plt.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'r:', linewidth=1, alpha=0.6, label="-10%")
    
    # Add labels and title
    plt.xlabel('True Diffusion Exponent α')
    plt.ylabel('Predicted Diffusion Exponent α')
    plt.title('Diffusion Exponent Prediction Performance and Density')
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    # Add performance metrics text box
    textstr = '\n'.join((
        f'MSE = {mse:.6f}',
        f'MAE = {mae:.6f}',
        f'RMSE = {rmse:.6f}',
        f'R² = {r2:.6f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('alpha_prediction_density_plot.png', dpi=300, bbox_inches='tight')
    print("Density scatter plot saved as 'alpha_prediction_density_plot.png'")
    plt.show()
    
    # Plot error distribution
    plot_error_distribution(y_true, y_pred)

if __name__ == "__main__":
    main()
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import calculate_diffusion_exponent

def validate_diffusion_exponent(trainer, test_trajectories, test_exponents, test_model_ids=None,
                               segment_length=100, noise_level=100, 
                               plot=True, n_samples=5, use_true_exponent=True,
                               condition_on_exponent=True, condition_on_model_id=False):
    """
    通过比较原始轨迹和延长轨迹的扩散指数来验证模型
    
    参数:
        trainer: 训练好的扩散训练器
        test_trajectories: 测试轨迹列表
        test_exponents: 真实扩散指数列表
        test_model_ids: 模型ID列表（可选）
        segment_length: 用作种子的段长度
        noise_level: 用于延长的噪声级别
        plot: 是否绘制结果
        n_samples: 要采样和绘制的轨迹数量
        use_true_exponent: 延长时是否基于真实指数进行条件化
        condition_on_exponent: 模型是否基于指数进行条件化
        condition_on_model_id: 模型是否基于模型ID进行条件化
        
    返回:
        结果字典
    """
    results = {
        'true_alphas': [],
        'predicted_alphas': [],
        'msd_errors': [],
        'extension_errors': []
    }
    
    # 处理一部分轨迹用于可视化
    n_to_process = min(n_samples, len(test_trajectories)) if plot else len(test_trajectories)
    
    # 使用tqdm跟踪进度
    for i in tqdm(range(n_to_process), desc="验证轨迹"):
        # 获取轨迹和元数据
        true_traj = np.array(test_trajectories[i])
        true_alpha = test_exponents[i]
        
        # 如果需要，对1D进行重塑
        if len(true_traj.shape) == 1:
            true_traj = true_traj.reshape(-1, 1)
        
        # 提取初始段
        initial_segment = true_traj[:segment_length]
        
        # 转换为张量并添加批次维度
        x_short = torch.FloatTensor(initial_segment).unsqueeze(0)  # (1, segment_length, dim)
        
        # 如果需要，准备条件输入
        exponent = None
        model_id = None
        
        if condition_on_exponent and use_true_exponent:
            exponent = torch.FloatTensor([[true_alpha]])
            
        if condition_on_model_id and test_model_ids is not None:
            model_id = torch.LongTensor([[test_model_ids[i]]])
        
        # 延长轨迹
        extended_traj = trainer.extend_trajectory(
            x_short, exponent=exponent, model_id=model_id, noise_level=noise_level
        )
        extended_traj = extended_traj.cpu().numpy()[0]  # 移除批次维度
        
        # 计算延长轨迹的扩散指数
        pred_alpha = calculate_diffusion_exponent(extended_traj, plot=False)
        
        # 计算误差
        msd_error = abs(true_alpha - pred_alpha)
        
        # 对于轨迹，只比较重叠部分（前segment_length个点）
        extension_error = np.mean((true_traj[:segment_length] - extended_traj[:segment_length])**2)
        
        # 存储结果
        results['true_alphas'].append(true_alpha)
        results['predicted_alphas'].append(pred_alpha)
        results['msd_errors'].append(msd_error)
        results['extension_errors'].append(extension_error)
        
        # 绘制可视化
        if plot and i < n_samples:
            plt.figure(figsize=(15, 10))
            
            # 绘制轨迹
            plt.subplot(2, 2, 1)
            t = np.arange(len(true_traj))
            plt.plot(t, true_traj, 'b-', label='真实', alpha=0.7)
            plt.plot(t[:len(extended_traj)], extended_traj, 'r-', label='延长', alpha=0.7)
            plt.axvline(x=segment_length, color='g', linestyle='--', label=f'截断点 (t={segment_length})')
            plt.legend()
            plt.title(f'轨迹比较 (模型 {test_model_ids[i] if test_model_ids else "?"}, α={true_alpha:.2f})')
            
            # 绘制真实轨迹的MSD
            plt.subplot(2, 2, 2)
            true_alpha_calc = calculate_diffusion_exponent(true_traj, plot=True)
            plt.title(f'真实轨迹MSD: α={true_alpha_calc:.3f} (给定: {true_alpha:.3f})')
            
            # 绘制延长轨迹的MSD
            plt.subplot(2, 2, 3)
            pred_alpha = calculate_diffusion_exponent(extended_traj, plot=True)
            plt.title(f'延长轨迹MSD: α={pred_alpha:.3f}')
            
            # 绘制alpha比较
            plt.subplot(2, 2, 4)
            plt.bar(['真实', '预测'], [true_alpha, pred_alpha])
            plt.axhline(y=1.0, color='r', linestyle='--', label='正常扩散')
            plt.ylabel('扩散指数 (α)')
            plt.title(f'扩散指数比较\n绝对误差: {abs(true_alpha-pred_alpha):.3f}')
            
            plt.tight_layout()
            plt.show()
    
    # 计算总体指标
    true_alphas = np.array(results['true_alphas'])
    pred_alphas = np.array(results['predicted_alphas'])
    
    alpha_mae = np.mean(np.abs(true_alphas - pred_alphas))
    alpha_rmse = np.sqrt(np.mean((true_alphas - pred_alphas)**2))
    
    print(f"总体扩散指数MAE: {alpha_mae:.4f}")
    print(f"总体扩散指数RMSE: {alpha_rmse:.4f}")
    
    # 绘制总体比较
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(true_alphas, pred_alphas, alpha=0.7)
        
        # 添加恒等线
        min_val = min(min(true_alphas), min(pred_alphas))
        max_val = max(max(true_alphas), max(pred_alphas))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加误差带 (±0.1)
        plt.fill_between([min_val, max_val], 
                        [min_val - 0.1, max_val - 0.1], 
                        [min_val + 0.1, max_val + 0.1], 
                        color='gray', alpha=0.2)
        
        plt.xlabel('真实扩散指数 (α)')
        plt.ylabel('预测扩散指数 (α)')
        plt.title('扩散指数: 真实 vs 预测')
        plt.grid(True)
        plt.show()
        
        # 绘制误差直方图
        plt.figure(figsize=(8, 6))
        plt.hist(true_alphas - pred_alphas, bins=20)
        plt.xlabel('误差 (真实 - 预测)')
        plt.ylabel('频率')
        plt.title('扩散指数预测误差分布')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True)
        plt.show()
        
        # 如果可用，按模型类型绘制误差
        if test_model_ids is not None:
            model_ids = np.array(test_model_ids[:n_to_process])
            unique_models = np.unique(model_ids)
            
            model_errors = []
            for model_id in unique_models:
                mask = (model_ids == model_id)
                model_mae = np.mean(np.abs(true_alphas[mask] - pred_alphas[mask]))
                model_errors.append(model_mae)
            
            plt.figure(figsize=(8, 6))
            plt.bar(unique_models, model_errors)
            plt.xlabel('模型ID')
            plt.ylabel('平均绝对误差')
            plt.title('按模型类型的扩散指数预测误差')
            plt.xticks(unique_models)
            plt.grid(True)
            plt.show()
    
    return results
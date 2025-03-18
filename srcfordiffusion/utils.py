import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def calculate_diffusion_exponent(trajectory, plot=False):
    """
    通过均方位移(MSD)从轨迹计算扩散指数。
    
    参数:
        trajectory: 形状为 (n_steps,) 或 (n_steps, dim) 的数组
        plot: 是否绘制MSD与时间的关系图
        
    返回:
        alpha: 扩散指数
    """
    # 确保轨迹至少是2D（即使对于1D数据）
    if len(trajectory.shape) == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, dim = trajectory.shape
    
    # 计算不同时间滞后的MSD
    max_lag = n_steps // 4  # 使用轨迹长度的最多1/4
    lags = np.arange(1, max_lag)
    msds = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # 计算所有可能的长度为'lag'的时间分离的平方位移
        squared_displacements = np.sum((trajectory[lag:] - trajectory[:-lag])**2, axis=1)
        msds[i] = np.mean(squared_displacements)
    
    # 对MSD(t) ~ t^alpha拟合幂律
    log_lags = np.log(lags)
    log_msds = np.log(msds)
    
    # 对数-对数尺度上的线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_msds)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(lags, msds, 'o', label='MSD数据')
        plt.loglog(lags, np.exp(intercept) * lags**slope, 'r-', 
                 label=f'拟合: MSD ~ t^{slope:.3f}')
        plt.xlabel('时间滞后')
        plt.ylabel('MSD')
        plt.legend()
        plt.title(f'扩散指数 (Alpha): {slope:.3f}')
        plt.grid(True)
        plt.show()
    
    return slope  # 这是扩散指数alpha
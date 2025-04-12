import numpy as np
import random
import torch
from torch.utils.data import Dataset, random_split
import scipy.stats as stats

def load_andi_data(N, Length, add_noise=True, noise_level=0.1):
    """
    加载ANDI数据集轨迹，包含不同的异常扩散模型
    
    参数:
        N: 每个模型的轨迹数量
        Length: 每个轨迹的长度
        add_noise: 是否添加噪声，默认为True
        noise_level: 噪声水平，默认为0.1 (10%)
        
    返回:
        X: 轨迹列表
        Y: 扩散指数列表
        model_ids: 模型ID列表
    """
    from andi_datasets.datasets_theory import datasets_theory
    AD_instance = datasets_theory()
    
    # 从不同模型生成数据（每个模型可调整的数量）
    data1 = AD_instance.create_dataset(T=Length, N_models=140, exponents=np.random.uniform(0.05, 1.0, size=140), models=0, dimension=1)
    data2 = AD_instance.create_dataset(T=Length, N_models=140, exponents=np.random.uniform(0.05, 1.0, size=140), models=1, dimension=1)
    data3 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 2.0, size=N), models=2, dimension=1)
    data4 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(1.0, 2.0, size=N),  models=3, dimension=1)
    data5 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 2.0, size=N), models=4, dimension=1)

    # 合并所有数据
    merged_all_data = np.vstack([data1, data2, data3, data4, data5])
    
    # 根据参数决定是否添加噪声
    if add_noise:
        all_data = AD_instance.create_noisy_diffusion_dataset(merged_all_data, T=Length)
    else:
        all_data = merged_all_data  # 不添加噪声，直接使用原始数据

    # 提取轨迹、标签和模型ID
    X = []
    Y = []
    model_ids = []
    for i in range(len(all_data)):
        trajectory = all_data[i, 2:]  # 轨迹数据从第3列开始
        X.append(trajectory.astype(np.float32))
        Y.append(float(all_data[i, 1]))      # 第2列是扩散指数
        model_ids.append(int(all_data[i, 0]))  # 第1列是模型ID

    # 打乱数据并保持对应关系
    combined = list(zip(X, Y, model_ids))
    random.shuffle(combined)
    X, Y, model_ids = zip(*combined)
    
    return list(X), list(Y), list(model_ids)

class TrajectoryDataset(Dataset):
    """轨迹段数据集"""
    def __init__(self, trajectories, exponents, model_ids=None, segment_length=100, prediction_length=100, normalize=True):
        """
        参数:
            trajectories: 轨迹数组列表
            exponents: 每个轨迹的扩散指数列表
            model_ids: 每个轨迹的模型ID列表（可选）
            segment_length: 输入段的长度(T_in)
            prediction_length: 要预测的未来轨迹的长度(T_out)
            normalize: 是否对轨迹进行归一化
        """
        self.segments = []
        self.targets = []
        self.segment_exponents = []
        self.segment_model_ids = []
        self.normalize = normalize
        self.stats = {}  # 存储归一化统计数据
        
        # 处理每个轨迹
        for i, traj in enumerate(trajectories):
            # 确保轨迹是numpy数组
            traj = np.array(traj)
            
            # 如果需要，对1D数据进行重塑
            if len(traj.shape) == 1:
                traj = traj.reshape(-1, 1)
            
            # 如果需要，进行归一化
            if normalize:
                # 计算此轨迹的统计数据
                mean = np.mean(traj, axis=0)
                std = np.std(traj, axis=0)
                std = np.where(std < 1e-6, 1.0, std)  # 避免除以零
                
                # 存储统计数据
                self.stats[len(self.segments)] = {'mean': mean, 'std': std}
                
                # 归一化
                traj_normalized = (traj - mean) / std
                traj_to_use = traj_normalized
            else:
                traj_to_use = traj
            
            # 创建滑动窗口段
            n_steps = len(traj_to_use)
            for j in range(0, n_steps - segment_length - prediction_length + 1, segment_length // 2):  # 重叠段
                self.segments.append(traj_to_use[j:j+segment_length])
                self.targets.append(traj_to_use[j+segment_length:j+segment_length+prediction_length])
                self.segment_exponents.append(exponents[i])
                if model_ids is not None:
                    self.segment_model_ids.append(model_ids[i])
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        target = torch.FloatTensor(self.targets[idx])
        exponent = torch.FloatTensor(np.array([self.segment_exponents[idx]]))
        
        # 如果有模型ID则包含
        if hasattr(self, 'segment_model_ids') and len(self.segment_model_ids) > 0:
            model_id = torch.tensor(self.segment_model_ids[idx], dtype=torch.long)
            return segment, target, exponent, model_id
        else:
            return segment, target, exponent

def split_dataset(dataset, test_size=0.1, val_size=0.1, seed=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        dataset: 要分割的数据集
        test_size: 测试集的比例
        val_size: 验证集的比例
        seed: 随机种子
        
    返回:
        train_dataset, val_dataset, test_dataset: 分割后的数据集
    """
    # 设置随机种子以确保可复现性
    generator = torch.Generator().manual_seed(seed)
    
    # 计算各部分大小
    total_size = len(dataset)
    test_size_abs = int(total_size * test_size)
    val_size_abs = int(total_size * val_size)
    train_size = total_size - test_size_abs - val_size_abs
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size_abs, test_size_abs],
        generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset

# 同时加载清洁和有噪声的数据集的实用函数
def load_clean_and_noisy_data(N, Length, noise_level=0.1):
    """
    同时加载清洁和带噪声的数据集
    适合扩散模型训练（清洁数据）和验证（噪声数据）
    
    参数:
        N: 每个模型的轨迹数量
        Length: 每个轨迹的长度
        noise_level: 噪声数据的噪声水平
        
    返回:
        clean_data: (X, Y, model_ids) 干净数据的元组
        noisy_data: (X, Y, model_ids) 带噪声数据的元组
    """
    # 加载相同种子的数据以保持一致性
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    # 加载清洁数据
    X_clean, Y_clean, model_ids_clean = load_andi_data(N, Length, add_noise=False)
    
    # 重置随机状态以获取相同的轨迹
    random.setstate(random_state)
    np.random.set_state(np_state)
    
    # 加载噪声数据
    X_noisy, Y_noisy, model_ids_noisy = load_andi_data(N, Length, add_noise=True, noise_level=noise_level)
    
    return (X_clean, Y_clean, model_ids_clean), (X_noisy, Y_noisy, model_ids_noisy)
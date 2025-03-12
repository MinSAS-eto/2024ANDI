import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from andi_datasets.datasets_theory import datasets_theory
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1. 先写一个函数，只在这里调用一次 challenge_theory_dataset
###############################################################################
def load_andi_data(N, Length):
    """
    返回 X, Y，其中 X 是 list/array of arrays (变长序列)，Y 是 list/array of labels
    """
    AD_instance = datasets_theory()
    
    # 生成不同模型的数据
    data1 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 1.0, size=N), models=0, dimension=1)
    data2 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 1.0, size=N), models=1, dimension=1)
    data3 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 2.0, size=N), models=2, dimension=1)
    data4 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(1.0, 2.0, size=N), models=3, dimension=1)
    data5 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 1.0, size=N), models=4, dimension=1)

    # 合并所有数据
    all_data = np.vstack([data1, data2, data3, data4, data5])

    # 提取轨迹和标签
    X = []
    Y = []
    
    # 根据文档描述:
    # 每行对应一条轨迹
    # 第一列是模型标签，第二列是扩散系数，其余是轨迹数据
    for i in range(len(all_data)):
        trajectory = all_data[i, 2:]  # 从第3列开始是轨迹数据
        X.append(trajectory)
        Y.append(all_data[i, 1])      # 第2列是扩散系数
    
    # 打乱数据
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X, Y = zip(*combined)
    
    # 转回列表
    X = list(X)
    Y = list(Y)
    
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
# 5. 简化 collate_fn，移除标签标准化操作
###############################################################################
def collate_fn(batch):  # 移除label_scaler参数
    xs, ys = zip(*batch)
    xs_tensor = []
    
    for x in xs:
        # x 如果是 numpy 或 list, 转成 torch
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)

    # 直接堆叠成批次 - 不需要padding
    batch_x = torch.stack(xs_tensor, dim=0)
    
    # 直接转换标签为张量，不做标准化
    ys_tensor = torch.tensor(ys, dtype=torch.float32)

    # 不再返回长度值
    return batch_x, ys_tensor

#6. 定义 TimeReversedTransform以概率p随机反转时间序列
class TimeReversedTransform:
    """
    时间反转变换，以概率p随机反转时间序列
    
    Args:
        p (float): 应用此变换的概率，取值范围 [0, 1]
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        # x 应该是一个时间序列
        if random.random() < self.p:
            # 沿着时间维度翻转
            return x.flip(0)  
        return x
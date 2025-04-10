import torch
import numpy as np
import random
from math import sqrt
from torch.utils.data import Dataset, DataLoader
from andi_datasets.datasets_theory import datasets_theory
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1. 先写一个函数，只在这里调用一次 challenge_theory_dataset
###############################################################################
def load_andi_data(N, Length):
    """
    返回 X, Y, model_ids
    X：轨迹数据
    Y：扩散系数
    model_ids：模型类型ID
    """
    AD_instance = datasets_theory()
    
    # 生成不同模型的数据
    data1 = AD_instance.create_dataset(T=Length, N_models=160, exponents=np.random.uniform(0.05, 1.0, size=160), models=0, dimension=1)
    data2 = AD_instance.create_dataset(T=Length, N_models=160, exponents=np.random.uniform(0.05, 1.0, size=160), models=1, dimension=1)
    data3 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 2.0, size=N), models=2, dimension=1)
    data4 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(1.0, 2.0, size=N), models=3, dimension=1)
    data5 = AD_instance.create_dataset(T=Length, N_models=N, exponents=np.random.uniform(0.05, 2.0, size=N), models=4, dimension=1)

    # 合并所有数据
    merged_all_data = np.vstack([data1, data2, data3, data4, data5])
    all_data = AD_instance.create_noisy_diffusion_dataset(merged_all_data, T=Length)
    
    # 提取轨迹、标签和模型ID
    X = []
    Y = []
    model_ids = []  # 新增：存储模型ID
    
    for i in range(len(all_data)):
        trajectory = all_data[i, 2:]  # 从第3列开始是轨迹数据
        X.append(trajectory)
        Y.append(all_data[i, 1])      # 第2列是扩散系数
        model_ids.append(int(all_data[i, 0]))  # 第1列是模型ID
    
    # 打乱数据（保持对应关系）
    combined = list(zip(X, Y, model_ids))
    random.shuffle(combined)
    X, Y, model_ids = zip(*combined)
    
    # 转回列表
    X = list(X)
    Y = list(Y)
    model_ids = list(model_ids)
    
    return X, Y, model_ids


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
# 4. 重新定义 AnDiDataset，支持可选的模型ID返回
###############################################################################
class AnDiDataset(Dataset):
    def __init__(self, X, Y, model_ids=None, transform=None, return_model_id=False):
        """
        X: list of arrays (每条序列长度不定)
        Y: list/array of 对应标签
        model_ids: list/array of 模型ID
        transform: 如果需要对 x 做转换, 传入 transform
        return_model_id: 是否在__getitem__中返回model_id
        """
        super().__init__()
        self.X = X
        self.Y = Y
        self.model_ids = model_ids if model_ids is not None else [0] * len(X)
        self.transform = transform
        self.return_model_id = return_model_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
            
        if self.return_model_id:
            return x, y, self.model_ids[idx]
        else:
            return x, y


###############################################################################
# 5. 定义训练和测试用的不同collate_fn
###############################################################################
def train_collate_fn(batch):
    """训练时使用的collate_fn，只处理x和y"""
    xs, ys = zip(*batch)
    xs_tensor = []
    
    for x in xs:
        # x 如果是 numpy 或 list, 转成 torch
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)

    # 直接堆叠成批次
    batch_x = torch.stack(xs_tensor, dim=0)
    
    # 转换标签为张量
    ys_tensor = torch.tensor(ys, dtype=torch.float32)

    return batch_x, ys_tensor

def test_collate_fn(batch):
    """测试时使用的collate_fn，处理x、y和model_id"""
    # 解包三个值
    xs, ys, model_ids = zip(*batch)
    xs_tensor = []
    
    for x in xs:
        # x 如果是 numpy 或 list, 转成 torch
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)

    # 堆叠
    batch_x = torch.stack(xs_tensor, dim=0)
    
    # 转换标签和模型ID为张量
    ys_tensor = torch.tensor(ys, dtype=torch.float32)
    model_ids_tensor = torch.tensor(model_ids, dtype=torch.long)

    return batch_x, ys_tensor, model_ids_tensor

# 为保持向后兼容，保留原有的collate_fn
collate_fn = train_collate_fn

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
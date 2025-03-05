import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
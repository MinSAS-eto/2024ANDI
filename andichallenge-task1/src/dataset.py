import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from andi_datasets.datasets_challenge import challenge_theory_dataset
from sklearn.preprocessing import StandardScaler
class StandardScalerTransform:
    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler

    def __call__(self, x):

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

class TimeReverseTransform:
    def __call__(self, x):

        if isinstance(x, torch.Tensor):
            x = torch.flip(x, dims=[0])
        else:
            x = np.array(x, dtype=np.float32)[::-1]
            x = torch.tensor(x, dtype=torch.float32)
        return x



def create_and_fit_scaler(N, tasks):

    # 先从 andi 生成数据
    X1, Y1, _, _, _, _ = challenge_theory_dataset(N=N, tasks=tasks, dimensions=1)
    X = X1[0] 
    all_data = np.concatenate(X)
    # reshape
    all_data = all_data.reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(all_data)

    return scaler

class AnDiDataset(Dataset):
    def __init__(self, N, tasks, transform=None, scaler=None):
        super().__init__()
        self.transform = transform
        # 使用传入的 N 和 tasks 参数生成数据集，维度固定为 1
        X1, Y1, _, _, _, _ = challenge_theory_dataset(N=N, tasks=tasks, dimensions=1)
        self.X = X1[0]  # 确保选择正确的维度
        self.Y = Y1[0]  # 确保选择正确的维度
        self.N = N
        self.tasks = tasks
        # 如果有 scaler，可以内部再做 transform
        self.scaler = scaler

    def __len__(self):
        # 返回样本数量
        return len(self.X)

    def __getitem__(self, idx):
        # 根据索引获取样本
        x = self.X[idx]
        y = self.Y[idx]
        # 如果想要在 dataset 内就做标准化，而不依赖 transform
        # 也可以在这里直接调用 scaler.transform(x)
        # 这里暂时保留 transform 机制
        # 如有 transform 则进行转换
        if self.transform:
            x = self.transform(x)

        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_tensor = []
    lengths_x = []
    for x in xs:
        # 此时 x 应该已经是预处理后的 torch.Tensor
        # 下面只做浮点转换和统计长度
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)
        lengths_x.append(len(x))

    # 对 x 进行 padding
    padded_x = torch.nn.utils.rnn.pad_sequence(
        xs_tensor, batch_first=True, padding_value=0
    )

    # 将 y 直接转换为 tensor，不需要 padding
    ys_tensor = torch.tensor(ys, dtype=torch.float32)

    return padded_x, lengths_x, ys_tensor

import torch
from torch.utils.data import Dataset, DataLoader
from andi_datasets.datasets_challenge import challenge_theory_dataset

class AnDiDataset(Dataset):
    def __init__(self, N, tasks, transform=None):
        super().__init__()
        self.transform = transform
        # 使用传入的 N 和 tasks 参数生成数据集，维度固定为 1
        X1, Y1, _, _, _, _ = challenge_theory_dataset(N=N, tasks=tasks, dimensions=1)
        self.X = X1[0]  # 确保选择正确的维度
        self.Y = Y1[0]  # 确保选择正确的维度
        self.N = N
        self.tasks = tasks
        
    def __len__(self):
        # 返回样本数量
        return len(self.X)

    def __getitem__(self, idx):
        # 根据索引获取样本
        x = self.X[idx]
        y = self.Y[idx]
        # 如有转换函数则进行转换
        if self.transform:
            x = self.transform(x)
            
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    
    xs_tensor = []
    lengths_x = []
    for x in xs:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)
        lengths_x.append(len(x))
    
    # 对 x 进行 padding
    padded_x = torch.nn.utils.rnn.pad_sequence(xs_tensor, batch_first=True, padding_value=0)
    # 将 y 直接转换为 tensor，不需要 padding
    ys_tensor = torch.tensor(ys, dtype=torch.float32)
    
    return padded_x, lengths_x, ys_tensor

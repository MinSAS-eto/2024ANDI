import torch
from torch.utils.data import Dataset
import andi_datasets

class AnDiDataset(Dataset):
    """
    生成 1D 轨迹数据（Task=1），存放在 Dataset 中。
    """

    def __init__(self, N, tasks, transform=None, T=50):
        """
        N: 生成的轨迹数量
        tasks: 哪个任务（这里假设=1）
        transform: 可选的变换函数，用于对轨迹做进一步预处理（若需要）
        T: 每个轨迹的长度
        """
        super().__init__()
        self.transform = transform

        # 固定只生成 1D 的数据
        X, Y = andi_datasets().andi_dataset(
            N=N, tasks=tasks, dimensions=1, T=T
        )

        # 从生成的字典中取出 1 维对应的轨迹数据
        self.X = X[1]  # shape: (N, L) 当 d==1 时
        self.Y = Y[1]  # shape: (N, )  (Task1 为回归值)

        # 数据集长度统一为 N（假设每个维度都生成了 N 条轨迹）
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        返回 1D 数据对应的样本
         - 返回的 x 为轨迹数据（形状可能为 (L,)）
         - y 为对应的回归值
        """
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from utils import train, evaluate
from dataset import (
    AnDiDataset, 
    collate_fn, 
    create_and_fit_scaler, 
    StandardScalerTransform, 
    TimeReverseTransform
)
from model import CNNBiLSTM

###############################################################################
# 1. 定义一个包装器，用于给 subset 添加 transform
###############################################################################
class TransformedSubset(torch.utils.data.Dataset):
    """
    包装一个 subset（通常由 random_split 返回的 Subset），
    在 __getitem__ 中对 (x, y) 施加 transform(x)。
    """
    def __init__(self, subset, transform=None):
        super().__init__()
        self.subset = subset          # random_split 返回的 Subset
        self.transform = transform    # 可为 None 或 Compose

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]  # 先从 subset 中取原始 (x, y)
        if self.transform:
            x = self.transform(x)
        return x, y


def main():
    # 超参数设置
    N = 1000
    tasks = 1
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3

    # 1) 先用训练数据 fit 得到 scaler（这里为演示，直接在 N=1000 上 fit）
    scaler = create_and_fit_scaler(N=N, tasks=tasks)
    standardize = StandardScalerTransform(scaler)
    reverse_time = TimeReverseTransform()

    # 2) 构造原始 dataset（不带 transform）
    raw_dataset = AnDiDataset(N=N, tasks=tasks, transform=None)

    # 3) random_split 切分成 train/val/test
    train_size = int(0.8 * len(raw_dataset))
    val_size = int(0.1 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size - val_size
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size]
    )

    # 4) 只在训练集上做「标准化 + 时间反转」，验证/测试只做「标准化」
    train_transform = T.Compose([standardize, reverse_time])
    val_transform = standardize
    test_transform = standardize

    # 5) 用包装器给 subset 加上不同 transforms
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset   = TransformedSubset(val_subset,   transform=val_transform)
    test_dataset  = TransformedSubset(test_subset,  transform=test_transform)

    # 6) 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, 
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, 
                              shuffle=False, collate_fn=collate_fn)

    # 7) 定义模型、损失、优化器
    model = CNNBiLSTM()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, threshold=0.1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Training Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Accuracy: {val_acc:.4f}")
        
        # Early Stopping 逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # 加载最优模型评估
    best_model = CNNBiLSTM()
    best_model.load_state_dict(torch.load("best_model.pt"))
    best_model.to(device)

    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device, threshold=0.1)
    print(f"\n[Final Test] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
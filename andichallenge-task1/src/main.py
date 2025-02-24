import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import AnDiDataset
from model import CNNBiLSTM
from utils import train_one_epoch, validate


def main():
    # 超参数设置
    N = 2000
    T = 50
    dims = 1
    tasks = 1        
    batch_size = 32
    epochs = 10
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据集
    dataset = AnDiDataset(
        N=N,
        tasks=tasks,
        dimensions=dims
    )

    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = CNNBiLSTM(
        input_length=T,
        conv_channels=16,
        lstm_hidden_size=32,
        lstm_layers=1,
        bidirectional=True
    )
    model.to(device)

    # 定义损失和优化器（回归 => MSELoss）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
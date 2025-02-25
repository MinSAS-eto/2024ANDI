import torch
from torch.utils.data import DataLoader
from utils import train, evaluate
from dataset import AnDiDataset, collate_fn
from model import CNNBiLSTM

def main():
    # 超参数设置
    N = 1000                    # 样本数，可根据需要调整
    tasks = 1           # 任务列表，根据实际需求设定
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    # 创建数据集和 DataLoader
    dataset = AnDiDataset(N=N, tasks=tasks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 初始化模型、损失函数和优化器
    model = CNNBiLSTM()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        val_loss = evaluate(model, dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

if __name__ == '__main__':
    main()
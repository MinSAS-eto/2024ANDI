import torch
from torch.utils.data import DataLoader, random_split
from utils import train, evaluate
from dataset import AnDiDataset, collate_fn
from model import CNNBiLSTM

def main():
    # 超参数设置
    N = 100
    tasks = 1
    batch_size = 32
    num_epochs = 1
    learning_rate = 1e-3

    # 创建数据集及划分数据集
    dataset = AnDiDataset(N=N, tasks=tasks)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    best_model = CNNBiLSTM()
    best_model.load_state_dict(torch.load("best_model.pt"))  # 加载最优权重
    best_model.to(device)

    # 在测试集上验证
    test_loss, test_acc = evaluate(best_model, test_loader, criterion, device, threshold=0.1)
    print(f"\n[Final Test] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
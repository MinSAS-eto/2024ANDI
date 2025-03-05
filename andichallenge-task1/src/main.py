import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split
from utils import train, evaluate
from dataset import (
    AnDiDataset, 
    collate_fn, 
    create_and_fit_scaler, 
    StandardScalerTransform, 
    load_andi_data
)
from model import ImprovedCNNBiLSTM

class TransformedSubset(torch.utils.data.Dataset):
    """
    包装一个 subset（通常由 random_split 返回的 Subset），
    在 __getitem__ 中对 (x, y) 施加 transform(x)。
    """
    def __init__(self, subset, transform=None):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def main():
    # 超参数设置
    N = 6000
    tasks = 1
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # 1) 获取数据
    X, Y = load_andi_data(N, tasks)

    # 2) 用 X 来 fit scaler
    scaler = create_and_fit_scaler(X)
    standardize = StandardScalerTransform(scaler)
    y_scaler = StandardScaler()
    y_scaler.fit(np.array(Y).reshape(-1, 1))
    # 3) 构造原始 dataset（不带 transform）
    raw_dataset = AnDiDataset(X, Y, transform=None)

    # 4) random_split 切分成 train/val/test
    train_size = int(0.8 * len(raw_dataset))
    val_size = int(0.1 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size - val_size
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size]
    )

    # 5) 只在训练集上做「标准化 + 时间反转」，验证/测试只做「标准化」
    train_transform = standardize
    val_transform = standardize
    test_transform = standardize

    # 6) 用包装器给 subset 加上不同 transforms
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset   = TransformedSubset(val_subset,   transform=val_transform)
    test_dataset  = TransformedSubset(test_subset,  transform=test_transform)

    # 7) 构造 DataLoader，添加标签归一化
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=lambda batch: collate_fn(batch, y_scaler))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=lambda batch: collate_fn(batch, y_scaler))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, collate_fn=lambda batch: collate_fn(batch, y_scaler))

    # 8) 定义模型、损失、优化器
    # 使用改进后的模型 ImprovedCNNBiLSTM，注意参数可根据任务调优
    model = ImprovedCNNBiLSTM(
        conv_channels=64,         # 基础卷积通道数
        lstm_hidden_size=128,      # LSTM 隐状态维度
        lstm_layers=4,            # 使用4层LSTM
        bidirectional=True,
        dropout_rate=0.5,
        kernel_sizes=[3, 5, 7]    # 多尺度卷积核尺寸
    )
    criterion = nn.L1Loss()  # MAE损失
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    orig_val_losses = []  # 用于记录原始空间的验证损失
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 获取训练结果，并提取损失值
        train_result = train(model, train_loader, criterion, optimizer, device)
        train_loss = train_result[0] if isinstance(train_result, tuple) else train_result
        
        # 获取验证结果，并提取损失值
        val_result = evaluate(model, val_loader, criterion, device, y_scaler=y_scaler)
        val_loss = val_result[0] if isinstance(val_result, tuple) else val_result
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 如果有原始空间损失，记录并打印
        if isinstance(val_result, tuple) and len(val_result) > 1:
            orig_val_loss = val_result[1]
            orig_val_losses.append(orig_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Training Loss: {train_loss:.4f}, "
                  f"Validation Loss (Norm): {val_loss:.4f}, "
                  f"Validation Loss (Orig): {orig_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Training Loss: {train_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}")
        
        # Early Stopping 逻辑保持不变 (使用归一化损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # 绘制训练和验证 Loss 曲线 (归一化空间)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("norm_loss_curve.png")
    plt.show()
    
    # 如果有原始空间损失，绘制原始空间损失曲线
    if orig_val_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(orig_val_losses) + 1), orig_val_losses, label='Validation Loss (Original Scale)')
        plt.title('Validation Loss in Original Scale')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("orig_loss_curve.png")
        plt.show()
    
    # 加载最优模型并评估
    best_model = ImprovedCNNBiLSTM(
        conv_channels=64,
        lstm_hidden_size=128,
        lstm_layers=4,
        bidirectional=True,
        dropout_rate=0.5,
        kernel_sizes=[3, 5, 7]
    )
    best_model.load_state_dict(torch.load("best_model.pt"))
    best_model.to(device)
    
    # 评估并显示两种空间的损失
    test_result = evaluate(best_model, test_loader, criterion, device, y_scaler=y_scaler)
    test_norm_loss = test_result[0] if isinstance(test_result, tuple) else test_result
    
    print(f"\n[Final Test] Normalized Loss: {test_norm_loss:.4f}")
    
    # 如果有原始空间测试损失，也显示出来
    if isinstance(test_result, tuple) and len(test_result) > 1:
        test_orig_loss = test_result[1]
        print(f"[Final Test] Original Scale Loss: {test_orig_loss:.4f}")

if __name__ == '__main__':
    main()
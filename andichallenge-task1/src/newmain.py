import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T
import random
from torch.utils.data import DataLoader, random_split
from utils import train, evaluate, TransformedSubset
from dataset import (
    AnDiDataset, 
    collate_fn, 
    create_and_fit_scaler, 
    StandardScalerTransform, 
    TimeReversedTransform,
    load_andi_data,

)
from newmodel import SimpleLSTM

def main():
    # 超参数设置
    N = 50
    Length = 100
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # 1) 获取数据
    X, Y = load_andi_data(N, Length)

    # 2) 用 X 来 fit scaler
    scaler = create_and_fit_scaler(X)
    standardize = StandardScalerTransform(scaler)
    
    # 3) 构造原始 dataset（不带 transform）
    raw_dataset = AnDiDataset(X, Y, transform=None)

    # 4) random_split 切分成 train/val/test
    train_size = int(0.8 * len(raw_dataset))
    val_size = int(0.1 * len(raw_dataset))
    test_size = len(raw_dataset) - train_size - val_size
    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_size, val_size, test_size]
    )

    # 5) 只在训练集上做「标准化」，验证/测试也做「标准化」
    train_transform = T.Compose([standardize, TimeReversedTransform(p=0.5)])
    val_transform = standardize
    test_transform = standardize

    # 6) 用包装器给 subset 加上不同 transforms
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset   = TransformedSubset(val_subset,   transform=val_transform)
    test_dataset  = TransformedSubset(test_subset,  transform=test_transform)
    print(f"训练集大小: {len(train_subset)}")
    print(f"理论总批次数: {len(train_subset) // batch_size * num_epochs}")

    # 7) 构造 DataLoader（不带标签归一化）
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, collate_fn=collate_fn)

    # 8) 定义模型、损失、优化器
    model = SimpleLSTM()
    # 使用MSE损失进行训练
    criterion = nn.MSELoss()  # 修改为MSE损失
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 添加学习率调度器 - 移除verbose参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA是否可用:", torch.cuda.is_available())
    print("GPU数量:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("当前GPU:", torch.cuda.get_device_name())
    print("PyTorch版本:", torch.__version__)
    print("CUDA版本:", torch.version.cuda if hasattr(torch.version, 'cuda') else "不支持CUDA")
    print(f"使用设备: {device}")
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    print(f"开始训练，总epoch数: {num_epochs}")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 获取训练结果，并提取损失值
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # 获取验证结果，并提取损失值
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率调度器
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Training Loss (MSE): {train_loss:.4f}, "
              f"Validation Loss (MSE): {val_loss:.4f}, "
              f"LR: {curr_lr:.6f}")
        
        # 检测并显示学习率变化
        if curr_lr != prev_lr:
            print(f"学习率已从 {prev_lr:.6f} 调整为 {curr_lr:.6f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"模型已保存 (val_loss: {val_loss:.4f})")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # 绘制训练和验证 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.5)
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    
    # 加载最优模型并评估
    best_model = SimpleLSTM()
    best_model.load_state_dict(torch.load("best_model.pt",weights_only=True))
    best_model.to(device)
    
    # 定义MSE和MAE两种损失函数用于最终评估
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    # 评估最终测试集损失
    mse_loss = evaluate(best_model, test_loader, mse_criterion, device)
    mae_loss = evaluate(best_model, test_loader, mae_criterion, device)
    
    print(f"\n[Final Test] MSE Loss: {mse_loss:.4f}")
    print(f"[Final Test] MAE Loss: {mae_loss:.4f}")

if __name__ == '__main__':
    main()
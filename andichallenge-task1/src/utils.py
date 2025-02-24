import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    单个epoch的训练逻辑
    """
    model.train()
    total_loss = 0.0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        preds = model(x)  # [batch_size, 1] (回归)
        y = y.view(-1, 1).float()  # 保证与 preds 同形状 (batch_size,1)

        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    验证/评估集上的评估逻辑
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            y = y.view(-1, 1).float()

            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss
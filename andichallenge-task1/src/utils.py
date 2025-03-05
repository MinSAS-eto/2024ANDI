import torch
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 包装 dataloader 以显示训练进度
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, batch in pbar:
        x, lengths, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, lengths)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        # 在进度条中显示当前 batch 的 loss
        pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device, y_scaler=None):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            x, lengths, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x, lengths)
            
            # 计算归一化空间中的损失（用于显示在进度条）
            loss = criterion(outputs.squeeze(), y)
            
            # 收集预测和目标值，用于后续可能的逆变换
            predictions.append(outputs.cpu())
            targets.append(y.cpu())
            
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': loss.item()})
    
    # 如果提供了y_scaler，则在原始空间计算损失
    if y_scaler is not None:
        # 合并所有预测和目标
        all_preds = torch.cat(predictions)
        all_targets = torch.cat(targets)
        
        # 逆变换回原始空间
        orig_preds = torch.tensor(y_scaler.inverse_transform(
            all_preds.numpy().reshape(-1, 1))).squeeze()
        orig_targets = torch.tensor(y_scaler.inverse_transform(
            all_targets.numpy().reshape(-1, 1))).squeeze()
        
        # 计算原始空间的损失
        orig_loss = criterion(orig_preds, orig_targets)
        
        # 返回两种损失
        norm_loss = running_loss / len(dataloader.dataset)
        return norm_loss, orig_loss.item()
    
    # 否则仅返回归一化空间的损失
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
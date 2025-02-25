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

def evaluate(model, dataloader, criterion, device, threshold=0.1):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # 使用 tqdm 包装 dataloader 以显示验证进度
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            x, lengths, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x, lengths)
            loss = criterion(outputs.squeeze(), y)
            running_loss += loss.item() * x.size(0)
            
            # 计算正确率：当预测值与真实值的绝对误差小于 threshold 时认为预测正确
            preds = outputs.squeeze()
            correct += (torch.abs(preds - y) < threshold).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

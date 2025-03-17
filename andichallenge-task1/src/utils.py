import torch
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 包装 dataloader 以显示训练进度
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)
    for batch_idx, batch in pbar:
        # 修改这里: 只解包两个元素
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # 修改这里: 不再传递lengths参数
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        # 在进度条中显示当前 batch 的 loss
        pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device, y_scaler=None, return_predictions=False):
    """
    评估模型性能
    
    参数:
    - model: 要评估的模型
    - dataloader: 数据加载器
    - criterion: 损失函数
    - device: 计算设备
    - y_scaler: 标签标准化器（如果有）
    - return_predictions: 是否返回预测和目标值，默认为False
    
    返回:
    - 如果return_predictions=False: 仅返回损失值
    - 如果return_predictions=True: 返回(损失值, 预测值, 目标值)元组
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            # 只解包两个元素
            x, y = batch
            x, y = x.to(device), y.to(device)
            # 不再传递lengths参数
            outputs = model(x)
            
            # 计算损失（模型输出直接与未标准化的标签比较）
            loss = criterion(outputs.squeeze(), y)
            
            # 只有在需要返回预测值时才收集
            if return_predictions:
                predictions.append(outputs.cpu())
                targets.append(y.cpu())
            
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': loss.item()})
    
    # 计算整体损失
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # 根据参数决定返回内容
    if return_predictions:
        # 合并所有预测和目标
        all_preds = torch.cat(predictions)
        all_targets = torch.cat(targets)
        return epoch_loss, all_preds.numpy(), all_targets.numpy()
    else:
        # 只返回损失值
        return epoch_loss

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

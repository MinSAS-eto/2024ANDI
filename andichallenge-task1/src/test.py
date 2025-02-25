import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from andi_datasets.datasets_challenge import challenge_theory_dataset
from tqdm import tqdm

class AnDiDataset(Dataset):
    def __init__(self, N, tasks, transform=None):
        super().__init__()
        self.transform = transform
        # 使用传入的 N 和 tasks 参数生成数据集，维度固定为 1
        X1, Y1, _, _, _, _ = challenge_theory_dataset(N=N, tasks=tasks, dimensions=1)
        self.X = X1[0]  # 确保选择正确的维度
        self.Y = Y1[0]  # 确保选择正确的维度
        self.N = N
        self.tasks = tasks

        # 打印生成的数据以进行检查
        print("X1[0] length:", len(X1[0]))
        print("Y1[0] length:", len(Y1[0]))

    def __len__(self):
        # 返回样本数量
        return len(self.X)

    def __getitem__(self, idx):
        # 根据索引获取样本
        x = self.X[idx]
        y = self.Y[idx]
        # 如有转换函数则进行转换
        if self.transform:
            x = self.transform(x)
            
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    
    xs_tensor = []
    lengths_x = []
    for x in xs:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        xs_tensor.append(x)
        lengths_x.append(len(x))
    
    # 对 x 进行 padding
    padded_x = torch.nn.utils.rnn.pad_sequence(xs_tensor, batch_first=True, padding_value=0)
    # 将 y 直接转换为 tensor，不需要 padding
    ys_tensor = torch.tensor(ys, dtype=torch.float32)
    
    return padded_x, lengths_x, ys_tensor

class CNNBiLSTM(nn.Module):
    """
    使用一维卷积 + 双向 LSTM 的示例网络，用于回归任务（预测扩散指数 α）。
    假设输入 shape = [batch_size, T]，其中 T 为 padding 后的最大序列长度。
    """
    def __init__(self,
                 conv_channels=16,
                 lstm_hidden_size=32,
                 lstm_layers=1,
                 bidirectional=True):
        super(CNNBiLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(lstm_hidden_size * num_directions, 1)

    def forward(self, x, lengths):
        """
        x: [batch_size, T]，其中 T 为 padding 后的最大序列长度
        lengths: 原始序列长度列表，长度为 batch_size
        """
        # 将输入扩展为 [batch_size, 1, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)        # => [batch_size, conv_channels, T]
        x = self.relu(x)
        
        # 转换为 LSTM 所需的形状 [batch_size, T, conv_channels]
        x = x.transpose(1, 2)    # => [batch_size, T, conv_channels]
        
        # 为 pack_padded_sequence 做准备：需按照长度降序排列
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=x.device)
        lengths_sorted, sorted_idx = torch.sort(lengths_tensor, descending=True)
        x_sorted = x[sorted_idx]
        
        # pack sequence
        packed_input = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed_input)
        
        # pad packed sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch_size, max_seq_len_in_batch, hidden_size * num_directions]
        
        # 取每个序列最后一个非 padding 的时间步输出
        last_outputs = []
        for i, seq_len in enumerate(lengths_sorted):
            last_outputs.append(output[i, seq_len - 1, :])
        last_outputs = torch.stack(last_outputs, dim=0)
        
        # 恢复原始排序
        _, original_idx = torch.sort(sorted_idx)
        last_outputs = last_outputs[original_idx]
        
        out = self.fc(last_outputs)  # => [batch_size, 1]
        return out
    
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    # 使用 tqdm 包装 dataloader 以显示验证进度
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            x, lengths, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x, lengths)
            loss = criterion(outputs.squeeze(), y)
            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
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
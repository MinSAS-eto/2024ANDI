import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNBiLSTM(nn.Module):
    """
    使用多卷积层 + BatchNorm + 双层 LSTM 的示例网络，用于回归任务（预测扩散指数 α）。
    假设输入 shape = [batch_size, T]，其中 T 为 padding 后的最大序列长度。
    """
    def __init__(self,
                 conv_channels=16,
                 lstm_hidden_size=32,
                 lstm_layers=2,    # 双层 LSTM
                 bidirectional=True):
        super(CNNBiLSTM, self).__init__()
        
        # Mish 激活函数（若你的 PyTorch 版本不支持 nn.Mish，可自行实现）
        self.mish = nn.Mish()
        
        # 第1个卷积层
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # 第2个卷积层
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # 在卷积层后添加 BatchNorm
        self.batchnorm = nn.BatchNorm1d(conv_channels)

        # 在卷积之后添加一个“小”全连接层，对 CNN 提取的特征进行组合
        self.fc_small = nn.Linear(conv_channels, conv_channels)
        
        # 双层 LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(lstm_hidden_size * num_directions, 1)

    def forward(self, x, lengths):
        """
        x: [batch_size, T]，其中 T 为 padding 后的最大序列长度
        lengths: 原始序列长度列表，长度为 batch_size
        """
        # => [batch_size, 1, T] (给 Conv1d 用)
        x = x.unsqueeze(1)
        
        # 第1个卷积 + Mish
        x = self.conv1(x)
        x = self.mish(x)

        # 第2个卷积 + Mish
        x = self.conv2(x)
        x = self.mish(x)

        # BatchNorm 在通道维度上进行归一化
        # x shape 此时是 [batch_size, conv_channels, T]
        x = self.batchnorm(x)

        # 转置成 [batch_size, T, conv_channels] 以适配 LSTM 的输入
        x = x.transpose(1, 2)

        # 通过“小”全连接层组合特征（对每个时间步做映射）
        x = self.fc_small(x)
        x = self.mish(x)

        # 处理 lengths 以 pack_padded_sequence
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=x.device)
        lengths_sorted, sorted_idx = torch.sort(lengths_tensor, descending=True)
        x_sorted = x[sorted_idx]
        
        # pack sequence
        packed_input = pack_padded_sequence(
            x_sorted, 
            lengths_sorted.cpu(), 
            batch_first=True, 
            enforce_sorted=True
        )
        packed_output, _ = self.lstm(packed_input)
        
        # pad packed sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch_size, max_seq_len_in_batch, hidden_size * num_directions]
        
        # 取每个序列最后一个非 padding 的时间步输出
        last_outputs = []
        for i, seq_len in enumerate(lengths_sorted):
            last_outputs.append(output[i, seq_len - 1, :])
        last_outputs = torch.stack(last_outputs, dim=0)
        last_outputs = self.dropout(last_outputs)
        
        # 恢复原始 batch 的顺序
        _, original_idx = torch.sort(sorted_idx)
        last_outputs = last_outputs[original_idx]
        
        # 全连接层得到最终回归输出
        out = self.fc(last_outputs)  # => [batch_size, 1]
        return out
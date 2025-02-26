import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.dropout = nn.Dropout(p=0.5)
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
        last_outputs = self.dropout(last_outputs)
        
        # 恢复原始排序
        _, original_idx = torch.sort(sorted_idx)
        last_outputs = last_outputs[original_idx]
        
        out = self.fc(last_outputs)  # => [batch_size, 1]
        return out
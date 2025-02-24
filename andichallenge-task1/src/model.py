import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    """
    使用一维卷积 + 双向 LSTM 的示例网络，用于回归任务（预测扩散指数 α）。
    假设输入 shape = [batch_size, T]。
    """
    def __init__(self,
                 input_length=50,
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

    def forward(self, x):
        # x: [batch_size, T] => 先变成 [batch_size, 1, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)        # => [batch_size, conv_channels, T]
        x = self.relu(x)
        
        # LSTM需要 [batch_size, seq_len, feature_dim]
        x = x.transpose(1, 2)    # => [batch_size, T, conv_channels]

        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch_size, T, hidden_size * num_directions]

        # 取最后时刻
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_size * num_directions]

        out = self.fc(last_out)  # => [batch_size, 1]
        return out

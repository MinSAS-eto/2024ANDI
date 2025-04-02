import torch
import torch.nn as nn

class BaselineBiLSTM(nn.Module):
    """
    一个简化的双向LSTM模型，用于等长序列回归任务。
    修改后不再需要lengths参数。
    """
    def __init__(self,
                 input_size=1,        # 每个时间步输入的特征维度
                 hidden_size=128,     # LSTM的隐藏层大小
                 num_layers=2,        # LSTM的层数
                 dropout=0.2,         # 在LSTM层间的Dropout
                 bidirectional=True   # 是否使用双向LSTM
                 ):
        super(BaselineBiLSTM, self).__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 定义双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # 线性层：将最后一层的hidden投影到最终输出（回归为一个值）
        # 如果是双向，则隐藏维度 = hidden_size * 2
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)
        
        # 可选的dropout，位于输出全连接前
        self.out_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        简化后的前向传递，不再需要lengths参数
        
        x: [B, T] 或 [B, T, input_size] 的张量（这里默认为 [B, T] -> 需要unsqueeze(-1)）
        返回: [B] 大小的输出（每个batch样本一个回归值）
        """
        
        # 如果x是[B, T]，可以先升维成 [B, T, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # 直接运行LSTM (不再需要pack/unpack操作)
        outputs, (h_n, c_n) = self.lstm(x)
        
        # 提取最后一层的hidden state
        if self.bidirectional:
            # 对双向来说，h_n的形状: [num_layers * 2, B, hidden_size]
            # 取最后一层: indices = (num_layers-1)*2 和 (num_layers-1)*2 + 1
            last_layer_fw = h_n[-2]  # 正向
            last_layer_bw = h_n[-1]  # 反向
            out = torch.cat((last_layer_fw, last_layer_bw), dim=-1)
        else:
            # 单向时: h_n形状 [num_layers, B, hidden_size]
            out = h_n[-1]  # 取最后一层
       
        # out形状: [B, hidden_size * num_directions]
        # 全连接输出
        out = self.out_dropout(out)
        out = self.fc(out)        # [B, 1]
        out = out.squeeze(-1)     # [B]

        return out
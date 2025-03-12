import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力层：使用较少的注意力头(2个)提取不同子空间的特征表示
    """
    def __init__(self, hidden_size, num_heads=2, dropout_rate=0.1):  # 注意力头数改为2
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 添加注意力权重的dropout
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        batch_size, seq_len, _ = hidden_states.size()
        
        # 计算查询、键和值，并分割成多个头
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, T, HD]
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # [B, NH, T, HD]
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)   # [B, NH, T, HD]
        
        # 计算注意力分数
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, NH, T, T]
        
        # 应用softmax获得注意力权重
        attention = F.softmax(energy, dim=-1)  # [B, NH, T, T]
        attention = self.attn_dropout(attention)
        
        # 计算加权上下文向量
        context = torch.matmul(attention, v)  # [B, NH, T, HD]
        
        # 重排并合并多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [B, T, H]
        
        # 最终线性投影
        context = self.out_proj(context)
        context = self.output_dropout(context)
        
        return context, attention   


class SimpleLSTMWithAttention(nn.Module):
    """
    简化版带双头自注意力机制的LSTM网络，移除多尺度特征融合，优化用于等长序列回归任务。
    """
    def __init__(self,
                 input_size=1,
                 lstm_hidden_size=256,
                 lstm_layers=3,
                 bidirectional=True,
                 dropout_rate=0.3):
        super(SimpleLSTMWithAttention, self).__init__()
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.LayerNorm(input_size * 2),
            nn.GELU()
        )
        
        # 深层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        direction_factor = 2 if bidirectional else 1
        hidden_dim = lstm_hidden_size * direction_factor
        
        # 层标准化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 特征级dropout
        self.feature_dropout = nn.Dropout1d(dropout_rate * 0.7)
        
        # 双头自注意力层
        self.attention = MultiHeadSelfAttention(
            hidden_dim, 
            num_heads=2,  # 注意力头数减为2个
            dropout_rate=dropout_rate * 0.6
        )
        
        # 双层全连接网络，带残差结构
        fc_hidden = hidden_dim // 2
        self.fc1 = nn.Linear(hidden_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.fc_residual = nn.Linear(hidden_dim, fc_hidden // 2)  # 残差连接
        
        # 批标准化
        self.batch_norm1 = nn.BatchNorm1d(fc_hidden)
        self.batch_norm2 = nn.BatchNorm1d(fc_hidden // 2)
        
        # 输出层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(fc_hidden // 2, 1)
        
        # 激活函数
        self.activation = nn.Mish()
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """优化参数初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)  # LSTM使用正交初始化
                elif 'fc' in name or 'linear' in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')  # He初始化
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(-1)  # [B, T, 1]
        
        # 输入投影
        x = self.input_proj(x)
        
        # 直接运行LSTM
        lstm_output, _ = self.lstm(x)  # [B, T, H]
        
        # 应用层标准化
        lstm_output = self.layer_norm1(lstm_output)

        # 移除多尺度特征融合部分
        
        # 应用特征级dropout
        lstm_output_reshaped = lstm_output.permute(0, 2, 1)  # [B, H, T]
        lstm_output_reshaped = self.feature_dropout(lstm_output_reshaped)
        lstm_output = lstm_output_reshaped.permute(0, 2, 1)  # [B, T, H]
        
        # 应用双头自注意力
        context, _ = self.attention(lstm_output)  # [B, T, H]
        
        # 残差连接 + 层标准化
        context = context + lstm_output  # 残差连接
        context = self.layer_norm2(context)  # 再次标准化
        
        # 简化的全局池化 (直接平均)
        pooled = context.mean(dim=1)  # [B, H]
        
        # 层次化全连接网络，带残差结构
        fc1_out = self.fc1(pooled)  # [B, H/2]
        fc1_out = self.batch_norm1(fc1_out)
        fc1_out = self.activation(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = self.fc2(fc1_out)  # [B, H/4]
        fc2_out = self.batch_norm2(fc2_out)
        
        # 残差连接（降维）
        residual = self.fc_residual(pooled)  # [B, H/4]
        fc2_out = fc2_out + residual  # 残差连接
        fc2_out = self.activation(fc2_out)
        
        # 最终预测
        out = self.dropout(fc2_out)
        out = self.fc_out(out).squeeze(-1)  # [B]

        return out


# 保留原始SimpleLSTM以兼容现有代码
class SimpleLSTM(SimpleLSTMWithAttention):
    pass
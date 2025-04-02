import torch
import torch.nn as nn
import torch.nn.functional as F

# 多头自注意力层（注意力头数设置为2）
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        batch_size, seq_len, _ = hidden_states.size()
        # 计算 Q, K, V 并分头
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, T, HD]
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # [B, NH, T, HD]
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, NH, T, HD]
        
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, NH, T, T]
        attention = F.softmax(energy, dim=-1)
        attention = self.attn_dropout(attention)
        
        context = torch.matmul(attention, v)  # [B, NH, T, HD]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [B, T, H]
        context = self.out_proj(context)
        context = self.output_dropout(context)
        return context, attention

# 多尺度 CNN 模块：使用不同尺寸卷积核提取特征
class MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MultiScaleCNNBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k-1)//2))
            self.layernorms.append(nn.LayerNorm(out_channels))
        self.activation = nn.Mish()
    
    def forward(self, x):
        # x: [B, in_channels, T]
        outs = []
        for conv, ln in zip(self.convs, self.layernorms):
            conv_out = conv(x)  # [B, out_channels, T]
            conv_out = conv_out.transpose(1, 2)  # [B, T, out_channels]
            conv_out = ln(conv_out)
            conv_out = conv_out.transpose(1, 2)  # [B, out_channels, T]
            conv_out = self.activation(conv_out)
            outs.append(conv_out)
        out = torch.cat(outs, dim=1)  # 拼接后通道数为 out_channels * len(kernel_sizes)
        return out

# 简单的残差卷积块
class ResidualCNNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualCNNBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln = nn.LayerNorm(channels)
        self.activation = nn.Mish()
    
    def forward(self, x):
        # x: [B, channels, T]
        out = self.conv(x)  # [B, channels, T]
        out = out.transpose(1, 2)  # [B, T, channels]
        out = self.ln(out)
        out = out.transpose(1, 2)  # [B, channels, T]
        out = self.activation(out)
        return x + out  # 残差连接

# 优化用于等长序列的模型（移除变长序列处理）
class EquilenCNNBiLSTMAttention(nn.Module):
    def __init__(self,
                 input_size=1,
                 conv_channels=256,
                 kernel_sizes=[3,5,7],
                 num_residual_blocks=1,
                 lstm_hidden_size=256,
                 lstm_layers=3,
                 bidirectional=True,
                 dropout_rate=0.3):
        super(EquilenCNNBiLSTMAttention, self).__init__()
        
        self.mish = nn.Mish()
        num_kernels = len(kernel_sizes)
        # 多尺度 CNN 模块（输入通道为1）
        self.multi_scale_cnn = MultiScaleCNNBlock(in_channels=input_size, out_channels=conv_channels, kernel_sizes=kernel_sizes)
        # 残差连接，将原始输入通过1×1卷积映射至多尺度输出的通道数
        self.residual_conv = nn.Conv1d(input_size, conv_channels * num_kernels, kernel_size=1)
        # 融合层：1×1卷积降维，输出通道数为 conv_channels
        self.fusion_conv = nn.Conv1d(conv_channels * num_kernels, conv_channels, kernel_size=1)
        self.layernorm_after_fusion = nn.LayerNorm(conv_channels)
        
        # 额外的 CNN 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualCNNBlock(conv_channels, kernel_size=3) for _ in range(num_residual_blocks)
        ])
        
        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.num_directions = 2 if bidirectional else 1
        hidden_dim = lstm_hidden_size * self.num_directions
        
        # 双头自注意力层
        self.attention = MultiHeadSelfAttention(hidden_size=hidden_dim, num_heads=2, dropout_rate=dropout_rate * 0.6)
        self.layernorm_post_attn = nn.LayerNorm(hidden_dim)
        
        # 跨模块残差：对 CNN 全局特征进行投影
        self.cnn_res_proj = nn.Linear(conv_channels, hidden_dim)
        
        # 全连接层（带残差结构）
        fc_hidden = hidden_dim // 2
        self.fc1 = nn.Linear(hidden_dim, fc_hidden)
        self.batch_norm1 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.batch_norm2 = nn.BatchNorm1d(fc_hidden // 2)
        self.fc_residual = nn.Linear(hidden_dim, fc_hidden // 2)
        self.fc_out = nn.Linear(fc_hidden // 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Mish()
        
        self._init_weights()
    
    def _init_weights(self):
        # 对卷积、全连接以及LSTM参数进行初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'fc' in name or 'linear' in name or 'conv' in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        简化的前向传播函数，移除变长序列处理
        x: [B, T]，T 为固定序列长度
        """
        # 1. 多尺度 CNN 特征提取
        # 将输入扩展至 [B, 1, T]
        x = x.unsqueeze(1)
        cnn_features = self.multi_scale_cnn(x)  # [B, conv_channels * num_kernels, T]
        residual = self.residual_conv(x)         # [B, conv_channels * num_kernels, T]
        cnn_features = cnn_features + residual
        fused = self.fusion_conv(cnn_features)     # [B, conv_channels, T]
        fused = self.mish(fused)
        
        # 转换维度至 [B, T, conv_channels] 以便做 LayerNorm
        fused = fused.transpose(1, 2)
        fused = self.layernorm_after_fusion(fused)
        fused = self.activation(fused)
        
        # 保存 CNN 全局信息（全局平均池化，用于跨模块残差）
        cnn_avg = fused.mean(dim=1)  # [B, conv_channels]
        
        # 经过额外的 CNN 残差块
        fused = fused.transpose(1, 2)  # 转回 [B, conv_channels, T]
        for block in self.residual_blocks:
            fused = block(fused)
        fused = fused.transpose(1, 2)  # [B, T, conv_channels]
        
        # 2. BiLSTM 层 - 简化版，不需要处理变长序列
        lstm_out, _ = self.lstm(fused)  # [B, T, hidden_dim]
        
        # 3. 双头自注意力：对 LSTM 输出进行上下文整合
        attn_out, _ = self.attention(lstm_out)  # [B, T, hidden_dim]
        # 残差连接 + 层归一化
        attn_out = self.layernorm_post_attn(attn_out + lstm_out)
        
        # 4. 简化的全局池化 - 直接平均池化
        pooled = attn_out.mean(dim=1)  # [B, hidden_dim]
        
        # 5. 跨模块残差：将 CNN 全局特征投影后与 LSTM全局特征相加
        cnn_proj = self.cnn_res_proj(cnn_avg)  # [B, hidden_dim]
        pooled = pooled + cnn_proj
        
        # 6. 全连接层（带残差连接）
        fc1_out = self.fc1(pooled)
        fc1_out = self.batch_norm1(fc1_out)
        fc1_out = self.activation(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.batch_norm2(fc2_out)
        residual_fc = self.fc_residual(pooled)
        fc2_out = fc2_out + residual_fc
        fc2_out = self.activation(fc2_out)
        
        fc2_out = self.dropout(fc2_out)
        out = self.fc_out(fc2_out).squeeze(-1)  # 输出形状 [B]
        
        return out
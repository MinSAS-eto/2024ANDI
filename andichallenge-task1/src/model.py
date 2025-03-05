import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MultiScaleCNNBlock(nn.Module):
    """
    多尺度 CNN 模块：使用不同尺寸的卷积核提取多尺度特征，
    并在每个卷积后使用 LayerNorm（在时间步维度上标准化）。
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super(MultiScaleCNNBlock, self).__init__()
        self.convs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        for k in kernel_sizes:
            # 使用 padding 保持序列长度不变
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k-1)//2))
            # LayerNorm 对于卷积输出先将 [B, C, T] 转为 [B, T, C] 后进行归一化
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
        # 拼接多个尺度的特征，通道数变为 out_channels * len(kernel_sizes)
        out = torch.cat(outs, dim=1)
        return out

class ResidualCNNBlock(nn.Module):
    """
    一个简单的残差卷积块：
    对输入进行卷积、LayerNorm 和激活，然后与原始输入相加。
    """
    def __init__(self, channels, kernel_size=3):
        super(ResidualCNNBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln = nn.LayerNorm(channels)  # 后续需要转换维度为 [B, T, channels]
        self.activation = nn.Mish()
        
    def forward(self, x):
        # x: [B, channels, T]
        out = self.conv(x)  # [B, channels, T]
        out = out.transpose(1, 2)  # [B, T, channels]
        out = self.ln(out)
        out = out.transpose(1, 2)  # [B, channels, T]
        out = self.activation(out)
        return x + out  # 残差连接

class AttentionLayer(nn.Module):
    """
    简单的注意力机制，对 LSTM 输出每个时间步计算注意力得分，
    并利用 softmax 加权求和得到上下文向量。
    """
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, lstm_outputs, lengths):
        # lstm_outputs: [B, T, D]
        attn_scores = self.attention(lstm_outputs)  # [B, T, 1]
        attn_scores = attn_scores.squeeze(-1)  # [B, T]
        
        # 构造 mask，防止对 padding 部分计算注意力
        max_len = lstm_outputs.size(1)
        mask = torch.arange(max_len, device=lstm_outputs.device).unsqueeze(0) < torch.tensor(lengths, device=lstm_outputs.device).unsqueeze(1)
        attn_scores[~mask] = float('-inf')
        
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T]
        attn_weights = attn_weights.unsqueeze(-1)  # [B, T, 1]
        context = torch.sum(lstm_outputs * attn_weights, dim=1)  # [B, D]
        return context

class ImprovedCNNBiLSTM(nn.Module):
    """
    改进版网络：
      1. 使用多尺度 CNN 提取特征，并添加残差连接及额外的 CNN 层加深网络；
      2. 使用 LayerNorm 对特征进行归一化；
      3. LSTM 层增加到 4 层，并在最后结合了来自 CNN 部分的全局特征进行残差融合；
      4. LSTM 层后添加注意力机制获得全局上下文向量，
         最后经过 dropout 和全连接层进行回归预测。
    
    输入: x -> [B, T] (T 为 padding 后的最大序列长度)
    """
    def __init__(self,
                 conv_channels=64,
                 lstm_hidden_size=128,
                 lstm_layers=4,   # 增加 LSTM 层数
                 bidirectional=True,
                 dropout_rate=0.5,
                 kernel_sizes=[3,5,7],
                 num_residual_blocks=1):  # 增加额外 CNN 残差块的数量
        super(ImprovedCNNBiLSTM, self).__init__()
        
        self.mish = nn.Mish()
        # 多尺度 CNN 模块：输入通道为1
        self.multi_scale_cnn = MultiScaleCNNBlock(in_channels=1, out_channels=conv_channels, kernel_sizes=kernel_sizes)
        
        # 残差连接：将原始输入经过 1×1 卷积映射到多尺度输出相同的通道数
        self.residual_conv = nn.Conv1d(1, conv_channels * len(kernel_sizes), kernel_size=1)
        
        # 融合层：1×1 卷积融合多尺度特征，输出通道数降为 conv_channels
        self.fusion_conv = nn.Conv1d(conv_channels * len(kernel_sizes), conv_channels, kernel_size=1)
        # LayerNorm 用于融合后的特征（需先转置到 [B, T, conv_channels]）
        self.layernorm_after_fusion = nn.LayerNorm(conv_channels)
        
        # 额外的 CNN 残差块
        self.num_residual_blocks = num_residual_blocks
        self.residual_blocks = nn.ModuleList([
            ResidualCNNBlock(conv_channels, kernel_size=3) for _ in range(num_residual_blocks)
        ])
        
        # LSTM 层：输入维度 = conv_channels，层数增加至 lstm_layers
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        num_directions = 2 if bidirectional else 1
        
        # 注意力层：对 LSTM 输出进行注意力加权
        self.attention_layer = AttentionLayer(lstm_hidden_size * num_directions)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size * num_directions, 1)
        
        # 跨模块残差连接：
        # 对 CNN 融合后的特征（全局平均池化）进行投影，以匹配 LSTM 输出的维度
        self.cnn_res_proj = nn.Linear(conv_channels, lstm_hidden_size * num_directions)
        
        self.init_weights()
    
    def init_weights(self):
        # 对卷积层和全连接层进行合适的初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, lengths):
        # 输入 x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        
        # 多尺度 CNN 特征提取
        cnn_features = self.multi_scale_cnn(x)  # [B, conv_channels * len(kernel_sizes), T]
        # 残差连接：原始输入经过 1×1 卷积映射
        residual = self.residual_conv(x)  # [B, conv_channels * len(kernel_sizes), T]
        cnn_features = cnn_features + residual
        
        # 融合多尺度特征
        fused = self.mish(self.fusion_conv(cnn_features))  # [B, conv_channels, T]
        
        # 转换维度以应用 LayerNorm：变为 [B, T, conv_channels]
        fused = fused.transpose(1, 2)
        fused = self.layernorm_after_fusion(fused)
        fused = self.mish(fused)  # [B, T, conv_channels]
        
        # 保存 CNN 融合特征的全局信息（全局平均池化），用于残差连接
        cnn_avg = fused.mean(dim=1)  # [B, conv_channels]
        
        # 将 fused 转换回 [B, conv_channels, T] 以便通过额外的 CNN 残差块
        fused = fused.transpose(1, 2)  # [B, conv_channels, T]
        for block in self.residual_blocks:
            fused = block(fused)  # [B, conv_channels, T]
        # 转换回 [B, T, conv_channels]，作为 LSTM 输入
        fused = fused.transpose(1, 2)  # [B, T, conv_channels]
        
        # 处理变长序列：pack 之后送入 LSTM
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=fused.device)
        packed_input = pack_padded_sequence(fused, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: [B, T, lstm_hidden_size * num_directions]
        
        # 注意力机制整合 LSTM 输出，得到上下文向量
        context = self.attention_layer(output, lengths)  # [B, lstm_hidden_size * num_directions]
        
        # 跨模块残差：将 CNN 全局特征投影后与上下文相加
        cnn_res = self.cnn_res_proj(cnn_avg)  # [B, lstm_hidden_size * num_directions]
        context = context + cnn_res
        
        context = self.dropout(context)
        out = self.fc(context)  # [B, 1]
        return out.squeeze(-1)  # 将形状从[B, 1]变为[B]
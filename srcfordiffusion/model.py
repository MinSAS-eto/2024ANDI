import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================
# 注意力和Transformer组件
# ==============================

class SelfAttention(nn.Module):
    """轨迹序列的自注意力层"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, embed_dim)
        residual = x
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        out = self.layer_norm(residual + attn_output)
        return out

class FeedForward(nn.Module):
    """带残差连接的简单前馈网络"""
    def __init__(self, embed_dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        out = self.layer_norm(residual + x)
        return out

class TransformerEncoderLayer(nn.Module):
    """带自注意力的Transformer编码器层"""
    def __init__(self, embed_dim, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
    
    def forward(self, x):
        x = self.self_attn(x)
        x = self.feed_forward(x)
        return x

class PositionalEncoding(nn.Module):
    """Transformer模型的位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# ==============================
# 扩散模型组件
# ==============================

class TimeEmbedding(nn.Module):
    """时间步嵌入层"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(1, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, t):
        # t: (batch_size,)
        t = t.unsqueeze(-1).float()  # (batch_size, 1)
        t = self.linear1(t)
        t = F.silu(t)
        t = self.linear2(t)
        return t  # (batch_size, embedding_dim)

class ExponentEmbedding(nn.Module):
    """扩散指数嵌入层"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(1, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, alpha):
        # alpha: (batch_size, 1)
        alpha = alpha.float()
        alpha = self.linear1(alpha)
        alpha = F.silu(alpha)
        alpha = self.linear2(alpha)
        return alpha  # (batch_size, embedding_dim)

class ModelIDEmbedding(nn.Module):
    """模型ID嵌入层"""
    def __init__(self, num_models, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_models, embedding_dim)
    
    def forward(self, model_id):
        # model_id: (batch_size, 1)
        return self.embedding(model_id.squeeze(-1))  # (batch_size, embedding_dim)

class TrajectoryAttentionBlock(nn.Module):
    """用注意力处理轨迹数据的块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, seq_len, heads=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 卷积层
        self.layer1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.layer2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # 标准化
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # 注意力机制
        self.attention = SelfAttention(out_channels, num_heads=heads)
    
    def forward(self, x, time_emb):
        # x: (batch_size, in_channels, seq_len)
        # time_emb: (batch_size, time_emb_dim)
        
        # 第一层
        h = self.layer1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # 时间嵌入
        time_emb = self.time_mlp(time_emb)  # (batch_size, out_channels)
        h = h + time_emb.unsqueeze(-1)  # 添加到每个位置
        
        # 转置以进行注意力 [batch, channel, sequence] -> [batch, sequence, channel]
        h_attn = h.permute(0, 2, 1)
        
        # 应用注意力
        h_attn = self.attention(h_attn)
        
        # 转置回来 [batch, sequence, channel] -> [batch, channel, sequence]
        h_attn = h_attn.permute(0, 2, 1)
        
        # 添加第一层输出的残差连接
        h = h + h_attn
        
        # 第二个卷积层
        h = self.layer2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # 输入的残差连接
        return h + self.res_conv(x)

class TransformerTrajectoryBlock(nn.Module):
    """处理轨迹数据的Transformer块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, seq_len, 
                n_heads=4, dropout=0.1):
        super().__init__()
        
        # 时间嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 输入投影
        self.in_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Transformer编码器层
        self.transformer = TransformerEncoderLayer(
            embed_dim=out_channels,
            num_heads=n_heads,
            ff_dim=out_channels*4,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(out_channels, max_len=seq_len)
        
        # 输出投影
        self.out_proj = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 残差连接
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # 层标准化
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, time_emb):
        # x: (batch_size, in_channels, seq_len)
        # time_emb: (batch_size, time_emb_dim)
        
        # 保存残差
        residual = x
        
        # 投影输入
        h = self.in_proj(x)  # (batch_size, out_channels, seq_len)
        
        # 时间嵌入
        time_feat = self.time_mlp(time_emb)  # (batch_size, out_channels)
        h = h + time_feat.unsqueeze(-1)  # 添加到每个位置
        
        # 转置用于transformer [batch, channel, sequence] -> [batch, sequence, channel]
        h = h.permute(0, 2, 1)  # (batch_size, seq_len, out_channels)
        
        # 添加位置编码
        h = self.pos_encoding(h)
        
        # 应用transformer
        h = self.transformer(h)
        
        # 转置回来 [batch, sequence, channel] -> [batch, channel, sequence]
        h = h.permute(0, 2, 1)  # (batch_size, out_channels, seq_len)
        
        # 输出投影
        h = self.out_proj(h)
        
        # 残差连接
        return h + self.res_conv(residual)

class TrajectoryBlock(nn.Module):
    """处理轨迹数据的块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, seq_len=None, n_heads=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.layer1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.layer2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
    
    def forward(self, x, time_emb):
        # x: (batch_size, in_channels, seq_len)
        # time_emb: (batch_size, time_emb_dim)
        
        # 第一层
        h = self.layer1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # 时间嵌入
        time_emb = self.time_mlp(time_emb)  # (batch_size, out_channels)
        h = h + time_emb.unsqueeze(-1)  # 添加到每个位置
        
        # 第二层
        h = self.layer2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # 残差连接
        return h + self.res_conv(x)

class ConditionalTrajectoryDiffusionModel(nn.Module):
    """带条件输入和transformer/注意力组件的轨迹扩散模型"""
    def __init__(self, input_dim=1, hidden_dims=[32, 64, 128, 256], time_emb_dim=32, 
                condition_on_exponent=True, condition_on_model_id=False, num_models=5,
                seq_len=100, use_transformer=True, use_attention=True, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.condition_on_exponent = condition_on_exponent
        self.condition_on_model_id = condition_on_model_id
        self.use_transformer = use_transformer
        self.use_attention = use_attention
        self.seq_len = seq_len
        
        # 时间步嵌入
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # 基于条件标志的附加嵌入
        cond_dim = time_emb_dim
        if condition_on_exponent:
            self.exponent_embed = ExponentEmbedding(time_emb_dim)
            cond_dim += time_emb_dim
        
        if condition_on_model_id:
            self.model_id_embed = ModelIDEmbedding(num_models, time_emb_dim)
            cond_dim += time_emb_dim
        
        # 下采样块
        self.input_conv = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        
        # 选择使用哪种块类型
        block_type = TransformerTrajectoryBlock if use_transformer else (
            TrajectoryAttentionBlock if use_attention else TrajectoryBlock
        )
        
        in_channels = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.down_blocks.append(
                block_type(in_channels, dim, cond_dim, seq_len, n_heads=n_heads)
            )
            in_channels = dim
        
        # 中间块 - 始终使用transformer作为中间块以获得更好的全局上下文
        self.middle_block = TransformerTrajectoryBlock(
            hidden_dims[-1], hidden_dims[-1], cond_dim, seq_len, n_heads=n_heads
        )
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        for dim in reversed(hidden_dims[:-1]):
            self.up_blocks.append(
                block_type(hidden_dims[-1], dim, cond_dim, seq_len, n_heads=n_heads)
            )
            hidden_dims[-1] = dim
        
        # 输出层
        self.output_conv = nn.Conv1d(hidden_dims[0], input_dim, kernel_size=3, padding=1)
    
    def get_condition_embedding(self, t, exponent=None, model_id=None):
        """合并所有条件嵌入"""
        # 时间嵌入（始终使用）
        t_emb = self.time_embed(t)
        
        embeds = [t_emb]
        
        # 如果启用，添加指数嵌入
        if self.condition_on_exponent and exponent is not None:
            exponent_emb = self.exponent_embed(exponent)
            embeds.append(exponent_emb)
            
        # 如果启用，添加模型ID嵌入
        if self.condition_on_model_id and model_id is not None:
            model_id_emb = self.model_id_embed(model_id)
            embeds.append(model_id_emb)
        
        # 连接所有嵌入
        return torch.cat(embeds, dim=1)
    
    def forward(self, x, t, exponent=None, model_id=None):
        """
        参数:
            x: 有噪声的轨迹 (batch_size, seq_len, input_dim)
            t: 时间步 (batch_size,)
            exponent: 扩散指数 (batch_size, 1) [可选]
            model_id: 模型ID (batch_size, 1) [可选]
        
        返回:
            预测的噪声或轨迹
        """
        # 改变形状为 (batch_size, input_dim, seq_len) 用于1D卷积
        x = x.permute(0, 2, 1)
        
        # 获取组合条件嵌入
        cond_emb = self.get_condition_embedding(t, exponent, model_id)
        
        # 初始投影
        h = self.input_conv(x)
        
        # 下采样块
        residuals = [h]
        for block in self.down_blocks:
            h = block(h, cond_emb)
            residuals.append(h)
        
        # 中间块
        h = self.middle_block(h, cond_emb)
        
        # 上采样块带跳跃连接
        for block in self.up_blocks:
            h = block(h + residuals.pop(), cond_emb)
        
        # 输出投影
        h = self.output_conv(h)
        
        # 返回原始形状 (batch_size, seq_len, input_dim)
        return h.permute(0, 2, 1)
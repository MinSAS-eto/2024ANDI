import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 保留CNN-BiLSTM模型的原始组件
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
        batch_size, seq_len, _ = hidden_states.size()
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        attention = self.attn_dropout(attention)
        
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        context = self.out_proj(context)
        context = self.output_dropout(context)
        return context, attention

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
        outs = []
        for conv, ln in zip(self.convs, self.layernorms):
            conv_out = conv(x)
            conv_out = conv_out.transpose(1, 2)
            conv_out = ln(conv_out)
            conv_out = conv_out.transpose(1, 2)
            conv_out = self.activation(conv_out)
            outs.append(conv_out)
        out = torch.cat(outs, dim=1)
        return out

class ResidualCNNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualCNNBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.ln = nn.LayerNorm(channels)
        self.activation = nn.Mish()
    
    def forward(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.ln(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        return x + out


# 受TimeDiT启发的扩散过程组件

class SinusoidalEmbedding(nn.Module):
    """
    扩散时间步的正弦嵌入，类似于Transformer中的位置编码
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaptiveLayerNorm(nn.Module):
    """
    使用条件信息的自适应层归一化
    受TimeDiT中AdaLN条件注入的启发
    """
    def __init__(self, dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2)
        )
    
    def forward(self, x, condition):
        x_norm = self.norm(x)
        # 从条件中获取缩放和偏移参数
        scale_shift = self.condition_proj(condition)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x_norm * (1 + scale) + shift


# 修改后的主模型：用于异常扩散指数预测的CNN-BiLSTM-Diffusion
class CNNBiLSTMDiffusionExponent(nn.Module):
    def __init__(self,
                 input_size=1,
                 conv_channels=256,
                 kernel_sizes=[3,5,7],
                 num_residual_blocks=1,
                 lstm_hidden_size=256,
                 lstm_layers=3,
                 bidirectional=True,
                 dropout_rate=0.3,
                 time_embedding_dim=128,
                 sequence_length=100,
                 exponent_range=(0.1, 2.0)):  # 异常扩散指数的范围
        super(CNNBiLSTMDiffusionExponent, self).__init__()
        
        self.mish = nn.Mish()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.exponent_range = exponent_range  # 存储指数范围用于缩放
        num_kernels = len(kernel_sizes)
        
        # 扩散时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )
        
        # 多尺度CNN模块
        self.multi_scale_cnn = MultiScaleCNNBlock(in_channels=input_size, out_channels=conv_channels, kernel_sizes=kernel_sizes)
        self.residual_conv = nn.Conv1d(input_size, conv_channels * num_kernels, kernel_size=1)
        self.fusion_conv = nn.Conv1d(conv_channels * num_kernels, conv_channels, kernel_size=1)
        
        # 条件自适应LayerNorm
        self.layernorm_after_fusion = AdaptiveLayerNorm(conv_channels, time_embedding_dim)
        
        # 残差CNN块
        self.residual_blocks = nn.ModuleList([
            ResidualCNNBlock(conv_channels, kernel_size=3) for _ in range(num_residual_blocks)
        ])
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.num_directions = 2 if bidirectional else 1
        hidden_dim = lstm_hidden_size * self.num_directions
        
        # 带自适应归一化的注意力机制
        self.attention = MultiHeadSelfAttention(hidden_size=hidden_dim, num_heads=2, dropout_rate=dropout_rate * 0.6)
        self.layernorm_post_attn = AdaptiveLayerNorm(hidden_dim, time_embedding_dim)
        
        # 特征处理层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.features_norm = nn.LayerNorm(hidden_dim // 2)
        
        # 修改后的输出：不是预测噪声，而是预测扩散指数
        # 现在我们使用全局池化来为每个序列获取单个指数值
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 用于指数预测的输出投影（每个序列一个值）
        self.fc_exponent = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.Mish(),
            nn.Linear(hidden_dim // 4, 1)  # 扩散指数的单个值
        )
        
        # 用于去噪扩散中的噪声预测
        self.fc_out = nn.Linear(hidden_dim // 2, input_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Mish()
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif ('fc' in name or 'linear' in name or 'conv' in name):
                    if param.dim() >= 2:  # 确保参数至少有2维
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    else:  # 对于1维参数
                        nn.init.normal_(param, mean=0.0, std=0.02)
                else:  # 其他权重参数
                    if param.dim() >= 2:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, timesteps, predict_exponent=False):
        """
        CNN-BiLSTM-Diffusion模型用于扩散指数预测的前向函数
        x: 输入轨迹 [B, T] 或 [B, T, C]
        timesteps: 扩散时间步 [B]
        predict_exponent: 是预测扩散指数(True)还是去噪(False)
        """
        # 确保x具有正确的形状 [B, T, C]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        
        batch_size, seq_len, channels = x.shape
        
        # 检查序列长度是否匹配预期
        if seq_len != self.sequence_length:
            # 如果不匹配，截断或填充
            if seq_len > self.sequence_length:
                x = x[:, :self.sequence_length, :]  # 截断
            else:
                # 填充（用零）
                padding = torch.zeros(batch_size, self.sequence_length - seq_len, channels, device=x.device)
                x = torch.cat([x, padding], dim=1)
            seq_len = self.sequence_length
        
        # 处理时间嵌入
        time_emb = self.time_embed(timesteps)  # [B, time_emb_dim]
        
        # 准备CNN输入 [B, C, T]
        x_cnn = x.transpose(1, 2)  # [B, C, T]
        
        # 1. 多尺度CNN特征提取
        cnn_features = self.multi_scale_cnn(x_cnn)  # [B, conv_channels * num_kernels, T]
        residual = self.residual_conv(x_cnn)  # [B, conv_channels * num_kernels, T]
        cnn_features = cnn_features + residual
        fused = self.fusion_conv(cnn_features)  # [B, conv_channels, T]
        fused = self.mish(fused)
        
        # 转换为 [B, T, C] 用于LayerNorm和自适应条件
        fused = fused.transpose(1, 2)  # [B, T, conv_channels]
        
        # 使用时间嵌入条件应用自适应LayerNorm
        fused = self.layernorm_after_fusion(fused, time_emb.unsqueeze(1).expand(-1, seq_len, -1))
        fused = self.activation(fused)
        
        # 保存CNN全局信息用于跨模块残差
        cnn_avg = fused.mean(dim=1)  # [B, conv_channels]
        
        # 使用残差CNN块处理
        fused = fused.transpose(1, 2)  # 返回 [B, C, T]
        for block in self.residual_blocks:
            fused = block(fused)
        fused = fused.transpose(1, 2)  # 返回 [B, T, C]
        
        # 2. BiLSTM层
        lstm_out, _ = self.lstm(fused)  # [B, T, hidden_dim]
        
        # 3. 带时间自适应归一化的多头注意力
        attn_out, _ = self.attention(lstm_out)  # [B, T, hidden_dim]
        
        # 使用时间嵌入应用自适应LayerNorm
        attn_out = self.layernorm_post_attn(attn_out + lstm_out, time_emb.unsqueeze(1).expand(-1, seq_len, -1))
        
        # 处理特征
        features = self.fc1(attn_out)  # [B, T, fc_hidden]
        features = self.features_norm(features)
        features = self.activation(features)
        features = self.dropout(features)
        
        if predict_exponent:
            # 对时间维度进行全局池化以进行指数预测
            # 首先转置使时间维度在最后一个位置用于池化
            pooled_features = features.transpose(1, 2)  # [B, fc_hidden, T]
            pooled_features = self.global_pool(pooled_features).squeeze(-1)  # [B, fc_hidden]
            
            # 预测扩散指数
            exponent = self.fc_exponent(pooled_features)  # [B, 1]
            
            # 缩放到异常扩散的适当范围（例如0.1到2.0）
            min_val, max_val = self.exponent_range
            exponent = torch.sigmoid(exponent) * (max_val - min_val) + min_val
            
            return exponent
        else:
            # 用于去噪：预测每个时间步的噪声
            noise_pred = self.fc_out(features)  # [B, T, input_size]
            return noise_pred
    
    def predict_exponent(self, trajectory):
        """
        直接从轨迹预测异常扩散指数
        trajectory: 输入轨迹 [B, T] 或 [B, T, C]
        返回: 在exponent_range指定范围内的异常扩散指数
        """
        # 确保轨迹具有正确的形状 [B, T, C]
        if len(trajectory.shape) == 2:
            trajectory = trajectory.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        
        # 使用零时间步进行直接预测（无扩散过程）
        batch_size = trajectory.shape[0]
        t = torch.zeros(batch_size, device=trajectory.device).long()
        
        # 设置predict_exponent=True以获取指数预测
        exponent = self.forward(trajectory, t, predict_exponent=True)
        
        return exponent


# 用于异常扩散指数预测的增强扩散模型包装器
class AnomalousDiffusionExponentModel:
    def __init__(self, model, beta_schedule='linear', timesteps=1000, device='cuda'):
        """
        用于训练和采样的扩散模型包装器，具有扩散指数预测功能
        model: 具有指数预测能力的CNN-BiLSTM-Diffusion骨干网络
        beta_schedule: 噪声方差调度（'linear'，'cosine'等）
        timesteps: 扩散步骤数
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # 定义前向扩散过程的beta调度
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        elif beta_schedule == 'cosine':
            # 来自改进DDPM的余弦调度
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0, 0.999)
        
        # 预计算扩散参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散q(x_t | x_{t-1})和其他参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 计算后验q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].view(1), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_0, t):
        """
        前向扩散过程：q(x_t | x_0)
        x_0: [B, T] 或 [B, T, C] - 原始干净数据
        t: [B] - 时间步
        """
        # 确保x_0具有正确的形状 [B, T, C]
        if len(x_0.shape) == 2:
            x_0 = x_0.unsqueeze(-1)  # [B, T] -> [B, T, 1]
            
        noise = torch.randn_like(x_0)
        x_t = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        return x_t, noise
    
    def train_step(self, x_0, target_exponents=None):
        """
        扩散模型训练步骤，具有可选的指数预测
        x_0: [B, T] 或 [B, T, C] - 原始干净数据
        target_exponents: [B, 1] - 用于监督学习的可选地面真实扩散指数
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # 确保x_0具有正确的形状
        if len(x_0.shape) == 2:
            x_0 = x_0.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        
        # 应用前向扩散以获取噪声样本和真实噪声
        x_t, true_noise = self.add_noise(x_0, t)
        
        # 用模型预测噪声
        noise_pred = self.model(x_t, t, predict_exponent=False)
        
        # 计算去噪损失
        denoising_loss = F.mse_loss(noise_pred, true_noise)
        
        # 如果提供了目标指数，也预测并训练指数
        if target_exponents is not None:
            # 从原始轨迹预测指数
            pred_exponents = self.model(x_0, torch.zeros_like(t), predict_exponent=True)
            
            # 指数预测损失
            exponent_loss = F.mse_loss(pred_exponents, target_exponents)
            
            # 组合损失（加权和）
            total_loss = denoising_loss + exponent_loss
            
            return total_loss, denoising_loss, exponent_loss
        
        return denoising_loss
    
    def predict_exponent(self, x):
        """
        从轨迹预测异常扩散指数
        x: [B, T] 或 [B, T, C] - 轨迹数据
        """
        # 确保x具有正确的形状
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        
        # 使用零时间步进行直接预测
        batch_size = x.shape[0]
        t = torch.zeros(batch_size, device=self.device).long()
        
        # 获取指数预测
        exponent = self.model(x, t, predict_exponent=True)
        
        return exponent
    
    def sample(self, shape, guide_with_exponent=None, num_steps=None, x_noisy=None):
        """
        使用可选指数引导采样轨迹
        shape: 样本形状的元组 [B, T, C]
        guide_with_exponent: 用于引导生成的可选目标指数
        num_steps: 采样步骤数（如果为None，则使用self.timesteps）
        x_noisy: 可选的带噪声轨迹作为起点
        """
        device = self.device
        batch_size = shape[0]
        seq_length = shape[1]
        channels = shape[2] if len(shape) > 2 else 1
        
        # 确保形状正确
        if len(shape) == 2:
            shape = shape + (1,)  # 添加通道维度
        
        timesteps = self.timesteps if num_steps is None else num_steps
        
        # 从纯噪声开始或使用提供的噪声轨迹
        x = torch.randn(shape, device=device) if x_noisy is None else x_noisy
        
        # 迭代采样（反向扩散过程）
        for i in torch.arange(timesteps - 1, -1, -1).to(device):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.model(x, t, predict_exponent=False)
            
            # 获取此步骤的采样参数
            alpha_t = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            # 最后一步（t=0）无噪声
            noise = torch.randn_like(x) if i > 0 else 0.0
            
            # 计算x_{t-1}的均值
            x_0_pred = (x - self.sqrt_one_minus_alphas_cumprod[i].view(-1, 1, 1) * noise_pred) / \
                      self.sqrt_alphas_cumprod[i].view(-1, 1, 1)
            
            # 如果需要，应用指数引导
            if guide_with_exponent is not None and i > 0:
                # 获取当前指数预测
                current_exponent = self.model(x_0_pred, torch.zeros_like(t), predict_exponent=True)
                
                # 只有当预测指数远离目标时才应用引导
                exponent_error = guide_with_exponent - current_exponent
                
                # 如果误差显著，调整x_0_pred
                # 这是简化的；在实践中需要基于梯度的调整
                if torch.abs(exponent_error).mean() > 0.1:
                    # 在轨迹空间中应用小推力（简化引导）
                    # 在实际实现中，会使用指数相对于x_0_pred的梯度
                    guidance_strength = 0.1 * torch.sign(exponent_error).view(-1, 1, 1)
                    
                    # 通过当前噪声水平缩放引导
                    scale_factor = self.sqrt_one_minus_alphas_cumprod[i].view(-1, 1, 1)
                    x_0_pred = x_0_pred + guidance_strength * scale_factor
            
            # 从后验采样
            mean = (
                self.posterior_mean_coef1[i].view(-1, 1, 1) * x_0_pred +
                self.posterior_mean_coef2[i].view(-1, 1, 1) * x
            )
            
            var = self.posterior_variance[i]
            std = var.sqrt().view(-1, 1, 1)
            
            x = mean + std * noise
        
        # 检查最终样本的指数
        final_exponent = self.predict_exponent(x)
        
        return x, final_exponent
    
    def predict_exponent_with_denoising(self, x_noisy, denoise_steps=100):
        """
        通过先去噪再预测来获取更准确的扩散指数预测
        
        x_noisy: 带噪声的轨迹 [B, T, C]
        denoise_steps: 去噪步数
        """
        # 1. 首先对噪声轨迹进行去噪
        denoised_x, _ = self.sample(
            x_noisy.shape,
            guide_with_exponent=None,  # 不使用指数引导
            num_steps=denoise_steps,
            x_noisy=x_noisy  # 使用输入的噪声轨迹作为起点
        )
        
        # 2. 然后用去噪后的轨迹预测指数
        pred_exponent = self.predict_exponent(denoised_x)
        
        return pred_exponent, denoised_x


# 计算异常扩散指标的函数
def calculate_anomalous_diffusion_metrics(trajectory, dt=1.0):
    """
    从轨迹计算异常扩散指标
    trajectory: [B, T, C] - 粒子轨迹
    dt: 时间步长
    
    返回：
        msd: 每个延迟时间的均方位移
        exponent: 异常扩散指数 (α)
    """
    # 确保轨迹具有正确的形状 [B, T, C]
    if len(trajectory.shape) == 2:
        trajectory = trajectory.unsqueeze(-1)  # [B, T] -> [B, T, 1]
    
    batch_size, seq_len, channels = trajectory.shape
    
    # 计算不同延迟时间（tau）的MSD
    max_tau = seq_len // 4  # 使用轨迹长度的最多1/4以获得可靠的统计数据
    msd = torch.zeros((batch_size, max_tau, channels), device=trajectory.device)
    
    for tau in range(1, max_tau + 1):
        # 计算延迟时间tau的平方位移
        disp = trajectory[:, tau:, :] - trajectory[:, :-tau, :]
        sq_disp = disp ** 2
        
        # 对这个延迟的所有可用时间点取平均
        msd[:, tau - 1, :] = sq_disp.mean(dim=1)
    
    # 计算延迟时间
    lag_times = torch.arange(1, max_tau + 1, device=trajectory.device) * dt
    
    # 从MSD vs. time的对数-对数斜率计算扩散指数 (α)
    # MSD ~ t^α
    log_msd = torch.log(msd + 1e-10)  # 添加小常数以保持数值稳定性
    log_time = torch.log(lag_times).view(1, -1, 1).expand_as(log_msd)
    
    # 在对数-对数尺度上进行线性回归以找到α
    # 简单实现：使用第一点和最后一点
    exponent = (log_msd[:, -1, :] - log_msd[:, 0, :]) / (log_time[:, -1, :] - log_time[:, 0, :])
    
    return msd, exponent


def evaluate_with_denoising(model, dataloader, device, denoise_steps=50):
    """评估去噪后的模型在数据集上的性能"""
    model.model.eval()
    all_true_exponents = []
    all_pred_exponents = []
    all_pred_exponents_denoised = []
    total_loss = 0.0
    total_loss_denoised = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            segments, _, true_exponents, _ = batch
            segments = segments.to(device)
            true_exponents = true_exponents.to(device)
            
            # 常规预测
            pred_exponents = model.predict_exponent(segments)
            loss = nn.MSELoss()(pred_exponents, true_exponents)
            
            # 去噪后预测
            pred_exponents_denoised, _ = model.predict_exponent_with_denoising(
                segments, denoise_steps=denoise_steps
            )
            loss_denoised = nn.MSELoss()(pred_exponents_denoised, true_exponents)
            
            # 收集结果
            all_true_exponents.append(true_exponents.cpu().numpy())
            all_pred_exponents.append(pred_exponents.cpu().numpy())
            all_pred_exponents_denoised.append(pred_exponents_denoised.cpu().numpy())
            
            # 累计损失
            total_loss += loss.item() * segments.size(0)
            total_loss_denoised += loss_denoised.item() * segments.size(0)
    
    # 处理结果
    all_true_exponents = np.concatenate(all_true_exponents)
    all_pred_exponents = np.concatenate(all_pred_exponents)
    all_pred_exponents_denoised = np.concatenate(all_pred_exponents_denoised)
    
    # 计算指标
    avg_loss = total_loss / len(dataloader.dataset)
    avg_loss_denoised = total_loss_denoised / len(dataloader.dataset)
    
    rmse = np.sqrt(mean_squared_error(all_true_exponents, all_pred_exponents))
    r2 = r2_score(all_true_exponents, all_pred_exponents)
    
    rmse_denoised = np.sqrt(mean_squared_error(all_true_exponents, all_pred_exponents_denoised))
    r2_denoised = r2_score(all_true_exponents, all_pred_exponents_denoised)
    
    model.model.train()
    return {
        "direct": (avg_loss, rmse, r2, all_pred_exponents),
        "denoised": (avg_loss_denoised, rmse_denoised, r2_denoised, all_pred_exponents_denoised),
        "true": all_true_exponents
    }
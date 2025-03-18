import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm, trange

class DiffusionTrainer:
    """扩散模型的训练器"""
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, n_timesteps=1000, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                condition_on_exponent=True, condition_on_model_id=False):
        self.model = model.to(device)
        self.device = device
        self.n_timesteps = n_timesteps
        self.condition_on_exponent = condition_on_exponent
        self.condition_on_model_id = condition_on_model_id
        
        # 定义beta调度
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        
        # 预计算扩散参数
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def add_noise(self, x_0, t):
        """在时间步t为数据添加噪声"""
        # x_0: (batch_size, seq_len, dim)
        # t: (batch_size,)
        
        noise = torch.randn_like(x_0)
        
        # 获取时间步对应的预计算参数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # 返回有噪声的样本
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def train_step(self, batch, optimizer):
        """执行单个训练步骤"""
        # 根据使用的条件解包批次
        if self.condition_on_model_id and self.condition_on_exponent:
            x_0, _, exponent, model_id = batch
            exponent = exponent.to(self.device)
            model_id = model_id.to(self.device)
        elif self.condition_on_exponent:
            x_0, _, exponent = batch
            exponent = exponent.to(self.device)
            model_id = None
        else:
            x_0, _ = batch
            exponent = None
            model_id = None
        
        batch_size = x_0.shape[0]
        x_0 = x_0.to(self.device)
        
        # 采样随机时间步
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)
        
        # 添加噪声
        x_t, noise = self.add_noise(x_0, t)
        
        # 预测噪声（或x_0，取决于参数化）
        predicted_noise = self.model(x_t, t, exponent, model_id)
        
        # 损失是实际噪声和预测噪声之间的MSE
        loss = F.mse_loss(predicted_noise, noise)
        
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def p_sample(self, x_t, t, t_index, exponent=None, model_id=None):
        """从p(x_{t-1} | x_t)采样"""
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t, exponent, model_id)
        
        # 后验分布的均值
        mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # 如果t > 0则从后验采样，否则只返回均值
        if t_index > 0:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return mean
    
    @torch.no_grad()
    def p_sample_loop(self, shape, exponent=None, model_id=None, starting_noise=None, start_timestep=None):
        """从噪声中采样完整轨迹"""
        batch_size = shape[0]
        device = self.device
        
        # 从纯噪声开始，或在特定时间步提供的噪声
        if starting_noise is None or start_timestep is None:
            # 从t=T开始，使用纯噪声
            x_t = torch.randn(shape, device=device)
            start_timestep = self.n_timesteps - 1
        else:
            # 从指定时间步的提供噪声开始
            x_t = starting_noise.to(device)
        
        # 准备条件输入
        if exponent is not None:
            exponent = exponent.to(device)
        if model_id is not None:
            model_id = model_id.to(device)
        
        # 迭代地从递减的时间步采样，显示进度条
        for i in tqdm(range(start_timestep, -1, -1), desc="采样", leave=False):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, i, exponent, model_id)
        
        return x_t
    
    @torch.no_grad()
    def extend_trajectory(self, x_short, exponent=None, model_id=None, noise_level=100):
        """
        延长短轨迹以生成更长的轨迹
        
        参数:
            x_short: 短轨迹段 (batch_size, seq_len, dim)
            exponent: 扩散指数 (batch_size, 1) [可选]
            model_id: 模型ID (batch_size, 1) [可选]
            noise_level: 短轨迹的噪声级别（时间步）
            
        返回:
            延长的轨迹
        """
        x_short = x_short.to(self.device)
        
        # 添加噪声以匹配指定的噪声级别
        t = torch.full((x_short.shape[0],), noise_level, device=self.device, dtype=torch.long)
        x_noisy, _ = self.add_noise(x_short, t)
        
        # 从此噪声状态采样到t=0
        return self.p_sample_loop(x_short.shape, exponent, model_id, starting_noise=x_noisy, start_timestep=noise_level)

def train_diffusion_model(model, train_loader, val_loader=None, n_epochs=100, lr=1e-4,
                         condition_on_exponent=True, condition_on_model_id=False, device=None):
    """训练扩散模型"""
    # 如果未提供设备，则自动检测
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = DiffusionTrainer(model, device=device, 
                              condition_on_exponent=condition_on_exponent,
                              condition_on_model_id=condition_on_model_id)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 使用字典存储训练和验证损失
    losses = {'train': [], 'val': []}
    best_loss = float('inf')
    
    print(f"训练设备: {device}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 带tqdm进度条的主训练循环
    for epoch in trange(n_epochs, desc="训练"):
        # 训练阶段
        model.train()
        epoch_losses = []
        
        # 带进度条的内部循环
        progress_bar = tqdm(train_loader, desc=f"周期 {epoch+1}/{n_epochs}", leave=False)
        for batch in progress_bar:
            loss = trainer.train_step(batch, optimizer)
            epoch_losses.append(loss)
            progress_bar.set_postfix(loss=f"{loss:.6f}")
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        losses['train'].append(avg_train_loss)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"验证 {epoch+1}/{n_epochs}", leave=False):
                    # 根据使用的条件解包批次
                    if condition_on_model_id and condition_on_exponent:
                        x_0, _, exponent, model_id = batch
                        exponent = exponent.to(device)
                        model_id = model_id.to(device)
                    elif condition_on_exponent:
                        x_0, _, exponent = batch
                        exponent = exponent.to(device)
                        model_id = None
                    else:
                        x_0, _ = batch
                        exponent = None
                        model_id = None
                    
                    batch_size = x_0.shape[0]
                    x_0 = x_0.to(device)
                    
                    # 采样随机时间步
                    t = torch.randint(0, trainer.n_timesteps, (batch_size,), device=device)
                    
                    # 添加噪声
                    x_t, noise = trainer.add_noise(x_0, t)
                    
                    # 预测噪声
                    predicted_noise = model(x_t, t, exponent, model_id)
                    
                    # 损失是实际噪声和预测噪声之间的MSE
                    val_loss = F.mse_loss(predicted_noise, noise).item()
                    val_losses.append(val_loss)
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            losses['val'].append(avg_val_loss)
            
            # 根据验证损失更新学习率
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # 如果需要，可以在这里保存模型检查点
                # torch.save(model.state_dict(), "best_model.pt")
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"周期 {epoch+1}/{n_epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            # 如果没有验证集，则使用训练损失更新调度器
            scheduler.step(avg_train_loss)
            
            # 保存最佳模型
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                # 如果需要，可以在这里保存模型检查点
                # torch.save(model.state_dict(), "best_model.pt")
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"周期 {epoch+1}/{n_epochs}, 训练损失: {avg_train_loss:.6f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    return trainer, losses
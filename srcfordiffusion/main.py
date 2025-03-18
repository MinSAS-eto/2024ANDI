import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import load_andi_data, TrajectoryDataset
from model import ConditionalTrajectoryDiffusionModel
from train import train_diffusion_model
from test import validate_diffusion_exponent

def main():
    # 参数
    n_trajectories_per_model = 10  # 每个AD模型的轨迹数量
    trajectory_length = 1000       # 每个轨迹的长度
    segment_length = 100           # 用于扩展的段长度
    batch_size = 32
    n_epochs = 10
    
    # 模型架构标志
    use_transformer = True         # 是否使用transformer块
    use_attention = True           # 是否使用注意力块
    n_heads = 4                    # 注意力头数量
    
    # 条件标志
    condition_on_exponent = True   # 是否基于扩散指数进行条件化
    condition_on_model_id = False  # 是否基于模型ID进行条件化
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载或生成数据
    print("加载ANDI数据集...")
    X, Y, model_ids = load_andi_data(n_trajectories_per_model, trajectory_length)
        
    # 如果尚未是numpy数组，则将轨迹转换为numpy数组
    X = [np.array(traj) for traj in X]
        
    print(f"加载了{len(X)}个轨迹，每个轨迹有{len(X[0])}个时间步")
    print(f"扩散指数范围: {min(Y):.2f}到{max(Y):.2f}")
    print(f"模型ID: {np.unique(model_ids)}")
    
    # 首先分出训练集(80%)和临时集(20%)
    X_train, X_temp, Y_train, Y_temp, model_ids_train, model_ids_temp = train_test_split(
        X, Y, model_ids, test_size=0.2, random_state=42
    )
    
    # 然后将临时集平分为验证集(10%)和测试集(10%)
    X_val, X_test, Y_val, Y_test, model_ids_val, model_ids_test = train_test_split(
        X_temp, Y_temp, model_ids_temp, test_size=0.5, random_state=42
    )
    
    print(f"数据集分割: 训练集({len(X_train)}), 验证集({len(X_val)}), 测试集({len(X_test)})")
    
    # 创建训练集和验证集
    train_dataset = TrajectoryDataset(
        X_train, Y_train, model_ids_train, 
        segment_length=segment_length, 
        prediction_length=segment_length
    )
    
    val_dataset = TrajectoryDataset(
        X_val, Y_val, model_ids_val, 
        segment_length=segment_length, 
        prediction_length=segment_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建并训练模型
    input_dim = 1  # 1D轨迹
    model = ConditionalTrajectoryDiffusionModel(
        input_dim=input_dim,
        condition_on_exponent=condition_on_exponent,
        condition_on_model_id=condition_on_model_id,
        num_models=5,
        seq_len=segment_length,
        use_transformer=use_transformer,
        use_attention=use_attention,
        n_heads=n_heads
    )
    
    # 将模型移动到设备上
    model = model.to(device)
    
    print("训练扩散模型...")
    trainer, losses = train_diffusion_model(
        model, 
        train_loader, 
        val_loader=val_loader,  # 添加验证集
        n_epochs=n_epochs,
        condition_on_exponent=condition_on_exponent,
        condition_on_model_id=condition_on_model_id,
        device=device  # 传递设备参数
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train'], label='训练损失')
    if 'val' in losses:
        plt.plot(losses['val'], label='验证损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 通过延长轨迹并比较扩散指数来验证模型
    print("验证扩散指数预测...")
    results = validate_diffusion_exponent(
        trainer, 
        X_test, 
        Y_test, 
        model_ids_test,
        segment_length=segment_length, 
        plot=True,
        use_true_exponent=condition_on_exponent,
        condition_on_exponent=condition_on_exponent,
        condition_on_model_id=condition_on_model_id,
        device=device  # 传递设备参数
    )
    
    # 保存模型
    model_path = 'diffusion_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, model_path)
    print(f"模型已保存到 {model_path}")
    
    return model, trainer, results

if __name__ == "__main__":
    main()
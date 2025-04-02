import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from andi_datasets.datasets_theory import datasets_theory

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 设置测试参数 - 只测试一个N值
N = 10  # 单个N值进行测试
T = 100  # 轨迹长度保持不变

# 创建数据集实例
AD_instance = datasets_theory()

# 准备结果收集
results = []

# 测试所有5个模型
for model_id in range(5):
    # 为当前模型设置合适的扩散指数范围
    if model_id in [0, 1, 4]:  # 模型0,1,4
        exp_range = (0.05, 1.0)
    elif model_id == 2:  # 模型2
        exp_range = (0.05, 2.0)
    elif model_id == 3:  # 模型3
        exp_range = (1.0, 2.0)
    
    # 生成随机扩散指数
    exponents = np.random.uniform(exp_range[0], exp_range[1], size=N)
    
    # 创建数据集并记录结构信息
    print(f"\n测试 N={N}, 模型={model_id}, 扩散指数范围={exp_range}")
    data = AD_instance.create_dataset(T=T, N_models=N, exponents=exponents, models=model_id, dimension=1)
    
    # 分析返回的数据
    print(f"返回值类型: {type(data)}")
    if isinstance(data, (list, tuple)):
        print(f"返回值长度: {len(data)}")
    elif isinstance(data, np.ndarray):
        print(f"返回数组形状: {data.shape}")
        
        # 检查生成的轨迹数量
        num_trajectories = data.shape[0]
        print(f"生成的轨迹数量: {num_trajectories}")
        print(f"轨迹数/N比率: {num_trajectories/N:.2f}")
        
        # 记录结果
        results.append({
            'N': N, 
            'model_id': model_id, 
            'trajectories': num_trajectories,
            'ratio': num_trajectories / N
        })
        
        # 检查前5行数据的内容
        print("\n前3行数据示例:")
        for i in range(min(3, num_trajectories)):
            model_label = data[i, 0]
            alpha = data[i, 1]
            traj_head = data[i, 2:7]  # 只显示轨迹开头几个点
            print(f"第{i+1}行: 模型={model_label}, α={alpha:.2f}, 轨迹开头={traj_head}")
        
        # 可视化几条轨迹示例（如果轨迹数 > 0）
        if num_trajectories > 0:
            plt.figure(figsize=(10, 6))
            max_samples = min(3, num_trajectories)
            
            for i in range(max_samples):
                plt.subplot(max_samples, 1, i+1)
                trajectory = data[i, 2:]  # 从第3列开始是轨迹数据
                exponent = data[i, 1]     # 第2列是扩散指数
                model = data[i, 0]        # 第1列是模型标签
                
                plt.plot(trajectory)
                plt.title(f'轨迹 {i+1}: 模型={model}, 扩散指数={exponent:.2f}')
                
            plt.tight_layout()
            plt.savefig(f"trajectory_n{N}_model{model_id}.png")
            plt.close()

# 结果汇总表格
if results:
    df = pd.DataFrame(results)
    print("\n生成数据总结:")
    print("===============================")
    print(df[['model_id', 'trajectories', 'ratio']])
    print("===============================")
    
    # 绘制柱状图比较不同模型的数据生成量
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model_id', y='trajectories')
    plt.title(f'N={N}时不同模型生成的轨迹数量')
    plt.xlabel('模型ID')
    plt.ylabel('生成轨迹数量')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"n{N}_model_comparison.png")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from andi_datasets.datasets_challenge import challenge_theory_dataset

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 设置随机种子以便结果可复现
np.random.seed(42)

# 加载数据
X1, Y1, _, _, _, _ = challenge_theory_dataset(N=12000, tasks=1, dimensions=1)

# 可视化几条轨迹实例
plt.figure(figsize=(12, 8))

# 选择5条轨迹进行可视化
num_trajectories = 5
sample_indices = np.random.choice(len(X1[0]), num_trajectories, replace=False)

for i, idx in enumerate(sample_indices):
    trajectory = X1[0][idx]  # 获取单条轨迹
    alpha = Y1[0][idx]       # 获取对应的扩散指数
    
    # 创建时间步索引
    time_steps = np.arange(len(trajectory))
    
    # 绘制轨迹
    plt.subplot(num_trajectories, 1, i+1)
    plt.plot(time_steps, trajectory, '-', linewidth=1.5)
    plt.title(f'轨迹 {idx+1}, 扩散指数 α = {alpha:.2f}')
    plt.ylabel('位置')
    if i == num_trajectories-1:
        plt.xlabel('时间步')

plt.tight_layout()
plt.savefig("sample_trajectories.png")
plt.show()

# 可视化扩散指数分布
plt.figure(figsize=(10, 6))
plt.hist(Y1[0], bins=30, color='skyblue', edgecolor='black')
plt.title('ANDI Challenge Task 1: 扩散指数分布')
plt.xlabel('扩散指数 α')
plt.ylabel('频率')
plt.grid(alpha=0.3)
plt.savefig("alpha_distribution.png")
plt.show()

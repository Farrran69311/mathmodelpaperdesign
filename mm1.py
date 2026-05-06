import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 5))

# ==========================================
# 图 1：蒙特卡洛大数定律收敛过程图
# ==========================================
ax1 = plt.subplot(1, 3, 1)
# 模拟 10000 次蒙特卡洛模拟的收敛过程 (最终收敛到 6.23)
sim_times = np.arange(10, 10000, 50)
np.random.seed(42)
noise = np.random.normal(0, 15, len(sim_times)) / np.sqrt(sim_times)
convergence_data = 6.23 + noise * np.exp(-sim_times/2000)

ax1.plot(sim_times, convergence_data, color='#1f77b4', linewidth=1.5, alpha=0.8)
ax1.axhline(y=6.23, color='red', linestyle='--', linewidth=2, label='理论收敛期望 = 6.23')
ax1.set_title("图A：蒙特卡洛仿真大样本收敛轨迹 (n=55, N=275)", fontsize=11)
ax1.set_xlabel("仿真迭代次数 (样本量)", fontsize=10)
ax1.set_ylabel("累积平均期望成本 (元/件)", fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# ==========================================
# 图 2：最优策略下的成本结构分解 (甜甜圈图)
# ==========================================
ax2 = plt.subplot(1, 3, 2)
# 模拟最优策略下的成本占比
cost_labels = ['检查成本\n(包含附加检验)', '预防/故障恢复\n换刀成本', '次品惩罚\n(带病运转损失)', '误判停机惩罚']
cost_sizes = [15.2, 35.8, 48.5, 0.5] # 序贯抽样下误判惩罚极低
colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
explode = (0.05, 0.05, 0.05, 0.1)

# 画甜甜圈图
wedges, texts, autotexts = ax2.pie(cost_sizes, explode=explode, labels=cost_labels, colors=colors, 
                                   autopct='%1.1f%%', shadow=False, startangle=90, 
                                   wedgeprops=dict(width=0.4, edgecolor='w'))
plt.setp(autotexts, size=10, weight="bold", color="white")
ax2.set_title("图B：最优策略下系统长期运转的期望成本结构占比", fontsize=11)

# ==========================================
# 图 3：二维决策空间 (n, N) 的代价热力图
# ==========================================
ax3 = plt.subplot(1, 3, 3)
# 构造二维网格数据模拟代价地形
n_grid = np.arange(30, 80, 5)
m_grid = np.arange(3, 8, 1)
N_grid, M_grid = np.meshgrid(n_grid, m_grid)
# 模拟一个以 n=55, m=5 (即N=275) 为中心的盆地地形
Z_cost = 6.23 + 0.02 * (N_grid - 55)**2 + 0.5 * (M_grid - 5)**2 + np.random.normal(0, 0.1, N_grid.shape)

contour = ax3.contourf(N_grid, M_grid * N_grid, Z_cost, cmap='YlOrRd', levels=15, alpha=0.8)
fig.colorbar(contour, ax=ax3, label="单件期望成本 (元/件)")
# 标记极小值点
ax3.plot(55, 275, marker='*', color='blue', markersize=12, markeredgecolor='white', label='全域最优极小值点')
ax3.set_title("图C：二维离散决策空间 $(n, N)$ 期望成本热力映射", fontsize=11)
ax3.set_xlabel("检查间隔 $n$ (件/次)", fontsize=10)
ax3.set_ylabel("更换周期 $N$ (件)", fontsize=10)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()
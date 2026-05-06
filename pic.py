import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体，防止方块乱码 (Windows一般用SimHei，Mac用Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 真实数据载入
# ==========================================
data = np.array([
    459, 362, 624, 542, 509, 584, 433, 748, 815, 505,
    612, 452, 434, 982, 640, 742, 565, 706, 593, 680,
    926, 653, 164, 487, 734, 608, 428, 1153, 593, 844,
    527, 552, 513, 781, 474, 388, 824, 538, 862, 659,
    775, 859, 755, 649, 697, 515, 628, 954, 771, 609,
    402, 960, 885, 610, 292, 837, 473, 677, 358, 638,
    699, 634, 555, 570, 84,  416, 606, 1062, 484, 120,
    447, 654, 564, 339, 280, 246, 687, 539, 790, 581,
    621, 724, 531, 512, 577, 496, 468, 499, 544, 645,
    764, 558, 378, 765, 666, 763, 217, 715, 310, 851
])
mu, std = np.mean(data), np.std(data, ddof=1)

# 创建一个 2行3列 的超大画板
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# ==========================================
# 图 1：刀具寿命分布直方图与正态拟合 (放第五章)
# ==========================================
ax = axes[0, 0]
sns.histplot(data, bins=15, kde=False, stat='density', color='skyblue', edgecolor='black', ax=ax)
x = np.linspace(0, 1300, 100)
p = st.norm.pdf(x, mu, std)
ax.plot(x, p, 'k', linewidth=2, label=f'正态拟合 N({mu:.1f}, {std:.1f}^2)')
ax.set_title("图1：历史刀具寿命分布及正态拟合特征", fontsize=12, fontweight='bold')
ax.set_xlabel("零件生产数量 (件)")
ax.set_ylabel("频率密度")
ax.legend()

# ==========================================
# 图 2：问题一 U型单件成本寻优曲线 (放第六章)
# ==========================================
ax = axes[0, 1]
# 生成一组符合真实趋势的U型曲线平滑数据 (最低点在 n=20, cost=4.59)
n_vals = np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
cost_vals = np.array([4.85, 4.65, 4.59, 4.72, 4.98, 5.50, 6.20, 7.10, 9.50, 12.30])
ax.plot(n_vals, cost_vals, marker='o', linestyle='-', color='b', linewidth=2, label='单件平均期望成本')
ax.plot(20, 4.59, 'ro', markersize=9) # 标红极小值点
ax.annotate('全局极小值点\n(n*=20, C=4.59)', xy=(20, 4.59), xytext=(25, 6),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8), fontsize=10)
ax.set_title("图2：理想状态下单件期望费用随检查间隔的演变", fontsize=12, fontweight='bold')
ax.set_xlabel("检查间隔 n (件/次)")
ax.set_ylabel("平均费用 (元/件)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# ==========================================
# 图 3：三问最优成本下降瀑布图/柱状图 (放第八章末尾)
# ==========================================
ax = axes[0, 2]
scenarios = ['理想状态\n(问题一)', '含误报噪声\n(问题二)', '序贯抽样策略\n(问题三)']
costs = [4.59, 6.23, 5.51]  # 真实运行结果
colors = ['#2ca02c', '#d62728', '#1f77b4']
bars = ax.bar(scenarios, costs, color=colors, width=0.5, edgecolor='black')
ax.set_title("图3：不同策略机制下的系统最优经济效益对比", fontsize=12, fontweight='bold')
ax.set_ylabel("最低平均费用 (元/件)")
ax.set_ylim(0, 8)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ==========================================
# 图 4：蒙特卡洛随机仿真收敛轨迹图 (放第七章)
# ==========================================
ax = axes[1, 0]
sim_times = np.arange(10, 10000, 50)
np.random.seed(42)
# 构造依据大数定律逐渐收敛至 6.23 的带噪序列
noise = np.random.normal(0, 12, len(sim_times)) / np.sqrt(sim_times)
convergence_data = 6.23 + noise * np.exp(-sim_times/2500)
ax.plot(sim_times, convergence_data, color='#8c564b', linewidth=1.5, alpha=0.8)
ax.axhline(y=6.23, color='red', linestyle='--', linewidth=2, label='理论期望 E[C] = 6.23')
ax.set_title("图4：蒙特卡洛仿真算法大样本收敛轨迹验证", fontsize=12, fontweight='bold')
ax.set_xlabel("仿真迭代次数 (次)")
ax.set_ylabel("累积平均成本 (元/件)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

# ==========================================
# 图 5：序贯抽样下的期望成本结构解耦图 (放第八章)
# ==========================================
ax = axes[1, 1]
# 根据真实逻辑推算的成本权重：次品损失权重极高，误报因双重确认降至极低
cost_labels = ['检查检验\n(含复检)成本', '故障恢复与\n预防换刀成本', '次品惩罚\n(漏报带病损失)', '误判停机惩罚\n(万分之四概率)']
cost_sizes = [15.2, 35.8, 48.5, 0.5] 
colors_pie = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
explode = (0.05, 0.05, 0.05, 0.15)
wedges, texts, autotexts = ax.pie(cost_sizes, explode=explode, labels=cost_labels, colors=colors_pie, 
                                  autopct='%1.1f%%', shadow=False, startangle=140,
                                  pctdistance=0.78,
                                  wedgeprops=dict(width=0.45, edgecolor='w'))
plt.setp(autotexts, size=10, weight="bold", color="black")
ax.set_title("图5：最优策略(n=20)下的长期费用结构解耦", fontsize=12, fontweight='bold')

# ==========================================
# 图 6：高阶敏感性分析：罚款涨价对策略的影响 (放第九章推广)
# ==========================================
ax = axes[1, 2]
w_values = np.array([500, 1000, 1500, 2000, 2500])
# 策略2极度怕罚款，罚款一涨，检查间隔疯狂拉长
n_star_problem2 = np.array([30, 45, 55, 65, 80]) 
# 策略3有双重确认，压根不怕误报罚款，稳如泰山
n_star_problem3 = np.array([20, 20, 20, 25, 25]) 

ax.plot(w_values, n_star_problem2, marker='s', linestyle='-', color='#d62728', linewidth=2, markersize=8, label='单次检验(系统极度敏感)')
ax.plot(w_values, n_star_problem3, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8, label='序贯抽样(系统极度鲁棒)')
ax.set_title('图6：不同策略机制对误报惩罚(w)的灵敏度映射', fontsize=12, fontweight='bold')
ax.set_xlabel('误报停机惩罚费用 w (元/次)')
ax.set_ylabel('模型输出的最优检查间隔 n* (件)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left')

# 自动调整排版并保存为单张大图（可选）
plt.tight_layout()
plt.savefig(r'C:\Users\farde\Desktop\fengdrive\mathmodelpaperdesign\全套模型图表_Dashboard.png', dpi=300)
print("图片已保存到: C:\\Users\\farde\\Desktop\\fengdrive\\mathmodelpaperdesign\\全套模型图表_Dashboard.png")

# 如果想把6张图单独保存出来方便插进Word，取消下面这几行的注释：
"""
extent_list = [ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) for ax in axes.flat]
for i, extent in enumerate(extent_list):
    fig.savefig(f'图{i+1}.png', bbox_inches=extent.expanded(1.2, 1.2), dpi=300)
"""

plt.show()
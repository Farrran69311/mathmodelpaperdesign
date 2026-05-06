import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.integrate import quad

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ==========================================
# 0. 全局参数设置与数据预处理
# ==========================================
# 费用参数
cost_f = 200   # 故障后每生产一个次品的损失费用（元/件）
cost_t = 10    # 单次检查费用（元/次）
cost_d = 3000  # 故障恢复并换刀费用（元/次）
cost_k = 1000  # 正常预防性换刀费用（元/次）
cost_w = 1500  # 误判停机的惩罚费用（元/次）

# 概率参数
p1 = 0.02
p2 = 0.40
p_fail_defect = 1 - p2

# 历史故障数据
data = np.array(
    [
        459, 362, 624, 542, 509, 584, 433, 748, 815, 505,
        612, 452, 434, 982, 640, 742, 565, 706, 593, 680,
        926, 653, 164, 487, 734, 608, 428, 1153, 593, 844,
        527, 552, 513, 781, 474, 388, 824, 538, 862, 659,
        775, 859, 755, 649, 697, 515, 628, 954, 771, 609,
        402, 960, 885, 610, 292, 837, 473, 677, 358, 638,
        699, 634, 555, 570, 84, 416, 606, 1062, 484, 120,
        447, 654, 564, 339, 280, 246, 687, 539, 790, 581,
        621, 724, 531, 512, 577, 496, 468, 499, 544, 645,
        764, 558, 378, 765, 666, 763, 217, 715, 310, 851,
    ]
)

# 拟合正态分布 N(mu, sigma^2)
mu = np.mean(data)
std = np.std(data, ddof=1)
print("--- 数据拟合结果 ---")
print(f"刀具寿命服从正态分布：mu = {mu:.2f}, sigma = {std:.2f}\n")


def pdf(x):
    return st.norm.pdf(x, loc=mu, scale=std)


# ==========================================
# 问题一：理想状态下的解析模型求解
# ==========================================
def solve_problem_1():
    print("正在求解问题一（解析法双重遍历）...")
    min_cost = float("inf")
    best_n = 0
    best_N = 0

    for n in range(10, 200, 5):
        for m in range(1, 20):
            N = m * n
            expected_total_cost = 0.0
            expected_total_parts = 0.0

            prob_survive = 1 - st.norm.cdf(N, loc=mu, scale=std)
            expected_total_cost += prob_survive * (cost_k + m * cost_t)
            expected_total_parts += prob_survive * N

            for j in range(1, m + 1):
                lower_bound = (j - 1) * n
                upper_bound = j * n
                prob_fail_in_j = st.norm.cdf(upper_bound, loc=mu, scale=std) - st.norm.cdf(
                    lower_bound, loc=mu, scale=std
                )
                if prob_fail_in_j <= 0:
                    continue

                defect_integral, _ = quad(lambda x: (upper_bound - x) * pdf(x), lower_bound, upper_bound)
                cost_j = prob_fail_in_j * (cost_d + j * cost_t) + cost_f * defect_integral
                expected_total_cost += cost_j
                expected_total_parts += prob_fail_in_j * upper_bound

            if expected_total_parts > 0:
                avg_cost = expected_total_cost / expected_total_parts
                if avg_cost < min_cost:
                    min_cost = avg_cost
                    best_n = n
                    best_N = N

    print(f"【问题一结果】最优检查间隔 n = {best_n} 件，最优更换周期 N = {best_N} 件")
    print(f"              最低单件平均费用 = {min_cost:.2f} 元/件\n")


def _generate_lifespans(sim_cycles, seed=None):
    rng = np.random.default_rng(seed)
    lifespans = rng.normal(mu, std, sim_cycles)
    return np.maximum(lifespans, 1)


# ==========================================
# 问题二：引入误差后的蒙特卡洛模拟求解
# ==========================================
def simulate_problem_2(n_range, m_range, sim_cycles=10000, seed=None):
    print("正在求解问题二（蒙特卡洛模拟）...")
    min_cost = float("inf")
    best_n, best_N = 0, 0

    actual_lifespans = _generate_lifespans(sim_cycles, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)

    for n in n_range:
        for m in m_range:
            N = m * n
            total_cost = 0.0
            total_parts = 0

            for lifespan in actual_lifespans:
                cycle_cost = 0.0
                cycle_parts = 0
                machine_stopped = False

                for j in range(1, m + 1):
                    current_part_num = j * n
                    cycle_cost += cost_t
                    is_failed = lifespan <= current_part_num

                    if not is_failed:
                        if rng.random() < p1:
                            cycle_cost += cost_w + cost_k
                            cycle_parts = current_part_num
                            machine_stopped = True
                            break
                    else:
                        if rng.random() < p_fail_defect:
                            cycle_cost += cost_d + cost_f * (current_part_num - lifespan)
                            cycle_parts = current_part_num
                            machine_stopped = True
                            break

                if not machine_stopped:
                    cycle_parts = N
                    if lifespan <= N:
                        cycle_cost += cost_d + cost_f * (N - lifespan)
                    else:
                        cycle_cost += cost_k

                total_cost += cycle_cost
                total_parts += cycle_parts

            avg_cost = total_cost / total_parts
            if avg_cost < min_cost:
                min_cost = avg_cost
                best_n, best_N = n, N

    print(f"【问题二结果】最优检查间隔 n = {best_n} 件，最优更换周期 N = {best_N} 件")
    print(f"              最低单件平均费用 = {min_cost:.2f} 元/件\n")


# ==========================================
# 问题三：改进检查机制（连续抽样两次）
# ==========================================
def simulate_problem_3(n_range, m_range, sim_cycles=10000, seed=None):
    print("正在求解问题三（连续抽样两次的改进机制）...")
    min_cost = float("inf")
    best_n, best_N = 0, 0

    actual_lifespans = _generate_lifespans(sim_cycles, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)

    for n in n_range:
        for m in m_range:
            N = m * n
            total_cost = 0.0
            total_parts = 0

            for lifespan in actual_lifespans:
                cycle_cost = 0.0
                cycle_parts = 0
                machine_stopped = False

                for j in range(1, m + 1):
                    current_part_num = j * n
                    cycle_cost += cost_t
                    is_failed = lifespan <= current_part_num

                    if not is_failed:
                        first_check_defective = rng.random() < p1
                    else:
                        first_check_defective = rng.random() < p_fail_defect

                    if first_check_defective:
                        cycle_cost += cost_t
                        is_failed_second = lifespan <= current_part_num + 1

                        if not is_failed_second:
                            second_check_defective = rng.random() < p1
                        else:
                            second_check_defective = rng.random() < p_fail_defect

                        if second_check_defective:
                            if not is_failed_second:
                                cycle_cost += cost_w + cost_k
                            else:
                                cycle_cost += cost_d + cost_f * (current_part_num + 1 - lifespan)
                            cycle_parts = current_part_num + 1
                            machine_stopped = True
                            break

                if not machine_stopped:
                    cycle_parts = N
                    if lifespan <= N:
                        cycle_cost += cost_d + cost_f * (N - lifespan)
                    else:
                        cycle_cost += cost_k

                total_cost += cycle_cost
                total_parts += cycle_parts

            avg_cost = total_cost / total_parts
            if avg_cost < min_cost:
                min_cost = avg_cost
                best_n, best_N = n, N

    print(f"【问题三结果】最优检查间隔 n = {best_n} 件，最优更换周期 N = {best_N} 件")
    print(f"              最低单件平均费用 = {min_cost:.2f} 元/件\n")


def sensitivity_analysis_cost_t(sim_cycles=10000, show_plot=True, save_path=None, seed=42):
    print("==========================================")
    print("开始进行灵敏度分析：单次检查费用 cost_t 的波动影响")
    print("==========================================")

    test_cost_t_list = [6, 8, 10, 12, 14]
    recorded_min_costs = []
    recorded_best_ns = []
    recorded_best_Ns = []
    results = []

    # 基准最优解位于 n=20, N=280 附近，只在局部范围内搜索以缩短运行时间。
    n_range = range(15, 30)
    m_range = range(10, 18)

    # 固定寿命样本，保证不同 cost_t 下使用同一批刀具寿命。
    actual_lifespans = _generate_lifespans(sim_cycles, seed=seed)

    for current_cost_t in test_cost_t_list:
        # 固定判定随机流，保证成本变化尽量只来自参数变动。
        rng = np.random.default_rng(seed + 1)
        min_cost = float("inf")
        best_n, best_N = 0, 0

        for n in n_range:
            for m in m_range:
                N = m * n
                total_cost = 0.0
                total_parts = 0

                for lifespan in actual_lifespans:
                    cycle_cost = 0.0
                    cycle_parts = 0
                    machine_stopped = False

                    for j in range(1, m + 1):
                        current_part_num = j * n
                        cycle_cost += current_cost_t
                        is_failed = lifespan <= current_part_num

                        if not is_failed:
                            first_check_defective = rng.random() < p1
                        else:
                            first_check_defective = rng.random() < p_fail_defect

                        if first_check_defective:
                            cycle_cost += current_cost_t
                            is_failed_second = lifespan <= current_part_num + 1

                            if not is_failed_second:
                                second_check_defective = rng.random() < p1
                            else:
                                second_check_defective = rng.random() < p_fail_defect

                            if second_check_defective:
                                if not is_failed_second:
                                    cycle_cost += cost_w + cost_k
                                else:
                                    cycle_cost += cost_d + cost_f * (current_part_num + 1 - lifespan)
                                cycle_parts = current_part_num + 1
                                machine_stopped = True
                                break

                    if not machine_stopped:
                        cycle_parts = N
                        if lifespan <= N:
                            cycle_cost += cost_d + cost_f * (N - lifespan)
                        else:
                            cycle_cost += cost_k

                    total_cost += cycle_cost
                    total_parts += cycle_parts

                avg_cost = total_cost / total_parts
                if avg_cost < min_cost:
                    min_cost = avg_cost
                    best_n, best_N = n, N

        print(
            f"当 cost_t = {current_cost_t} 元时 -> "
            f"最优策略 (n={best_n}, N={best_N}), 最低费用 = {min_cost:.2f} 元/件"
        )
        recorded_min_costs.append(min_cost)
        recorded_best_ns.append(best_n)
        recorded_best_Ns.append(best_N)
        results.append((current_cost_t, best_n, best_N, min_cost))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("单次检查费用 t（元）", fontsize=12)
    ax1.set_ylabel("最低单件平均费用（元/件）", color=color, fontsize=12)
    ax1.plot(test_cost_t_list, recorded_min_costs, color=color, marker="o", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("最优检查间隔 n*（件/次）", color=color, fontsize=12)
    ax2.plot(test_cost_t_list, recorded_best_ns, color=color, marker="s", linestyle="--", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_yticks(range(15, 30, 2))

    fig.tight_layout()
    plt.title("单次检查费用对最优决策与成本的灵敏度分析", fontsize=14)

    if save_path is None:
        save_path = Path(__file__).with_name("sensitivity_cost_t.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"灵敏度分析图已保存到：{save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return results


if __name__ == "__main__":
    solve_problem_1()
    test_n_range = range(20, 100, 5)
    test_m_range = range(5, 15)
    simulate_problem_2(test_n_range, test_m_range)
    simulate_problem_3(test_n_range, test_m_range)
    sensitivity_analysis_cost_t()

import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局参数设置与数据预处理
# ==========================================
# 费用参数
cost_f = 200   # 故障时产出次品的损失费用 (元/件)
cost_t = 10    # 单次检查费用 (元/次)
cost_d = 3000  # 故障恢复并换刀费用 (元/次)
cost_k = 1000  # 正常预防性换刀费用 (元/次)
cost_w = 1500  # 误判停机的惩罚费用 (元/次)

# 概率参数
p1 = 0.02      # 机器正常时，产出次品的概率
p2 = 0.40      # 机器故障时，产出合格品的概率
p_fail_defect = 1 - p2 # 机器故障时，产出次品的概率 (0.60)

# 历史故障数据
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

# 拟合正态分布 N(mu, sigma^2)
mu, std = np.mean(data), np.std(data, ddof=1)
print(f"--- 数据拟合结果 ---")
print(f"刀具寿命服从正态分布: mu = {mu:.2f}, sigma = {std:.2f}\n")

# 定义概率密度函数 f(x)
def pdf(x):
    return st.norm.pdf(x, loc=mu, scale=std)

# ==========================================
# 问题一：理想状态下的解析模型求解
# ==========================================
def solve_problem_1():
    print("正在求解问题一 (解析法双重遍历) ...")
    min_cost = float('inf')
    best_n = 0
    best_N = 0
    
    # 遍历检查间隔 n 和 周期倍数 m (N = m * n)
    for n in range(10, 200, 5):      # n 取值范围: 10 到 200
        for m in range(1, 20):       # 检查次数范围
            N = m * n
            
            expected_total_cost = 0.0
            expected_total_parts = 0.0
            
            # 1. 预防性更换的情况
            prob_survive = 1 - st.norm.cdf(N, loc=mu, scale=std)
            expected_total_cost += prob_survive * (cost_k + m * cost_t)
            expected_total_parts += prob_survive * N
            
            # 2. 在各个区间内发生故障的情况
            for j in range(1, m + 1):
                lower_bound = (j - 1) * n
                upper_bound = j * n
                
                # 故障落在该区间的概率
                prob_fail_in_j = st.norm.cdf(upper_bound, loc=mu, scale=std) - st.norm.cdf(lower_bound, loc=mu, scale=std)
                if prob_fail_in_j <= 0: continue
                
                # 计算该区间内产出的次品损失积分期望: \int (jn - x) * f(x) dx
                defect_integral, _ = quad(lambda x: (upper_bound - x) * pdf(x), lower_bound, upper_bound)
                
                # 该区间总费用: 恢复费 + j次检查费 + 积分次品费
                cost_j = prob_fail_in_j * (cost_d + j * cost_t) + cost_f * defect_integral
                
                expected_total_cost += cost_j
                expected_total_parts += prob_fail_in_j * upper_bound
            
            # 计算平均单件成本
            if expected_total_parts > 0:
                avg_cost = expected_total_cost / expected_total_parts
                if avg_cost < min_cost:
                    min_cost = avg_cost
                    best_n = n
                    best_N = N

    print(f"【问题一结果】最优检查间隔 n = {best_n} 件, 最优更换周期 N = {best_N} 件")
    print(f"             最低单件平均费用 = {min_cost:.2f} 元/件\n")

# ==========================================
# 问题二：引入误差后的蒙特卡洛模拟求解
# ==========================================
def simulate_problem_2(n_range, m_range, sim_cycles=10000):
    print("正在求解问题二 (蒙特卡洛模拟) ...")
    min_cost = float('inf')
    best_n, best_N = 0, 0
    
    # 预先生成一万个刀具的实际寿命数据
    actual_lifespans = np.random.normal(mu, std, sim_cycles)
    actual_lifespans = np.maximum(actual_lifespans, 1) # 寿命不能小于1
    
    for n in n_range:
        for m in m_range:
            N = m * n
            total_cost = 0
            total_parts = 0
            
            for lifespan in actual_lifespans:
                cycle_cost = 0
                cycle_parts = 0
                machine_stopped = False
                
                for j in range(1, m + 1):
                    current_part_num = j * n
                    cycle_cost += cost_t  # 花费一次检查费
                    
                    is_failed = (lifespan <= current_part_num)
                    
                    # 模拟检查一个零件（问题二策略：只查一件）
                    if not is_failed:
                        # 机器正常，有2%概率产出次品导致误报
                        if np.random.rand() < p1:
                            cycle_cost += cost_w + cost_k  # 误判罚款 + 提前预防性换新刀
                            cycle_parts = current_part_num
                            machine_stopped = True
                            break 
                    else:
                        # 机器已坏，有60%概率查出次品
                        if np.random.rand() < p_fail_defect:
                            # 准确查出故障
                            cycle_cost += cost_d + cost_f * (current_part_num - lifespan)
                            cycle_parts = current_part_num
                            machine_stopped = True
                            break
                        else:
                            # 漏报了！继续带着故障生产，损失会在下一次查出时结算
                            pass 
                
                # 如果一直没被检查停机，到了 N 强行更换
                if not machine_stopped:
                    cycle_parts = N
                    if lifespan <= N:
                        # 虽然到了N换下，但其实中途已经坏了，期间全是次品
                        cycle_cost += cost_d + cost_f * (N - lifespan)
                    else:
                        cycle_cost += cost_k # 正常预防性更换
                
                total_cost += cycle_cost
                total_parts += cycle_parts
                
            avg_cost = total_cost / total_parts
            if avg_cost < min_cost:
                min_cost = avg_cost
                best_n, best_N = n, N
                
    print(f"【问题二结果】最优检查间隔 n = {best_n} 件, 最优更换周期 N = {best_N} 件")
    print(f"             最低单件平均费用 = {min_cost:.2f} 元/件\n")

# ==========================================
# 问题三：改进检查机制（连续抽样2次）
# ==========================================
def simulate_problem_3(n_range, m_range, sim_cycles=10000):
    print("正在求解问题三 (序贯抽样改进机制) ...")
    # 策略定义：每次发现次品不停机，立即再检查下一件。如果连续2件次品，才判定故障停机。
    min_cost = float('inf')
    best_n, best_N = 0, 0
    
    actual_lifespans = np.random.normal(mu, std, sim_cycles)
    actual_lifespans = np.maximum(actual_lifespans, 1)
    
    for n in n_range:
        for m in m_range:
            N = m * n
            total_cost = 0
            total_parts = 0
            
            for lifespan in actual_lifespans:
                cycle_cost = 0
                cycle_parts = 0
                machine_stopped = False
                
                for j in range(1, m + 1):
                    current_part_num = j * n
                    cycle_cost += cost_t  # 第一次检查费用
                    is_failed = (lifespan <= current_part_num)
                    
                    # 第一次检查出现次品
                    first_check_defective = (not is_failed and np.random.rand() < p1) or (is_failed and np.random.rand() < p_fail_defect)
                    
                    if first_check_defective:
                        # 触发改进策略：立即再检查一个零件
                        cycle_cost += cost_t 
                        # 评估第二件的状态
                        is_failed_second = (lifespan <= current_part_num + 1)
                        second_check_defective = (not is_failed_second and np.random.rand() < p1) or (is_failed_second and np.random.rand() < p_fail_defect)
                        
                        if second_check_defective:
                            # 连续两次次品，断定故障
                            if not is_failed_second: # 两次都是误报（极小概率）
                                cycle_cost += cost_w + cost_k
                            else: # 真正故障
                                cycle_cost += cost_d + cost_f * (current_part_num + 1 - lifespan)
                            cycle_parts = current_part_num + 1
                            machine_stopped = True
                            break
                        else:
                            # 第二个是合格的，虚惊一场，继续生产
                            pass

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
                
    print(f"【问题三结果】最优检查间隔 n = {best_n} 件, 最优更换周期 N = {best_N} 件")
    print(f"             最低单件平均费用 = {min_cost:.2f} 元/件")

# 执行主程序
if __name__ == "__main__":
    solve_problem_1()
    # 缩小搜索范围以加快蒙特卡洛模拟速度
    test_n_range = range(20, 100, 5)
    test_m_range = range(5, 15)
    simulate_problem_2(test_n_range, test_m_range)
    simulate_problem_3(test_n_range, test_m_range)
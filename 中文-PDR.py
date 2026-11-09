import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import math
import os
from matplotlib.ticker import MaxNLocator

# ==========================================
# [配置区域：单文件版本]
# ==========================================
# 使用相对于脚本目录的路径，避免项目移动后绝对路径失效。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, '实测数据', '第8次移动惯导_融合.csv')

# 图片和汇总结果的输出目录。
BASE_SAVE_DIR = os.path.join(SCRIPT_DIR, '中文结果图', 'PDR')

# Recommended paper name for the optimized PDR method:
# Adaptive Dual-Constraint PDR (ADC-PDR)
OPT_PDR_NAME = "ADC-PDR"
OPT_PDR_FULL_NAME = "自适应双约束 PDR"

# Whether to pop up each generated figure after saving.
# Set to False if you only want to save figures without displaying them.
SHOW_FIGURES = False

# True: the program pauses until you close the figure window.
# False: the program continues automatically after showing the figure briefly.
SHOW_BLOCKING = True

# ==========================================
# 真实运动轨迹 (矩形)
# ==========================================
GT_PTS = [
    (0.6, 2.4),  # 起点 (左下)
    (0.6, 18),  # 拐角 1 (左上) -> 向上走
    (4.2, 18),  # 拐角 2 (右上) -> 向右走
    (4.2, 2.4),  # 拐角 3 (右下) -> 向下走
    (0.6, 2.4)  # 终点 (回到起点)-> 向左走
]
INITIAL_HEADING = np.pi / 2

# 绘图全局设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 辅助函数区
# ==========================================
def lowpass_filter(data, cutoff=3.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def find_step_valleys(acc_data, distance=20):
    inverted_acc = -acc_data
    valleys, _ = find_peaks(inverted_acc, distance=distance)
    return valleys


def dynamic_peak_time_detection(acc_data, fs=100.0):
    MIN_AMP_BASE = 0.03
    MIN_DIST_SAMPLES = int(0.35 * fs)
    MAX_DIST_SAMPLES = int(1.5 * fs)
    WIN_SIZE = int(0.8 * fs)
    ALPHA = 0.10
    BETA = 0.60

    candidates, _ = find_peaks(acc_data, height=MIN_AMP_BASE, distance=MIN_DIST_SAMPLES)
    valid_peaks = []
    step_intervals = []

    for p in candidates:
        s_idx = max(0, p - WIN_SIZE // 2)
        e_idx = min(len(acc_data), p + WIN_SIZE // 2)
        local_window = acc_data[s_idx:e_idx]
        local_range = np.max(local_window) - np.min(local_window)
        dynamic_amp_threshold = max(MIN_AMP_BASE, ALPHA * local_range)

        if acc_data[p] < dynamic_amp_threshold:
            continue

        if len(valid_peaks) > 0:
            last_peak = valid_peaks[-1]
            time_diff = p - last_peak

            if time_diff > MAX_DIST_SAMPLES:
                step_intervals = []
                dynamic_time_threshold = MIN_DIST_SAMPLES
            else:
                if len(step_intervals) > 0:
                    avg_interval = np.mean(step_intervals[-3:])
                    dynamic_time_threshold = min(int(0.8 * fs), max(MIN_DIST_SAMPLES, int(BETA * avg_interval)))
                else:
                    dynamic_time_threshold = MIN_DIST_SAMPLES

            if time_diff < dynamic_time_threshold:
                continue

            if time_diff < int(1.0 * fs):
                step_intervals.append(time_diff)

        valid_peaks.append(p)

    return np.array(valid_peaks)


def get_segment_by_heading(theta):
    if theta > np.pi / 4:
        return '上'
    elif theta > -np.pi / 4:
        return '右'
    elif theta > -3 * np.pi / 4:
        return '下'
    else:
        return '左'


def save_show_close(save_path, dpi=300):
    """Save the current figure and defer displaying until all figures are ready."""
    plt.savefig(save_path, dpi=dpi)
    if not SHOW_FIGURES:
        plt.close()


def show_all_figures():
    """Pop up all generated figures at once."""
    if not SHOW_FIGURES:
        return
    try:
        plt.show(block=SHOW_BLOCKING)
    except TypeError:
        # Compatible with older Matplotlib versions that do not support block=
        plt.show()
    finally:
        plt.close('all')


# ==========================================
# 核心处理引擎
# ==========================================
def process_single_file(file_path, save_dir, file_name):
    print(f"\n---> 正在处理: {file_name}")

    try:
        imu_df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[!] {file_name} 数据读取失败跳过: {e}")
        return None

    imu_df['Timestamp'] = pd.to_datetime(imu_df['Timestamp'])
    samples = range(len(imu_df))

    gt_dists = [math.hypot(GT_PTS[i + 1][0] - GT_PTS[i][0], GT_PTS[i + 1][1] - GT_PTS[i][1]) for i in
                range(len(GT_PTS) - 1)]
    total_gt_distance = sum(gt_dists)
    start_time = imu_df['Timestamp'].iloc[0]
    total_dur = (imu_df['Timestamp'].iloc[-1] - start_time).total_seconds()

    def get_ground_truth(ts):
        elapsed = (ts - start_time).total_seconds()
        prog_dist = (elapsed / total_dur) * total_gt_distance if total_dur > 0 else 0
        accum_dist = 0
        for i in range(len(gt_dists)):
            if prog_dist <= accum_dist + gt_dists[i]:
                segment_prog = prog_dist - accum_dist
                ratio = segment_prog / gt_dists[i] if gt_dists[i] > 0 else 1.0
                x = GT_PTS[i][0] + ratio * (GT_PTS[i + 1][0] - GT_PTS[i][0])
                y = GT_PTS[i][1] + ratio * (GT_PTS[i + 1][1] - GT_PTS[i][1])
                return x, y
            accum_dist += gt_dists[i]
        return GT_PTS[-1][0], GT_PTS[-1][1]

    imu_df['Acc_Mag'] = np.sqrt(imu_df['加速度X(g)'] ** 2 + imu_df['加速度Y(g)'] ** 2 + imu_df['加速度Z(g)'] ** 2)
    imu_df['Acc_Linear'] = imu_df['Acc_Mag'] - imu_df['Acc_Mag'].mean()
    imu_df['Acc_Filtered'] = lowpass_filter(imu_df['Acc_Linear'].values)

    acc_raw_ms2 = imu_df['Acc_Linear'] * 9.80665
    acc_filtered_ms2 = imu_df['Acc_Filtered'] * 9.80665

    time_diffs = imu_df['Timestamp'].diff().dt.total_seconds().fillna(0.01).values
    gyro_y_rad = imu_df['角速度Y(°/s)'].values * (np.pi / 180.0)
    imu_df['Heading'] = INITIAL_HEADING + np.cumsum(gyro_y_rad * time_diffs)

    valleys = find_step_valleys(imu_df['Acc_Filtered'].values)
    K_WEINBERG = 0.82
    alpha_smooth = 0.65

    # =========================================================
    # 🔴 组别 A & B：传统基础 PDR / 消融组
    # [重磅更新：同样基于滤波后的加速度，严格控制变量]
    # =========================================================
    ACC_THRESHOLD_ORIG = 0.1  # 由于滤波削峰了，固定阈值可稍作降低，默认设0.12g左右
    peaks_orig, _ = find_peaks(imu_df['Acc_Filtered'].values, distance=35, height=ACC_THRESHOLD_ORIG)

    # 统计器：步数与距离
    fix_seg_counts = {'上': 0, '右': 0, '下': 0, '左': 0}
    fix_seg_dists = {'上': 0.0, '右': 0.0, '下': 0.0, '左': 0.0}

    # 组别A推演
    STEP_LENGTH_FIXED = 0.7
    fix_x, fix_y = GT_PTS[0][0], GT_PTS[0][1]
    fix_traj_x, fix_traj_y = [fix_x], [fix_y]
    fix_errors = []

    for p in peaks_orig:
        ts = imu_df.iloc[p]['Timestamp']
        theta = imu_df.iloc[p]['Heading']
        seg = get_segment_by_heading(theta)

        fix_seg_counts[seg] += 1
        fix_seg_dists[seg] += STEP_LENGTH_FIXED

        fix_x += STEP_LENGTH_FIXED * np.cos(theta)
        fix_y += STEP_LENGTH_FIXED * np.sin(theta)
        fix_traj_x.append(fix_x)
        fix_traj_y.append(fix_y)
        gt_x, gt_y = get_ground_truth(ts)
        fix_errors.append(math.sqrt((fix_x - gt_x) ** 2 + (fix_y - gt_y) ** 2))

    # 组别B：解算经验步长
    emp_raw_step_lengths = []
    for p in peaks_orig:
        left_valleys = valleys[valleys < p]
        s_idx = left_valleys[-1] if len(left_valleys) > 0 else max(0, p - 30)
        right_valleys = valleys[valleys > p]
        e_idx = right_valleys[0] if len(right_valleys) > 0 else min(len(imu_df) - 1, p + 30)
        s_idx = max(p - 60, s_idx)
        e_idx = min(p + 60, e_idx)
        acc_win = imu_df['Acc_Filtered'].iloc[s_idx:e_idx]
        acc_range = acc_win.max() - acc_win.min() if len(acc_win) > 0 else 0
        emp_raw_step_lengths.append(np.clip(K_WEINBERG * math.pow(max(1e-5, acc_range), 0.25), 0.4, 1.1))

    emp_step_lengths = []
    for i, l in enumerate(emp_raw_step_lengths):
        emp_step_lengths.append(l if i == 0 else alpha_smooth * l + (1 - alpha_smooth) * emp_step_lengths[-1])

    # 组别B推演
    emp_seg_counts = {'上': 0, '右': 0, '下': 0, '左': 0}
    emp_seg_dists = {'上': 0.0, '右': 0.0, '下': 0.0, '左': 0.0}
    emp_x, emp_y = GT_PTS[0][0], GT_PTS[0][1]
    emp_traj_x, emp_traj_y = [emp_x], [emp_y]
    emp_errors = []

    for idx, p in enumerate(peaks_orig):
        ts = imu_df.iloc[p]['Timestamp']
        theta = imu_df.iloc[p]['Heading']
        step_l = emp_step_lengths[idx]
        seg = get_segment_by_heading(theta)

        emp_seg_counts[seg] += 1
        emp_seg_dists[seg] += step_l

        emp_x += step_l * np.cos(theta)
        emp_y += step_l * np.sin(theta)
        emp_traj_x.append(emp_x)
        emp_traj_y.append(emp_y)
        gt_x, gt_y = get_ground_truth(ts)
        emp_errors.append(math.sqrt((emp_x - gt_x) ** 2 + (emp_y - gt_y) ** 2))

    # =========================================================
    # 🟢 组别 C：极限优化组 PDR (动态阈值寻峰 + 经验步长)
    # =========================================================
    peaks_opt = dynamic_peak_time_detection(imu_df['Acc_Filtered'].values, fs=100.0)

    dyn_seg_counts = {'上': 0, '右': 0, '下': 0, '左': 0}
    dyn_seg_dists = {'上': 0.0, '右': 0.0, '下': 0.0, '左': 0.0}

    raw_step_lengths = []
    weinberg_plot_data = []

    for p in peaks_opt:
        left_valleys = valleys[valleys < p]
        s_idx = left_valleys[-1] if len(left_valleys) > 0 else max(0, p - 30)
        right_valleys = valleys[valleys > p]
        e_idx = right_valleys[0] if len(right_valleys) > 0 else min(len(imu_df) - 1, p + 30)
        s_idx = max(p - 60, s_idx)
        e_idx = min(p + 60, e_idx)

        acc_win = imu_df['Acc_Filtered'].iloc[s_idx:e_idx]
        if len(acc_win) == 0:
            acc_range, max_idx, min_idx = 0, p, p
            max_val, min_val = imu_df['Acc_Filtered'].iloc[p], imu_df['Acc_Filtered'].iloc[p]
        else:
            max_val, min_val = acc_win.max(), acc_win.min()
            acc_range = max_val - min_val
            max_idx, min_idx = acc_win.idxmax(), acc_win.idxmin()

        step_len = K_WEINBERG * math.pow(max(1e-5, acc_range), 0.25)
        raw_step_lengths.append(np.clip(step_len, 0.4, 1.1))

        weinberg_plot_data.append({
            'peak': p, 'max_idx': max_idx, 'min_idx': min_idx,
            'max_val_ms2': max_val * 9.80665, 'min_val_ms2': min_val * 9.80665
        })

    step_lengths = []
    for i, l in enumerate(raw_step_lengths):
        step_lengths.append(l if i == 0 else alpha_smooth * l + (1 - alpha_smooth) * step_lengths[-1])

    dyn_x, dyn_y = GT_PTS[0][0], GT_PTS[0][1]
    dyn_traj_x, dyn_traj_y = [dyn_x], [dyn_y]
    dyn_errors = []

    for idx, p in enumerate(peaks_opt):
        ts = imu_df.iloc[p]['Timestamp']
        theta = imu_df.iloc[p]['Heading']
        step_l = step_lengths[idx]
        seg = get_segment_by_heading(theta)

        dyn_seg_counts[seg] += 1
        dyn_seg_dists[seg] += step_l

        dyn_x += step_l * np.cos(theta)
        dyn_y += step_l * np.sin(theta)
        dyn_traj_x.append(dyn_x)
        dyn_traj_y.append(dyn_y)
        gt_x, gt_y = get_ground_truth(ts)
        dyn_errors.append(math.sqrt((dyn_x - gt_x) ** 2 + (dyn_y - gt_y) ** 2))

    mean_error_fix = np.mean(fix_errors) if fix_errors else 0
    mean_error_emp = np.mean(emp_errors) if emp_errors else 0
    mean_error_dyn = np.mean(dyn_errors) if dyn_errors else 0
    mse_error_fix = np.mean(np.square(fix_errors)) if fix_errors else 0
    mse_error_emp = np.mean(np.square(emp_errors)) if emp_errors else 0
    mse_error_dyn = np.mean(np.square(dyn_errors)) if dyn_errors else 0

    # =========================================================
    # 📸 [图表渲染保存部分] (七大图表完整版)
    # =========================================================

    # 🖼️ 图1：滤波前后对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(samples, acc_raw_ms2, color='blue', linewidth=1)
    ax1.set_title("原始加速度", fontsize=14)
    ax1.set_ylabel("加速度（m/s$^2$）", fontsize=12)  # 修复上标警告
    ax2.plot(samples, acc_filtered_ms2, color='red', linewidth=1)
    ax2.set_title("滤波后加速度", fontsize=14)
    ax2.set_ylabel("加速度（m/s$^2$）", fontsize=12)  # 修复上标警告
    ax2.set_xlabel("样本序号", fontsize=12)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图1_加速度滤波对比.png"), dpi=200)

    # 🖼️ 图2：传统固定阈值寻峰图 (🌟 核心修改区)
    plt.figure(figsize=(10, 4))
    # 1. 曲线替换为滤波后的数据 acc_filtered_ms2
    plt.plot(samples, acc_filtered_ms2, color='#A9A9A9', linewidth=1.2, alpha=0.8, label='滤波后线性加速度')
    threshold_ms2 = ACC_THRESHOLD_ORIG * 9.80665
    plt.axhline(y=threshold_ms2, color='green', linestyle='--', linewidth=1.5, label=f'固定阈值（{ACC_THRESHOLD_ORIG}g）')
    if len(peaks_orig) > 0:
        # 2. 打点的 Y 坐标也替换为滤波后的对应值
        plt.plot(peaks_orig, acc_filtered_ms2.iloc[peaks_orig], 'bs', markersize=6, label=f'固定阈值峰值：{len(peaks_orig)} 步')
    plt.title("传统固定阈值计步检测", fontsize=14)
    plt.ylabel("加速度（m/s$^2$）", fontsize=12)  # 修复上标警告
    plt.xlabel("样本序号", fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图2_固定阈值计步检测.png"), dpi=200)

    # 🖼️ 图3：极限优化动阈寻峰图
    plt.figure(figsize=(10, 4))
    plt.plot(samples, acc_filtered_ms2, color='red', linewidth=1.2, alpha=0.8, label='滤波后加速度')
    plt.plot(peaks_opt, acc_filtered_ms2.iloc[peaks_opt], 'bo', markersize=6, label=f'自适应阈值峰值：{len(peaks_opt)} 步')
    plt.title("自适应双约束计步检测", fontsize=14)
    plt.ylabel("加速度（m/s$^2$）", fontsize=12)  # 修复上标警告
    plt.xlabel("样本序号", fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图3_自适应双约束计步检测.png"), dpi=200)

    # 🖼️ 图4：Weinberg 波谷特征提取特写
    plt.figure(figsize=(12, 4))
    if len(peaks_opt) >= 6:
        s_plot, e_plot = max(0, peaks_opt[1] - 40), min(len(imu_df) - 1, peaks_opt[5] + 40)
    else:
        s_plot, e_plot = 0, min(500, len(imu_df) - 1)
    plt.plot(range(s_plot, e_plot), acc_filtered_ms2.iloc[s_plot:e_plot], color='black', linewidth=1.5)
    for d in weinberg_plot_data:
        if s_plot <= d['peak'] <= e_plot:
            plt.plot(d['max_idx'], d['max_val_ms2'], '^', color='red', markersize=9)
            plt.plot(d['min_idx'], d['min_val_ms2'], 'v', color='blue', markersize=9)
            plt.vlines(x=d['max_idx'], ymin=d['min_val_ms2'], ymax=d['max_val_ms2'], colors='green',
                       linestyles='dashdot')
    plt.title("基于 Weinberg 模型的峰谷特征提取", fontsize=14)
    plt.ylabel("加速度（m/s$^2$）", fontsize=12)  # 修复上标警告
    plt.xlim(s_plot, e_plot)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图4_Weinberg峰谷特征提取.png"), dpi=200)

    # 🖼️ 图4：Weinberg 波谷特征提取特写
    plt.figure(figsize=(12, 4))
    if len(peaks_opt) >= 6:
        s_plot, e_plot = max(0, peaks_opt[1] - 40), min(len(imu_df) - 1, peaks_opt[5] + 40)
    else:
        s_plot, e_plot = 0, min(500, len(imu_df) - 1)
    plt.plot(range(s_plot, e_plot), acc_filtered_ms2.iloc[s_plot:e_plot], color='black', linewidth=1.5)
    for d in weinberg_plot_data:
        if s_plot <= d['peak'] <= e_plot:
            plt.plot(d['max_idx'], d['max_val_ms2'], '^', color='red', markersize=9)
            plt.plot(d['min_idx'], d['min_val_ms2'], 'v', color='blue', markersize=9)
            plt.vlines(x=d['max_idx'], ymin=d['min_val_ms2'], ymax=d['max_val_ms2'], colors='green',
                       linestyles='dashdot')
    plt.title("基于 Weinberg 模型的峰谷特征提取", fontsize=14)
    plt.ylabel("加速度（m/s$^2$）", fontsize=12)
    plt.xlim(s_plot, e_plot)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图4_Weinberg峰谷特征提取.png"), dpi=200)

    # 🖼️ 图5：步长序列图 (平滑前 vs 平滑后)
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, len(step_lengths) + 1), step_lengths, marker='o', color='#ff7f0e', markersize=6, label='平滑后步长')
    # plt.plot(range(1, len(raw_step_lengths) + 1), raw_step_lengths, marker='.', color='gray', linestyle=':', alpha=0.5,
    #          label='原始解算步长')
    plt.title("动态步长", fontsize=14, fontweight='bold')
    plt.xlabel("步序号", fontsize=12)
    plt.ylabel("步长（m）", fontsize=12)
    plt.ylim(0.2, 1.2)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper right', edgecolor='black', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图5_动态步长.png"), dpi=300)

    # 🖼️ 图6：航向角积分图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(samples, imu_df['角速度Y(°/s)'].values, color='teal', linewidth=1, label='Y 轴角速度')
    ax1.set_title("原始单轴角速度", fontsize=14)
    ax1.set_ylabel("角速度（°/s）", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, len(samples))
    ax1.grid(True, linestyle='--', alpha=0.6)

    cumulative_theta_deg = np.degrees(imu_df['Heading'].values)
    ax2.plot(samples, cumulative_theta_deg, color='purple', linewidth=1.5, label='绝对航向角')
    ax2.set_title("积分航向估计", fontsize=14)
    ax2.set_xlabel("样本序号", fontsize=12)
    ax2.set_ylabel("航向角（°）", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, len(samples))
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图6_航向角估计.png"), dpi=200)

    # # 🖼️ 图7：三组轨迹对比图 (压轴图表)
    # plt.figure(figsize=(10, 11))
    # gt_xs, gt_ys = [pt[0] for pt in GT_PTS], [pt[1] for pt in GT_PTS]
    # plt.plot(gt_xs, gt_ys, color='green', linestyle='--', linewidth=3.5, label='Ground Truth')
    # plt.plot(fix_traj_x, fix_traj_y, color='royalblue', marker='s', markersize=4, linestyle='-', linewidth=2,
    #          label=f'Traditional PDR [Fixed Threshold + Fixed Step] (Mean Error: {mean_error_fix:.2f}m)')
    # plt.plot(emp_traj_x, emp_traj_y, color='purple', marker='^', markersize=4, linestyle='-.', linewidth=2,
    #          label=f'Ablation PDR [Fixed Threshold + Weinberg Step] (Mean Error: {mean_error_emp:.2f}m)')
    # plt.plot(dyn_traj_x, dyn_traj_y, color='darkorange', marker='o', markersize=5, linestyle='-', linewidth=2.5,
    #          label=f'{OPT_PDR_NAME} [Adaptive Constraints + Weinberg Step] (Mean Error: {mean_error_dyn:.2f}m)')
    #
    # plt.scatter([GT_PTS[0][0]], [GT_PTS[0][1]], color='black', s=150, zorder=6, label='Start')
    # plt.scatter([GT_PTS[-1][0]], [GT_PTS[-1][1]], color='green', marker='s', s=120, zorder=6, label='End')
    # plt.title(f"[{file_name}] PDR 消融实验轨迹对比", fontsize=16)
    # plt.legend()
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"{file_name}_图7_三组轨迹对比.png"), dpi=300)
    # plt.close()
    # # =========================================================
    # # 数据汇总格式化：包含多维度的步数、距离
    # # =========================================================
    #
    # # 格式：上24步(15.2m)|右7步(4.5m)|...
    # def format_seg_info(counts, dists):
    #     return f"上{counts['上']}步({dists['上']:.1f}m) | " \
    #            f"右{counts['右']}步({dists['右']:.1f}m) | " \
    #            f"下{counts['下']}步({dists['下']:.1f}m) | " \
    #            f"左{counts['左']}步({dists['左']:.1f}m)"
    #
    # fix_info = format_seg_info(fix_seg_counts, fix_seg_dists)
    # emp_info = format_seg_info(emp_seg_counts, emp_seg_dists)
    # dyn_info = format_seg_info(dyn_seg_counts, dyn_seg_dists)
    #
    # total_dist_fix = sum(fix_seg_dists.values())
    # total_dist_emp = sum(emp_seg_dists.values())
    # total_dist_dyn = sum(dyn_seg_dists.values())
    #
    # # =========================================================
    # # 数据汇总格式化：重构成横纵互换的标准数据框格式
    # # =========================================================
    # total_dist_fix = sum(fix_seg_dists.values())
    # total_dist_emp = sum(emp_seg_dists.values())
    # total_dist_dyn = sum(dyn_seg_dists.values())
    #
    # # 返回一个列表，包含当前文件的 3 行数据（A组、B组、C组）
    # return [
    #     {
    #         "Test File": file_name,
    #         "Method": "Traditional PDR [Fixed Threshold + Fixed Step]",
    #         "误差(m)": round(mean_error_fix, 3),
    #         "Total Steps": len(peaks_orig),
    #         "Total Distance (m)": round(total_dist_fix, 2),
    #         "Up Side": f"{fix_seg_counts['上']} steps ({fix_seg_dists['上']:.1f}m)",
    #         "Right Side": f"{fix_seg_counts['右']} steps ({fix_seg_dists['右']:.1f}m)",
    #         "Down Side": f"{fix_seg_counts['下']} steps ({fix_seg_dists['下']:.1f}m)",
    #         "Left Side": f"{fix_seg_counts['左']} steps ({fix_seg_dists['左']:.1f}m)"
    #     },
    #     {
    #         "Test File": file_name,
    #         "Method": "Ablation PDR [Fixed Threshold + Weinberg Step]",
    #         "误差(m)": round(mean_error_emp, 3),
    #         "Total Steps": len(peaks_orig),  # 传统组和消融组步数一致
    #         "Total Distance (m)": round(total_dist_emp, 2),
    #         "Up Side": f"{emp_seg_counts['上']} steps ({emp_seg_dists['上']:.1f}m)",
    #         "Right Side": f"{emp_seg_counts['右']} steps ({emp_seg_dists['右']:.1f}m)",
    #         "Down Side": f"{emp_seg_counts['下']} steps ({emp_seg_dists['下']:.1f}m)",
    #         "Left Side": f"{emp_seg_counts['左']} steps ({emp_seg_dists['左']:.1f}m)"
    #     },
    #     {
    #         "Test File": file_name,
    #         "Method": f"{OPT_PDR_NAME} [Adaptive Constraints + Weinberg Step]",
    #         "误差(m)": round(mean_error_dyn, 3),
    #         "Total Steps": len(peaks_opt),
    #         "Total Distance (m)": round(total_dist_dyn, 2),
    #         "Up Side": f"{dyn_seg_counts['上']} steps ({dyn_seg_dists['上']:.1f}m)",
    #         "Right Side": f"{dyn_seg_counts['右']} steps ({dyn_seg_dists['右']:.1f}m)",
    #         "Down Side": f"{dyn_seg_counts['下']} steps ({dyn_seg_dists['下']:.1f}m)",
    #         "Left Side": f"{dyn_seg_counts['左']} steps ({dyn_seg_dists['左']:.1f}m)"
    #     }
    # ]
    # 🖼️ 图7：三组轨迹对比图 (压轴图表)
    plt.figure(figsize=(6.5, 8.5))
    gt_xs, gt_ys = [pt[0] for pt in GT_PTS], [pt[1] for pt in GT_PTS]
    plt.plot(gt_xs, gt_ys, color='green', linestyle='--', linewidth=3.5, label='真实轨迹')
    plt.plot(fix_traj_x, fix_traj_y, color='royalblue', marker='s', markersize=4, linestyle='-', linewidth=2,
             label=f'传统 PDR（平均误差：{mean_error_fix:.2f} m）')
    # plt.plot(emp_traj_x, emp_traj_y, color='purple', marker='^', markersize=4, linestyle='-.', linewidth=2,
    #          label=f'Ablation PDR [Fixed Threshold + Weinberg Step] (Mean Error: {mean_error_emp:.2f}m)')
    plt.plot(dyn_traj_x, dyn_traj_y, color='darkorange', marker='o', markersize=5, linestyle='-', linewidth=2.5,
             label=f'{OPT_PDR_NAME}（平均误差：{mean_error_dyn:.2f} m）')
    # plt.plot(fix_traj_x, fix_traj_y, color='royalblue', marker='s', markersize=4, linestyle='-', linewidth=2,
    #          label=f'Traditional PDR')
    # # plt.plot(emp_traj_x, emp_traj_y, color='purple', marker='^', markersize=4, linestyle='-.', linewidth=2,
    # #          label=f'Ablation PDR [Fixed Threshold + Weinberg Step] (Mean Error: {mean_error_emp:.2f}m)')
    # plt.plot(dyn_traj_x, dyn_traj_y, color='darkorange', marker='o', markersize=5, linestyle='-', linewidth=2.5,
    #          label=f'{OPT_PDR_NAME}')

    # plt.scatter([GT_PTS[0][0]], [GT_PTS[0][1]], color='black', s=150, zorder=6, label='Start')
    # plt.scatter([GT_PTS[-1][0]], [GT_PTS[-1][1]], color='green', marker='s', s=120, zorder=6, label='End')
    plt.title("PDR 轨迹对比", fontsize=16, fontweight='bold')
    plt.xlabel("X 坐标（m）", fontsize=12)
    plt.ylabel("Y 坐标（m）", fontsize=12)

    ax = plt.gca()

    # 收集所有轨迹点，用于自动确定显示范围
    all_x = gt_xs + fix_traj_x + dyn_traj_x
    all_y = gt_ys + fix_traj_y + dyn_traj_y

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # x方向适当放宽，避免图像太窄
    ax.set_xlim(x_min - 2, x_max + 7)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # 保证 x、y 坐标比例一致，轨迹不会被拉伸变形
    ax.set_aspect('equal', adjustable='box')



    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.55)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.25)
    plt.legend(loc='best', fontsize=10, frameon=True)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图7_PDR轨迹对比.png"), dpi=300)

    # ==========================================
    # 单独生成 PDR 轨迹图例框
    # ==========================================
    fig_legend, ax_legend = plt.subplots(figsize=(6, 2))

    # 构造虚拟句柄，只用于图例
    h_gt, = ax_legend.plot([], [], color='green', linestyle='--', linewidth=3.5,
                           label='真实轨迹')

    h_trad, = ax_legend.plot([], [], color='royalblue', linestyle='-', linewidth=2,
                             marker='s', markersize=5,
                             label=f'传统 PDR（平均误差：{mean_error_fix:.2f} m）')

    h_opt, = ax_legend.plot([], [], color='darkorange', linestyle='-', linewidth=2.5,
                            marker='o', markersize=5,
                            label=f'{OPT_PDR_NAME}（平均误差：{mean_error_dyn:.2f} m）')

    legend = ax_legend.legend(
        handles=[h_gt, h_trad, h_opt],
        loc='center',
        frameon=True,
        fontsize=12
    )

    ax_legend.axis('off')

    # 先渲染，再获取图例边界
    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())

    legend_png_path = os.path.join(save_dir, "PDR轨迹图例.png")

    # 只保存图例框
    fig_legend.savefig(legend_png_path, dpi=600, bbox_inches=bbox, pad_inches=0.02)


    plt.close(fig_legend)

    print(f"[*] PDR 图例框 PNG 已保存至: {legend_png_path}")



    # =========================================================
    # 🌟 新增：图8 CDF 累积误差分布图 (带 90% 分位线)
    # =========================================================
    plt.figure(figsize=(8, 6))

    # 辅助函数：计算CDF所需数据
    def calc_cdf(errors):
        if not errors: return [], []
        sorted_errs = np.sort(errors)
        p = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
        return sorted_errs, p

    sort_fix, p_fix = calc_cdf(fix_errors)
    sort_emp, p_emp = calc_cdf(emp_errors)
    sort_dyn, p_dyn = calc_cdf(dyn_errors)

    # 计算 90% 分位误差 (CE90)
    ce90_fix = np.percentile(fix_errors, 90) if fix_errors else 0
    ce90_emp = np.percentile(emp_errors, 90) if emp_errors else 0
    ce90_dyn = np.percentile(dyn_errors, 90) if dyn_errors else 0

    # 绘制 CDF 曲线
    plt.plot(sort_fix, p_fix, color='royalblue', linestyle='-', linewidth=2, label=f'传统 PDR（90%误差：{ce90_fix:.2f} m）')
    # plt.plot(sort_emp, p_emp, color='purple', linestyle='-.', linewidth=2, label=f'Ablation PDR (90% Error: {ce90_emp:.2f}m)')
    plt.plot(sort_dyn, p_dyn, color='darkorange', linestyle='-', linewidth=2.5, label=f'{OPT_PDR_NAME}（90%误差：{ce90_dyn:.2f} m）')

    # 添加 90% 水平参考线
    plt.axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(0, 0.92, '90% 累积概率（CE90）', color='dimgray', fontsize=11, fontweight='bold')

    plt.title("定位误差累积分布（CDF）", fontsize=14)
    plt.xlabel("定位误差（m）", fontsize=12)
    plt.ylabel("累积概率（CDF）", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    save_show_close(os.path.join(save_dir, "图8_定位误差累积分布.png"), dpi=300)

    # =========================================================
    # 数据汇总格式化：包含多维度的步数、距离
    # =========================================================

    def format_seg_info(counts, dists):
        return f"上{counts['上']}步({dists['上']:.1f}m) | " \
               f"右{counts['右']}步({dists['右']:.1f}m) | " \
               f"下{counts['下']}步({dists['下']:.1f}m) | " \
               f"左{counts['左']}步({dists['左']:.1f}m)"

    # 计算三组算法各自推算出的总距离
    total_dist_fix = sum(fix_seg_dists.values())
    total_dist_emp = sum(emp_seg_dists.values())
    total_dist_dyn = sum(dyn_seg_dists.values())

    # 返回一个列表，包含当前文件的 3 行数据（A组、B组、C组），并新增了 90% 误差列
    return [
        {
            "测试文件": file_name,
            "方法": "传统 PDR［固定阈值 + 固定步长］",
            "平均误差（m）": round(mean_error_fix, 3),
            "均方误差（m^2）": round(mse_error_fix, 3),
            "CE90（m）": round(ce90_fix, 3),
            "总步数": len(peaks_orig),
            "总距离（m）": round(total_dist_fix, 2),
            "上行段": f"{fix_seg_counts['上']} 步（{fix_seg_dists['上']:.1f} m）",
            "右行段": f"{fix_seg_counts['右']} 步（{fix_seg_dists['右']:.1f} m）",
            "下行段": f"{fix_seg_counts['下']} 步（{fix_seg_dists['下']:.1f} m）",
            "左行段": f"{fix_seg_counts['左']} 步（{fix_seg_dists['左']:.1f} m）"
        },
        {
            "测试文件": file_name,
            "方法": "消融 PDR［固定阈值 + Weinberg 步长］",
            "平均误差（m）": round(mean_error_emp, 3),
            "均方误差（m^2）": round(mse_error_emp, 3),
            "CE90（m）": round(ce90_emp, 3),
            "总步数": len(peaks_orig),
            "总距离（m）": round(total_dist_emp, 2),
            "上行段": f"{emp_seg_counts['上']} 步（{emp_seg_dists['上']:.1f} m）",
            "右行段": f"{emp_seg_counts['右']} 步（{emp_seg_dists['右']:.1f} m）",
            "下行段": f"{emp_seg_counts['下']} 步（{emp_seg_dists['下']:.1f} m）",
            "左行段": f"{emp_seg_counts['左']} 步（{emp_seg_dists['左']:.1f} m）"
        },
        {
            "测试文件": file_name,
            "方法": f"{OPT_PDR_NAME}［自适应约束 + Weinberg 步长］",
            "平均误差（m）": round(mean_error_dyn, 3),
            "均方误差（m^2）": round(mse_error_dyn, 3),
            "CE90（m）": round(ce90_dyn, 3),
            "总步数": len(peaks_opt),
            "总距离（m）": round(total_dist_dyn, 2),
            "上行段": f"{dyn_seg_counts['上']} 步（{dyn_seg_dists['上']:.1f} m）",
            "右行段": f"{dyn_seg_counts['右']} 步（{dyn_seg_dists['右']:.1f} m）",
            "下行段": f"{dyn_seg_counts['下']} 步（{dyn_seg_dists['下']:.1f} m）",
            "左行段": f"{dyn_seg_counts['左']} 步（{dyn_seg_dists['左']:.1f} m）"
        }
    ]


# ==========================================
# Single-file launcher
# ==========================================
def run_single_file_processing():
    if not os.path.exists(BASE_SAVE_DIR):
        os.makedirs(BASE_SAVE_DIR)

    print("=" * 110)
    print(f"开始：单文件 PDR 消融评估（{OPT_PDR_FULL_NAME}，{OPT_PDR_NAME}）")
    print("=" * 110)

    if not os.path.isfile(DATA_FILE):
        print(f"[!] 未找到 CSV 文件：{DATA_FILE}")
        print("[!] 请检查 DATA_FILE，确保它指向具体的 .csv 文件，而不是文件夹。")
        return

    file_name = os.path.splitext(os.path.basename(DATA_FILE))[0]
    save_dir = os.path.join(BASE_SAVE_DIR, file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    summary_results = process_single_file(DATA_FILE, save_dir, file_name)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1500)
        pd.set_option('display.unicode.east_asian_width', True)

        print("\n" + summary_df.to_string(index=False))

        summary_csv_path = os.path.join(save_dir, "PDR消融实验汇总.csv")
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print("\n" + "-" * 110)
        print(f"[完成] 图片和汇总表已保存至：{save_dir}")
        print(f"[完成] 汇总 CSV：{summary_csv_path}")
    else:
        print("未生成有效结果。")

    show_all_figures()


if __name__ == "__main__":
    run_single_file_processing()

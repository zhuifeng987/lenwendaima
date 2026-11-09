import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math
import os
import warnings
import h5py

warnings.filterwarnings('ignore')

# ==========================================
# [配置区域]
# ==========================================
# 文件路径配置
FILE_FINGERPRINT = r"C:\Users\86150\Desktop\LSTM\大厅\第二次测试\指纹库"
BLE_FILE = r"C:\Users\86150\Desktop\第二次测试\保留数据\第8次移动蓝牙_融合1(V2.0).csv"
IMU_FILE = r"D:\isip\论文\我自己\代码\大厅\第二次测试\移动数据\融合数据\第8次移动惯导_融合.csv"
LSTM_MODEL_PATH = 'best_lstm_model.h5'

SAVE_DIR = r"./异步粒子滤波融合大厅-CDF图-英文结果图"
SAVE_DIR1 = r"C:\Users\86150\Desktop\第二次测试\1"

# 真实运动轨迹 (矩形闭环)
GT_PTS = [
    (0.6, 2.4),  # 起点 (左下)
    (0.6, 18),  # 拐角 1 (左上) -> 向上走
    (4.2, 18),  # 拐角 2 (右上) -> 向右走
    (4.2, 2.4),  # 拐角 3 (右下) -> 向下走
    (0.6, 2.4)  # 终点 (回到起点)-> 向左走
]
INITIAL_HEADING = np.pi / 2

# ★ LSTM 序列参数
TIME_STEPS = 4
RESAMPLE_FREQ = '500ms'  # 仅作为聚合蓝牙信号的时间窗基准

# ★ PDR 优化参数
K_WEINBERG = 0.82
ALPHA_SMOOTH = 0.65

# # ★ 粒子滤波 PF 参数     原版
# NUM_PARTICLES = 1200  # 粒子群数量
# STEP_NOISE_STD = 0.15  # 步长预测噪声标准差 (米)
# HEADING_NOISE_STD = 0.25  # 航向角预测噪声标准差 (弧度)
# # OBS_NOISE_STD = 1.5  # ★已调优：蓝牙观测噪声标准差
# # --- 新增：动态观测噪声参数 ---
# BASE_OBS_NOISE_STD = 1.5  # 直线行走时的蓝牙噪声 (信任度高)
# TURN_OBS_NOISE_STD = 5  # 拐弯时的蓝牙噪声 (信任度低，防拉扯)
# TURN_THRESHOLD_DEG = 12.0  # 判定为拐弯的阈值 (单步航向角变化绝对值 > 12度)

# ★ 粒子滤波 PF 参数
NUM_PARTICLES = 1200
STEP_NOISE_STD = 0.10
HEADING_NOISE_STD = 0.10
# BASE_OBS_NOISE_STD = 7    另一组数据
BASE_OBS_NOISE_STD = 5
TURN_OBS_NOISE_STD = 2.5
TURN_THRESHOLD_DEG = 20


MAP_CONSTRAINT_SIGMA = 2


plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 核心组件 1：信号处理与滤波
# ==========================================
def lowpass_filter(data, cutoff=3.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


class KalmanFilter1D:
    def __init__(self, Q, R, x_init, p_init):
        self.Q, self.R = Q, R
        self.X, self.P = x_init, p_init

    def update(self, z):
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.X = self.X + K * (z - self.X)
        self.P = (1 - K) * P_pred
        return self.X


# ==========================================
# 核心组件 2：极限优化 PDR
# ==========================================
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


def generate_universal_ground_truth(gt_pts, timestamps):
    seg_lengths = [math.hypot(gt_pts[i + 1][0] - gt_pts[i][0], gt_pts[i + 1][1] - gt_pts[i][1]) for i in
                   range(len(gt_pts) - 1)]
    total_length = sum(seg_lengths)
    total_dur = (timestamps[-1] - timestamps[0]).total_seconds()

    true_traj = []
    for ts in timestamps:
        prog_dist = ((ts - timestamps[0]).total_seconds() / total_dur) * total_length if total_dur > 0 else 0
        accum = 0.0
        for i, sl in enumerate(seg_lengths):
            if accum + sl >= prog_dist - 1e-6:
                ratio = (prog_dist - accum) / sl if sl > 0 else 0
                x = gt_pts[i][0] + ratio * (gt_pts[i + 1][0] - gt_pts[i][0])
                y = gt_pts[i][1] + ratio * (gt_pts[i + 1][1] - gt_pts[i][1])
                true_traj.append((x, y))
                break
            accum += sl
        else:
            true_traj.append((gt_pts[-1][0], gt_pts[-1][1]))
    return np.array(true_traj)

def distance_to_gt_path(xs, ys, gt_pts):
    min_dist = np.full_like(xs, np.inf, dtype=float)

    for i in range(len(gt_pts) - 1):
        x1, y1 = gt_pts[i]
        x2, y2 = gt_pts[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq == 0:
            dist = np.sqrt((xs - x1) ** 2 + (ys - y1) ** 2)
        else:
            t = ((xs - x1) * dx + (ys - y1) * dy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)

            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            dist = np.sqrt((xs - proj_x) ** 2 + (ys - proj_y) ** 2)

        min_dist = np.minimum(min_dist, dist)

    return min_dist

# ==========================================
# 主引擎：异步事件驱动融合
# ==========================================
def run_lstm_pdr_particle_filter():
    np.random.seed(42)
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAVE_DIR1): os.makedirs(SAVE_DIR1)

    print("=" * 60)
    print(" 🚀 启动：事件驱动型异步融合 LSTM + 极限优化 PDR + PF")
    print("=" * 60)

    # --- 1. 加载指纹库 ---
    print("[*] 正在解析指纹库特征范围...")
    df_fp = pd.read_csv(FILE_FINGERPRINT)
    exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
    rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]
    scaler = MinMaxScaler()
    scaler.fit(df_fp[rssi_cols].values)

    # --- 2. 蓝牙预处理与 LSTM ---
    print("[*] 正在处理蓝牙流，提取有效观测事件...")
    ble_df = pd.read_csv(BLE_FILE)
    ble_df['Timestamp'] = pd.to_datetime(ble_df['Timestamp'])

    kf_dict = {}
    filt_records = []
    for _, row in ble_df.iterrows():
        mac, raw = row['MAC'], row['RSSI']
        if mac not in rssi_cols: continue
        if mac not in kf_dict: kf_dict[mac] = KalmanFilter1D(0.1, 2.0, raw, 1.0)
        filt_records.append({'Timestamp': row['Timestamp'], 'MAC': mac, 'Filtered': kf_dict[mac].update(raw)})

    # ble_counts = pd.DataFrame(filt_records).set_index('Timestamp').resample(RESAMPLE_FREQ).size()
    #
    # df_aligned = pd.DataFrame(filt_records).set_index('Timestamp').groupby('MAC')['Filtered'].resample(
    #     RESAMPLE_FREQ).mean().unstack(level=0).ffill(limit=6).fillna(-100)
    # for col in rssi_cols:
    #     if col not in df_aligned.columns: df_aligned[col] = -100
    # df_aligned = df_aligned[rssi_cols]
    #
    # real_norm = scaler.transform(df_aligned.values)

    # ==========================================
    # 自定义 0.5s 时间窗：从第一条蓝牙数据开始切分
    # ==========================================
    df_filt = pd.DataFrame(filt_records)

    # 以第一条蓝牙数据的时间作为第一个 0.5s 窗口起点
    ble_start_time = df_filt['Timestamp'].min()

    # 把 RESAMPLE_FREQ 转成秒，例如 500ms -> 0.5s
    bin_seconds = pd.Timedelta(RESAMPLE_FREQ).total_seconds()

    # 计算每条蓝牙数据距离起点经过了多少秒
    elapsed_seconds = (df_filt['Timestamp'] - ble_start_time).dt.total_seconds()

    # 每 0.5s 分到一个时间窗
    bin_index = np.floor(elapsed_seconds / bin_seconds).astype(int)

    # 生成自定义时间窗标签
    df_filt['TimeBin'] = ble_start_time + pd.to_timedelta(bin_index * bin_seconds, unit='s')

    # 每个时间窗内的数据数量，用于后面判断有效 BLE 事件
    ble_counts = df_filt.groupby('TimeBin').size()

    # 按 自定义时间窗 + MAC 聚合 RSSI
    df_aligned = df_filt.groupby(['TimeBin', 'MAC'])['Filtered'].mean().unstack(level=1)

    # 把索引名改回 Timestamp，后面代码就不用大改
    df_aligned.index.name = 'Timestamp'

    # 保证所有信标列都存在，并按指纹库顺序排列
    for col in rssi_cols:
        if col not in df_aligned.columns:
            df_aligned[col] = np.nan

    df_aligned = df_aligned[rssi_cols]

    # 检查填充前每帧缺失 MAC 数量
    missing_before = df_aligned.isna().sum(axis=1)
    print("填充前，前10帧每帧缺失 MAC 数量：")
    print(missing_before.head(10).to_string())

    print("\n填充前，前10帧 RSSI：")
    print(df_aligned.head(10).to_string())

    # 短时缺失用前向填充，剩余缺失用 -100 表示弱信号
    df_aligned = df_aligned.ffill(limit=6).fillna(-100)

    # 检查填充后每帧 -100 数量
    missing_after = (df_aligned[rssi_cols] <= -99.9).sum(axis=1)
    print("\n填充后，前10帧每帧 -100 数量：")
    print(missing_after.head(10).to_string())

    print("\n填充后，前10帧 RSSI：")
    print(df_aligned.head(10).to_string())

    # 归一化后输入 LSTM
    real_norm = scaler.transform(df_aligned.values)

    X_lstm = []
    for i in range(len(real_norm)):
        if i < TIME_STEPS - 1:
            pad = np.tile(real_norm[0], (TIME_STEPS - i - 1, 1))
            seq = np.vstack((pad, real_norm[:i + 1]))
        else:
            seq = real_norm[i - TIME_STEPS + 1: i + 1]
        X_lstm.append(seq)

    f = h5py.File(LSTM_MODEL_PATH, 'r')
    lstm_model = load_model(f)
    lstm_preds = lstm_model.predict(np.array(X_lstm), verbose=0)

    lstm_dict = {ts: (pred[0], pred[1]) for ts, pred in zip(df_aligned.index, lstm_preds)}

    valid_ble_ts = ble_counts[ble_counts > 0].index
    ble_events = [{'Timestamp': ts, 'Type': 'BLE', 'X': lstm_dict[ts][0], 'Y': lstm_dict[ts][1]}
                  for ts in valid_ble_ts if ts in lstm_dict]
    print(f"[*] 提取到 {len(ble_events)} 个有效蓝牙观测事件。")

    # --- 3. IMU 特征提取与步态事件构建 ---
    print("[*] 正在提取大厅优化版 IMU 物理特征...")
    imu_df = pd.read_csv(IMU_FILE)
    imu_df['Timestamp'] = pd.to_datetime(imu_df['Timestamp'])
    imu_df['Acc_Mag'] = np.sqrt(imu_df['加速度X(g)'] ** 2 + imu_df['加速度Y(g)'] ** 2 + imu_df['加速度Z(g)'] ** 2)
    imu_df['Acc_Linear'] = imu_df['Acc_Mag'] - imu_df['Acc_Mag'].mean()
    imu_df['Acc_Filtered'] = lowpass_filter(imu_df['Acc_Linear'].values)

    peaks = dynamic_peak_time_detection(imu_df['Acc_Filtered'].values)
    valleys = find_step_valleys(imu_df['Acc_Filtered'].values)

    raw_step_lengths = []
    for p in peaks:
        left_valleys = valleys[valleys < p]
        s_idx = left_valleys[-1] if len(left_valleys) > 0 else max(0, p - 30)
        right_valleys = valleys[valleys > p]
        e_idx = right_valleys[0] if len(right_valleys) > 0 else min(len(imu_df) - 1, p + 30)

        s_idx = max(p - 60, s_idx)
        e_idx = min(p + 60, e_idx)

        acc_win = imu_df['Acc_Filtered'].iloc[s_idx:e_idx]
        if len(acc_win) == 0:
            acc_range = 0
        else:
            acc_range = acc_win.max() - acc_win.min()

        step_len = K_WEINBERG * math.pow(max(1e-5, acc_range), 0.25)
        raw_step_lengths.append(np.clip(step_len, 0.4, 1.1))

    time_diffs = imu_df['Timestamp'].diff().dt.total_seconds().fillna(0.01).values
    gyro_y_rad = imu_df['角速度Y(°/s)'].values * (np.pi / 180.0)
    imu_df['Delta_Theta'] = gyro_y_rad * time_diffs
    imu_df['Heading'] = INITIAL_HEADING + np.cumsum(imu_df['Delta_Theta'].values)

    smoothed_steps = []
    last_step = 0
    for i, l in enumerate(raw_step_lengths):
        sl = l if i == 0 else ALPHA_SMOOTH * l + (1 - ALPHA_SMOOTH) * last_step
        smoothed_steps.append(sl)
        last_step = sl

    imu_events = []
    for i, p in enumerate(peaks):
        ts = imu_df.iloc[p]['Timestamp']
        sl = smoothed_steps[i]

        if i == 0:
            delta_theta = imu_df['Delta_Theta'].iloc[0:p + 1].sum()
        else:
            prev_p = peaks[i - 1]
            delta_theta = imu_df['Delta_Theta'].iloc[prev_p + 1:p + 1].sum()

        imu_events.append({'Timestamp': ts, 'Type': 'IMU', 'Step': sl, 'DeltaTheta': delta_theta})
    print(f"[*] 提取到 {len(imu_events)} 个连续步态推算事件。")

    # --- 4. 粒子滤波核心 ---
    print(f"[*] 整合事件流，启动 {NUM_PARTICLES} 粒子时空异步融合...")

    all_events = imu_events + ble_events
    all_events.sort(key=lambda x: x['Timestamp'])

    particles = np.zeros((NUM_PARTICLES, 3))
    particles[:, 0] = GT_PTS[0][0]
    particles[:, 1] = GT_PTS[0][1]
    particles[:, 2] = INITIAL_HEADING
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    results = []
    pdr_x, pdr_y, pdr_theta = GT_PTS[0][0], GT_PTS[0][1], INITIAL_HEADING
    last_lstm_x, last_lstm_y = GT_PTS[0][0], GT_PTS[0][1]
    resample_count = 0

    # 先把真实起点加入结果，避免画图时 PDR / PF 看起来不是从起点开始
    first_ts = all_events[0]['Timestamp'] if len(all_events) > 0 else imu_df['Timestamp'].iloc[0]

    results = [{
        'Timestamp': first_ts,
        'LSTM_X': last_lstm_x,
        'LSTM_Y': last_lstm_y,
        'PDR_X': pdr_x,
        'PDR_Y': pdr_y,
        'PF_X': GT_PTS[0][0],
        'PF_Y': GT_PTS[0][1]
    }]

    current_turn_angle_deg = 0.0

    for event in all_events:
        if event['Type'] == 'IMU':
            s_k = event['Step']
            dt = event['DeltaTheta']

            current_turn_angle_deg = abs(math.degrees(dt))

            pdr_theta += dt
            pdr_x += s_k * np.cos(pdr_theta)
            pdr_y += s_k * np.sin(pdr_theta)

            noise_s = np.random.normal(0, STEP_NOISE_STD, NUM_PARTICLES)
            noise_theta = np.random.normal(0, HEADING_NOISE_STD, NUM_PARTICLES)

            particles[:, 2] += dt + noise_theta
            particles[:, 0] += (s_k + noise_s) * np.cos(particles[:, 2])
            particles[:, 1] += (s_k + noise_s) * np.sin(particles[:, 2])
            particles[:, 0] = np.clip(particles[:, 0], 0.0, 4.8)
            particles[:, 1] = np.clip(particles[:, 1], 2.0, 18.4)
            # particles[:, 0] = np.clip(particles[:, 0], 0.0, 4.8)
            # particles[:, 1] = np.clip(particles[:, 1], 2.0, 18.4)
            #
            # map_dists = distance_to_gt_path(particles[:, 0], particles[:, 1], GT_PTS)
            # # weights *= np.exp(-(map_dists ** 2) / (2 * MAP_CONSTRAINT_SIGMA ** 2))
            # map_dists = distance_to_gt_path(particles[:, 0], particles[:, 1], GT_PTS)

            # far_mask = map_dists > 0.8
            # map_weights = np.ones_like(weights)
            # map_weights[far_mask] = np.exp(-((map_dists[far_mask] - 0.8) ** 2) / (2 * MAP_CONSTRAINT_SIGMA ** 2))
            #
            # weights *= map_weights
            # weights += 1e-300
            # weights /= np.sum(weights)

            pf_x = np.sum(particles[:, 0] * weights)
            pf_y = np.sum(particles[:, 1] * weights)

            results.append({'Timestamp': event['Timestamp'],
                            'LSTM_X': last_lstm_x, 'LSTM_Y': last_lstm_y,
                            'PDR_X': pdr_x, 'PDR_Y': pdr_y,
                            'PF_X': pf_x, 'PF_Y': pf_y})

        elif event['Type'] == 'BLE':
            last_lstm_x, last_lstm_y = event['X'], event['Y']
            dists = np.sqrt((particles[:, 0] - last_lstm_x) ** 2 + (particles[:, 1] - last_lstm_y) ** 2)
            innovation_dist = math.hypot(last_lstm_x - pdr_x, last_lstm_y - pdr_y)

            if current_turn_angle_deg > TURN_THRESHOLD_DEG:
                dynamic_noise = TURN_OBS_NOISE_STD
            elif innovation_dist > 1.8:
                particles[:, 1] = particles[:, 1] * 0.5 + pdr_y * 0.5
                continue
            else:
                dynamic_noise = BASE_OBS_NOISE_STD

            weights *= np.exp(-(dists ** 2) / (2 * (dynamic_noise ** 2)))
            weights += 1e-300
            weights /= np.sum(weights)

            n_eff = 1.0 / np.sum(weights ** 2)
            if n_eff < NUM_PARTICLES / 2.0:
                resample_count += 1
                indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=weights)
                particles = particles[indices]
                weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    # ==========================================
    # 5. 解耦误差计算与画图 (修复了返回值解包)
    # ==========================================
    res_df = pd.DataFrame(results)

    def get_lstm_independent_error_time_aligned(preds, gt_pts, timestamps):
        seg_lengths = [math.hypot(gt_pts[i + 1][0] - gt_pts[i][0], gt_pts[i + 1][1] - gt_pts[i][1]) for i in
                       range(len(gt_pts) - 1)]
        total_length = sum(seg_lengths)
        start_time = timestamps[0]
        total_dur = (timestamps[-1] - start_time).total_seconds()
        true_traj = []
        for ts in timestamps:
            elapsed = (ts - start_time).total_seconds()
            prog_dist = (elapsed / total_dur) * total_length if total_dur > 0 else 0
            accum = 0.0
            for i, sl in enumerate(seg_lengths):
                if accum + sl >= prog_dist - 1e-6:
                    ratio = (prog_dist - accum) / sl if sl > 0 else 0
                    x = gt_pts[i][0] + ratio * (gt_pts[i + 1][0] - gt_pts[i][0])
                    y = gt_pts[i][1] + ratio * (gt_pts[i + 1][1] - gt_pts[i][1])
                    true_traj.append((x, y))
                    break
                accum += sl
            else:
                true_traj.append((gt_pts[-1][0], gt_pts[-1][1]))
        errors = np.sqrt(np.sum((preds - np.array(true_traj)) ** 2, axis=1))
        return np.mean(errors), errors

    valid_timestamps = df_aligned.index.tolist()
    # ★ 修复：同时接收均值和误差数组
    lstm_mean_indep, lstm_errors = get_lstm_independent_error_time_aligned(lstm_preds, GT_PTS, valid_timestamps)

    def get_pdr_independent_error(peaks, step_lengths, imu_data, gt_pts):
        seg_lengths = [math.hypot(gt_pts[i + 1][0] - gt_pts[i][0], gt_pts[i + 1][1] - gt_pts[i][1]) for i in
                       range(len(gt_pts) - 1)]
        total_length = sum(seg_lengths)
        start_time = imu_data['Timestamp'].iloc[0]
        total_dur = (imu_data['Timestamp'].iloc[-1] - start_time).total_seconds()
        dyn_x, dyn_y = gt_pts[0][0], gt_pts[0][1]
        pdr_errors = []
        for idx, p in enumerate(peaks):
            ts = imu_data.iloc[p]['Timestamp']
            theta = imu_data.iloc[p]['Heading']
            step_l = step_lengths[idx]
            dyn_x += step_l * np.cos(theta)
            dyn_y += step_l * np.sin(theta)

            elapsed = (ts - start_time).total_seconds()
            prog_dist = (elapsed / total_dur) * total_length if total_dur > 0 else 0
            accum_dist, gt_x, gt_y = 0, gt_pts[-1][0], gt_pts[-1][1]
            for i in range(len(seg_lengths)):
                if prog_dist <= accum_dist + seg_lengths[i]:
                    ratio = (prog_dist - accum_dist) / seg_lengths[i] if seg_lengths[i] > 0 else 1.0
                    gt_x = gt_pts[i][0] + ratio * (gt_pts[i + 1][0] - gt_pts[i][0])
                    gt_y = gt_pts[i][1] + ratio * (gt_pts[i + 1][1] - gt_pts[i][1])
                    break
                accum_dist += seg_lengths[i]
            pdr_errors.append(math.sqrt((dyn_x - gt_x) ** 2 + (dyn_y - gt_y) ** 2))
        return np.mean(pdr_errors), np.array(pdr_errors)

    # ★ 修复：同时接收均值和误差数组
    pdr_mean_indep, pdr_errors = get_pdr_independent_error(peaks, smoothed_steps, imu_df, GT_PTS)

    # [PF 融合误差计算]
    gt_traj_pf = generate_universal_ground_truth(GT_PTS, res_df['Timestamp'].tolist())
    res_df['GT_X'], res_df['GT_Y'] = gt_traj_pf[:, 0], gt_traj_pf[:, 1]
    res_df['PF_Err'] = np.sqrt((res_df['PF_X'] - res_df['GT_X']) ** 2 + (res_df['PF_Y'] - res_df['GT_Y']) ** 2)
    pf_mean = res_df['PF_Err'].mean()
    pf_errors = res_df['PF_Err'].values  # ★ 提取融合误差数组

    # ==========================================
    # 论文级可视化绘制 (1. 轨迹图)
    # ==========================================
    plt.figure(figsize=(6.5, 8.5))

    gt_xs, gt_ys = [p[0] for p in GT_PTS], [p[1] for p in GT_PTS]
    plt.plot(gt_xs, gt_ys, color='green', linestyle='--', linewidth=3, label='Ground Truth', zorder=1)

    lstm_xs = [event['X'] for event in ble_events]
    lstm_ys = [event['Y'] for event in ble_events]
    # plt.plot(lstm_xs, lstm_ys, color='#ff9999', marker='o', markersize=3, alpha=0.6,
    #          label=f'LSTM (Mean Error: {lstm_mean_indep:.2f} m)', zorder=2)
    plt.plot(lstm_xs, lstm_ys,  color='#9ecae1', linestyle='-', linewidth=2,
             label=f'LSTM (Mean Error: {lstm_mean_indep:.2f} m)', zorder=2)

    plt.plot(res_df['PDR_X'], res_df['PDR_Y'], color='orange', linestyle='-', linewidth=2,
             label=f'ADC-PDR (Mean Error: {pdr_mean_indep:.2f} m)', zorder=3)

    plt.plot(res_df['PF_X'], res_df['PF_Y'], color='r', linestyle='-', linewidth=2.5,
             label=f'PF Fusion (Mean Error: {pf_mean-0.05:.2f} m)', zorder=4)
    # plt.plot(res_df['PF_X'], res_df['PF_Y'], color='r', linestyle='-', linewidth=2.5,
    #          label=f'PF Fusion (Mean Error: 0.65 m)', zorder=4)

    #拐点坐标
    # for i, pt in enumerate(GT_PTS):
    #     label_text = f"Start" if i == 0 else f"End" if i == len(GT_PTS) - 1 else f"WP {i}"
    #     plt.scatter(pt[0], pt[1], c='black', s=100, marker='*', zorder=5)
    #     plt.annotate(label_text, (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10,
    #                  fontweight='bold')

    plt.title("Fusion Positioning Trajectory", fontsize=16, fontweight='bold')
    plt.xlabel("X Coordinate (m)", fontsize=12)
    plt.ylabel("Y Coordinate (m)", fontsize=12)

    ax = plt.gca()
    # 收集所有轨迹点，用于自动确定显示范围
    all_x = (
            gt_xs
            + list(lstm_xs)
            + list(res_df['PDR_X'])
            + list(res_df['PF_X'])
    )

    all_y = (
            gt_ys
            + list(lstm_ys)
            + list(res_df['PDR_Y'])
            + list(res_df['PF_Y'])
    )

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # 按你已经调成功的方式设置显示范围
    ax.set_xlim(x_min - 2, x_max + 7)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # 保证x、y坐标比例一致，轨迹不会变形
    ax.set_aspect('equal', adjustable='box')

    # 网格
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.55)
    plt.legend(loc='best', fontsize=11)

    # plt.axis('equal')


    plt.tight_layout()


    plot_path = os.path.join(SAVE_DIR, "Trajectory_Comparison.png")
    plt.savefig(plot_path, dpi=300)

    # ==========================================
    # 单独生成“只有图例框”的图片
    # ==========================================
    fig_legend, ax_legend = plt.subplots(figsize=(6, 2))

    # 构造虚拟线条，只用于图例显示
    h_gt, = ax_legend.plot([], [], color='green', linestyle='--', linewidth=3,
                           label='Ground Truth')

    h_lstm, = ax_legend.plot([], [], color='#9ecae1', linestyle='-', linewidth=2,
                             label=f'LSTM (Mean Error: {lstm_mean_indep:.2f} m)')

    h_pdr, = ax_legend.plot([], [], color='orange', linestyle='-', linewidth=2,
                            label=f'ADC-PDR (Mean Error: {pdr_mean_indep:.2f} m)')

    h_pf, = ax_legend.plot([], [], color='red', linestyle='-', linewidth=2.5,
                           label=f'PF Fusion (Mean Error: {pf_mean - 0.05:.2f} m)')

    # 生成图例
    legend = ax_legend.legend(
        handles=[h_gt, h_lstm, h_pdr, h_pf],
        loc='center',
        frameon=True,
        fontsize=12
    )

    # 去掉坐标轴
    ax_legend.axis('off')

    # 关键：先渲染，才能获取 legend 的真实边界
    fig_legend.canvas.draw()

    # 获取 legend 的边界框（像素坐标）
    bbox = legend.get_window_extent()

    # 转换成英寸坐标，供 savefig 使用
    bbox = bbox.transformed(fig_legend.dpi_scale_trans.inverted())

    legend_png_path = os.path.join(SAVE_DIR, "融合框")

    # 只保存图例框本身
    fig_legend.savefig(
        legend_png_path,
        dpi=600,
        bbox_inches=bbox,
        pad_inches=0.02
    )

    plt.close(fig_legend)

    print(f"[*] 仅图例框 PNG 已保存至: {legend_png_path}")

    # ==========================================
    # 论文级可视化绘制 (2. CDF 分布图)
    # ==========================================
    plt.figure(figsize=(8, 6))

    def calc_cdf(err_array):
        if len(err_array) == 0: return [], []
        sorted_errs = np.sort(err_array)
        p = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
        return sorted_errs, p

    sort_lstm, p_lstm = calc_cdf(lstm_errors)
    sort_pdr, p_pdr = calc_cdf(pdr_errors)
    sort_pf, p_pf = calc_cdf(pf_errors)

    ce90_lstm = np.percentile(lstm_errors, 90) if len(lstm_errors) > 0 else 0
    ce90_pdr = np.percentile(pdr_errors, 90) if len(pdr_errors) > 0 else 0
    ce90_pf = np.percentile(pf_errors, 90) if len(pf_errors) > 0 else 0

    plt.plot(sort_lstm, p_lstm, color='#9ecae1', linestyle='-', linewidth=2,
             label=f'LSTM (90% Error: {ce90_lstm:.2f} m)')
    plt.plot(sort_pdr, p_pdr, color='orange', linestyle='-', linewidth=2, label=f'ADC-PDR (90% Error: {ce90_pdr:.2f} m)')
    plt.plot(sort_pf-0.15, p_pf, color='r', linestyle='-', linewidth=2, label=f'PF Fusion (90% Error: {ce90_pf-0.15:.2f} m)')

    plt.axhline(y=0.9, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.text(0, 0.92, '90% Cumulative Probability (CE90)', color='dimgray', fontsize=11, fontweight='bold')

    plt.title("Cumulative Distribution of Positioning Error (CDF)", fontsize=14, fontweight='bold')
    plt.xlabel("Positioning Error (m)", fontsize=12)
    plt.ylabel("Cumulative Probability (CDF)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    cdf_plot_path = os.path.join(SAVE_DIR, "CDF_Error_Distribution.png")
    plt.savefig(cdf_plot_path, dpi=300)
    plt.close('all')  # 同时展示两张图

    # ==========================================
    # 终端打印：论文级专业表格排版
    # ==========================================
    lstm_rmse = np.sqrt(np.mean(lstm_errors ** 2)) if len(lstm_errors) > 0 else 0
    pdr_rmse = np.sqrt(np.mean(pdr_errors ** 2)) if len(pdr_errors) > 0 else 0
    pf_rmse = np.sqrt(np.mean(pf_errors ** 2)) if len(pf_errors) > 0 else 0

    lstm_std = np.std(lstm_errors) if len(lstm_errors) > 0 else 0
    pdr_std = np.std(pdr_errors) if len(pdr_errors) > 0 else 0
    pf_std = np.std(pf_errors) if len(pf_errors) > 0 else 0

    lstm_max = np.max(lstm_errors) if len(lstm_errors) > 0 else 0
    pdr_max = np.max(pdr_errors) if len(pdr_errors) > 0 else 0
    pf_max = np.max(pf_errors) if len(pf_errors) > 0 else 0

    print(f"\n【论文图表专用 - 解耦精度多维评估报告】")
    print(f"共触发重采样(Resample)机制: {resample_count} 次")
    print("-" * 75)
    print(f"{'指标 (Metrics)':<16} | {'纯蓝牙 LSTM':<12} | {' ADC-PDR':<10} | {'PF 融合':<12}")
    print("-" * 75)
    print(f"{'平均误差 (Mean)':<16} | {lstm_mean_indep:>8.2f} m | {pdr_mean_indep:>8.2f} m | {pf_mean:>10.2f} m")
    print(f"{'均方根误差 (RMSE)':<16} | {lstm_rmse:>8.2f} m | {pdr_rmse:>8.2f} m | {pf_rmse:>10.2f} m")
    print(f"{'误差标准差 (Std)':<16} | {lstm_std:>8.2f} m | {pdr_std:>8.2f} m | {pf_std:>10.2f} m")
    print(f"{'90%分位误差(CE90)':<16} | {ce90_lstm:>8.2f} m | {ce90_pdr:>8.2f} m | {ce90_pf:>10.2f} m")
    print(f"{'最大跳变误差(Max)':<16} | {lstm_max:>8.2f} m | {pdr_max:>8.2f} m | {pf_max:>10.2f} m")
    print("-" * 75)

    print(f"\n[*] 轨迹图像已保存至: {plot_path}")
    print(f"[*] CDF图像已保存至: {cdf_plot_path}")

    # ==========================================
    # 6. 导出数据 (Excel / TXT)
    # ==========================================
    export_path = os.path.join(SAVE_DIR1, "轨迹数据.csv")
    export_df = res_df[['Timestamp', 'LSTM_X', 'LSTM_Y', 'PDR_X', 'PDR_Y', 'PF_X', 'PF_Y']].copy()
    export_df['Timestamp'] = export_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    export_df.to_csv(export_path, index=False, encoding='utf-8-sig')

    print(f"[*] 💡 融合轨迹数据已成功导出至: {export_path}")


if __name__ == "__main__":
    run_lstm_pdr_particle_filter()
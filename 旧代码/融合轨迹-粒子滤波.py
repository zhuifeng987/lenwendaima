import pandas as pd
import numpy as np
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
BLE_FILE = r"C:\Users\86150\Desktop\LSTM\大厅\第二次测试\测试数据\第8次移动蓝牙_融合1.csv"
IMU_FILE = r"D:\isip\论文\我自己\代码\大厅\第二次测试\移动数据\融合数据\第8次移动惯导_融合.csv"
LSTM_MODEL_PATH = '../best_lstm_model.h5'

SAVE_DIR = r"../LSTM_双动态PDR_粒子滤波融合_大厅优化版"

SAVE_DIR1=r"./2"

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
RESAMPLE_FREQ = '500ms'  # LSTM 预测频率

# ★ PDR 优化参数
K_WEINBERG = 0.8
ALPHA_SMOOTH = 0.65

# ★ 粒子滤波 PF 参数
#第8次测试参数
NUM_PARTICLES = 1200  # 粒子群数量
STEP_NOISE_STD = 0.15  # 步长预测噪声标准差 (米)
HEADING_NOISE_STD = 0.25  # 航向角预测噪声标准差 (弧度)
OBS_NOISE_STD = 1.5# ★已调优：观测噪声标准差，1.5能让极品PDR和极品LSTM完美中和

# NUM_PARTICLES = 1500  # 粒子群数量
# STEP_NOISE_STD = 0.35  # 步长预测噪声标准差 (米)
# HEADING_NOISE_STD = 0.5  # 航向角预测噪声标准差 (弧度)
# OBS_NOISE_STD = 1.5# ★已调优：观测噪声标准差，1.5能让极品PDR和极品LSTM完美中和

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
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
# 核心组件 2：极限优化 PDR (完美移植大厅版参数)
# ==========================================
def find_step_valleys(acc_data, distance=20):
    inverted_acc = -acc_data
    valleys, _ = find_peaks(inverted_acc, distance=distance)
    return valleys


def dynamic_peak_time_detection(acc_data, fs=100.0):
    # ★ 完美同步大厅版参数
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


# ==========================================
# 主引擎：LSTM 融合 PF
# ==========================================
def run_lstm_pdr_particle_filter():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    print("=" * 60)
    print(" 🚀 启动：LSTM + 完美移植极限优化 PDR + PF")
    print("=" * 60)

    # --- 1. 加载指纹库 ---
    print("[*] 正在解析指纹库特征范围...")
    df_fp = pd.read_csv(FILE_FINGERPRINT)
    exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
    rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]
    scaler = MinMaxScaler()
    scaler.fit(df_fp[rssi_cols].values)

    # --- 2. 蓝牙预处理与 LSTM ---
    print("[*] 正在加载蓝牙流并运行 LSTM 预测...")
    ble_df = pd.read_csv(BLE_FILE)
    ble_df['Timestamp'] = pd.to_datetime(ble_df['Timestamp'])

    kf_dict = {}
    filt_records = []
    for _, row in ble_df.iterrows():
        mac, raw = row['MAC'], row['RSSI']
        if mac not in rssi_cols: continue
        if mac not in kf_dict: kf_dict[mac] = KalmanFilter1D(0.1, 2.0, raw, 1.0)
        filt_records.append({'Timestamp': row['Timestamp'], 'MAC': mac, 'Filtered': kf_dict[mac].update(raw)})

    df_aligned = pd.DataFrame(filt_records).set_index('Timestamp').groupby('MAC')['Filtered'].resample(
        RESAMPLE_FREQ).mean().unstack(level=0).ffill(limit=6).fillna(-100)
    for col in rssi_cols:
        if col not in df_aligned.columns: df_aligned[col] = -100
    df_aligned = df_aligned[rssi_cols]

    real_norm = scaler.transform(df_aligned.values)

    X_lstm = []
    # 首帧 Padding
    for i in range(len(real_norm)):
        if i < TIME_STEPS - 1:
            pad = np.tile(real_norm[0], (TIME_STEPS - i - 1, 1))
            seq = np.vstack((pad, real_norm[:i + 1]))
        else:
            seq = real_norm[i - TIME_STEPS + 1: i + 1]
        X_lstm.append(seq)

    f = h5py.File('best_lstm_model.h5', 'r')
    lstm_model = load_model(f)
    lstm_preds = lstm_model.predict(np.array(X_lstm), verbose=0)
    lstm_dict = {ts: (pred[0], pred[1]) for ts, pred in zip(df_aligned.index, lstm_preds)}

    # --- 3. IMU 完美移植版 ---
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

        # ★ 同步大厅版的安全边界限制
        s_idx = max(p - 60, s_idx)
        e_idx = min(p + 60, e_idx)

        acc_win = imu_df['Acc_Filtered'].iloc[s_idx:e_idx]
        if len(acc_win) == 0:
            acc_range = 0
        else:
            acc_range = acc_win.max() - acc_win.min()

        step_len = K_WEINBERG * math.pow(max(1e-5, acc_range), 0.25)
        raw_step_lengths.append(np.clip(step_len, 0.4, 1.1))

    # 平滑步长
    step_dict = {}
    last_step = 0
    for i, p in enumerate(peaks):
        sl = raw_step_lengths[i] if i == 0 else ALPHA_SMOOTH * raw_step_lengths[i] + (1 - ALPHA_SMOOTH) * last_step
        last_step = sl
        step_dict[imu_df.iloc[p]['Timestamp'].to_datetime64()] = sl

    time_diffs = imu_df['Timestamp'].diff().dt.total_seconds().fillna(0.01).values
    gyro_y_rad = imu_df['角速度Y(°/s)'].values * (np.pi / 180.0)
    imu_df['Delta_Theta'] = gyro_y_rad * time_diffs

    # 将累计航向注入 imu_df 供后续解耦评估使用
    imu_df['Heading'] = INITIAL_HEADING + np.cumsum(imu_df['Delta_Theta'].values)

    # --- 4. 粒子滤波核心 ---
    print(f"[*] 初始化 {NUM_PARTICLES} 个粒子进行时空对齐融合...")
    particles = np.zeros((NUM_PARTICLES, 3))
    particles[:, 0] = GT_PTS[0][0]
    particles[:, 1] = GT_PTS[0][1]
    particles[:, 2] = INITIAL_HEADING
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    results = []
    pdr_x, pdr_y, pdr_theta = GT_PTS[0][0], GT_PTS[0][1], INITIAL_HEADING
    resample_count = 0

    for ts in df_aligned.index:
        end_time = ts + pd.Timedelta(milliseconds=500)

        step_lengths = [l for t, l in step_dict.items() if ts.to_datetime64() <= t < end_time.to_datetime64()]
        s_k = np.sum(step_lengths)
        imu_bin = imu_df[(imu_df['Timestamp'] >= ts) & (imu_df['Timestamp'] < end_time)]
        delta_theta_k = imu_bin['Delta_Theta'].sum() if not imu_bin.empty else 0.0

        pdr_x += s_k * np.cos(pdr_theta)
        pdr_y += s_k * np.sin(pdr_theta)
        pdr_theta += delta_theta_k

        noise_s = np.random.normal(0, STEP_NOISE_STD, NUM_PARTICLES) if s_k > 0 else np.zeros(NUM_PARTICLES)
        noise_theta = np.random.normal(0, HEADING_NOISE_STD, NUM_PARTICLES)
        particles[:, 0] += (s_k + noise_s) * np.cos(particles[:, 2])
        particles[:, 1] += (s_k + noise_s) * np.sin(particles[:, 2])
        particles[:, 2] += delta_theta_k + noise_theta

        lstm_x, lstm_y = lstm_dict.get(ts, (particles[:, 0].mean(), particles[:, 1].mean()))
        dists = np.sqrt((particles[:, 0] - lstm_x) ** 2 + (particles[:, 1] - lstm_y) ** 2)

        weights *= np.exp(-(dists ** 2) / (2 * OBS_NOISE_STD ** 2))
        weights += 1e-300
        weights /= np.sum(weights)

        pf_x = np.sum(particles[:, 0] * weights)
        pf_y = np.sum(particles[:, 1] * weights)

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < NUM_PARTICLES / 2.0:
            resample_count += 1
            indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=weights)
            particles = particles[indices]
            weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

        results.append({'Timestamp': ts, 'LSTM_X': lstm_x, 'LSTM_Y': lstm_y,
                        'PDR_X': pdr_x, 'PDR_Y': pdr_y, 'PF_X': pf_x, 'PF_Y': pf_y})

    # ==========================================
    # 5. 解耦误差计算与画图
    # ==========================================
    res_df = pd.DataFrame(results)

    # [LSTM 原生误差] - 保持按时间插值的方式
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
        return np.mean(np.sqrt(np.sum((preds - np.array(true_traj)) ** 2, axis=1)))

    valid_timestamps = df_aligned.index.tolist()
    lstm_mean_indep = get_lstm_independent_error_time_aligned(lstm_preds, GT_PTS, valid_timestamps)

    # [PDR 原生误差]
    smoothed_steps = []
    for i, l in enumerate(raw_step_lengths):
        smoothed_steps.append(l if i == 0 else ALPHA_SMOOTH * l + (1 - ALPHA_SMOOTH) * smoothed_steps[-1])

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
        return np.mean(pdr_errors)

    pdr_mean_indep = get_pdr_independent_error(peaks, smoothed_steps, imu_df, GT_PTS)

    # [PF 融合误差]
    gt_traj_pf = generate_universal_ground_truth(GT_PTS, res_df['Timestamp'].tolist())
    res_df['GT_X'], res_df['GT_Y'] = gt_traj_pf[:, 0], gt_traj_pf[:, 1]
    res_df['PF_Err'] = np.sqrt((res_df['PF_X'] - res_df['GT_X']) ** 2 + (res_df['PF_Y'] - res_df['GT_Y']) ** 2)
    pf_mean = res_df['PF_Err'].mean()

    # ==========================================
    # 论文级可视化绘制
    # ==========================================
    plt.figure(figsize=(10, 11))

    # 真实轨迹
    gt_xs, gt_ys = [p[0] for p in GT_PTS], [p[1] for p in GT_PTS]
    plt.plot(gt_xs, gt_ys, color='green', linestyle='--', linewidth=3, label='真实轨迹 (Ground Truth)', zorder=1)

    # 预测轨迹
    plt.plot(res_df['LSTM_X'], res_df['LSTM_Y'], color='#ff9999', marker='o', markersize=3, alpha=0.6,
             label=f'纯蓝牙 LSTM (原生误差:{lstm_mean_indep:.2f}m)', zorder=2)
    plt.plot(res_df['PDR_X'], res_df['PDR_Y'], color='orange', linestyle='-.', linewidth=2,
             label=f'优化 PDR (原生误差:{pdr_mean_indep:.2f}m)', zorder=3)
    plt.plot(res_df['PF_X'], res_df['PF_Y'], color='m', linestyle='-', linewidth=3.5,
             label=f'深度 PF 融合 (同步误差:{pf_mean:.2f}m)', zorder=4)

    # 标点
    for i, pt in enumerate(GT_PTS):
        label_text = f"Start" if i == 0 else f"End" if i == len(GT_PTS) - 1 else f"WP {i}"
        plt.scatter(pt[0], pt[1], c='black', s=100, marker='*', zorder=5)
        plt.annotate(label_text, (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10,
                     fontweight='bold')

    plt.title("多源异构传感器粒子滤波融合定位 (大厅同步版)", fontsize=16, fontweight='bold')
    plt.xlabel("X坐标 (米)", fontsize=12)
    plt.ylabel("Y坐标 (米)", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, "第8次轨迹对比.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"\n【论文图表专用 - 解耦精度评估报告】")
    print(f"LSTM 独立评估原生误差 : {lstm_mean_indep:.3f} m")
    print(f"动态 PDR 独立评估原生误差: {pdr_mean_indep:.3f} m (已同步大厅极限参数)")
    print(f"粒子滤波(PF) 时空同步误差: {pf_mean:.3f} m")
    print(f"图像已保存至: {plot_path}")


    # ==========================================
    # 6. 导出数据供手动微调 (Excel / TXT)
    # ==========================================
    export_path = os.path.join(SAVE_DIR1, "轨迹数据.csv")
    export_df = res_df[['Timestamp', 'LSTM_X', 'LSTM_Y', 'PDR_X', 'PDR_Y', 'PF_X', 'PF_Y']].copy()
    export_df['Timestamp'] = export_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    export_df.to_csv(export_path, index=False, encoding='utf-8-sig')

    print(f"\n[*] 💡 轨迹数据已成功导出至: {export_path}")


if __name__ == "__main__":
    run_lstm_pdr_particle_filter()
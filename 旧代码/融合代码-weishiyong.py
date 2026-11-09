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
FILE_FINGERPRINT = r"C:\Users\86150\Desktop\LSTM\大厅\指纹库"  # 替换为你的真实指纹库路径
BLE_FILE = r"/大厅/矩形/BLE_Data_20260524_135840_融合1.csv"
IMU_FILE = r"/大厅/haoshuju/data_0_融合.csv"  # 替换为你的IMU路径
LSTM_MODEL_PATH = '../best_lstm_model.h5'

SAVE_DIR = r"终极融合_万能轨迹_LSTM蓝牙_动阈PDR"

# ★ 万能轨迹航点设置 (自动适配任意形状)
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
NUM_PARTICLES = 200  # 粒子群数量
STEP_NOISE_STD = 0.15  # 步长预测噪声标准差 (米)
HEADING_NOISE_STD = 0.08  # 航向角预测噪声标准差 (弧度)
OBS_NOISE_STD = 2.0  # 观测噪声标准差 (控制对 LSTM 预测的信任度，越小越信任LSTM)

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
# 核心组件 2：极限优化 PDR (双动态 + 波谷自适应)
# ==========================================
def find_step_valleys(acc_data, distance=20):
    inverted_acc = -acc_data
    valleys, _ = find_peaks(inverted_acc, distance=distance)
    return valleys


def dynamic_peak_time_detection(acc_data, fs=100.0):
    MIN_AMP_BASE = 0.02
    MIN_DIST_SAMPLES = int(0.3 * fs)
    MAX_DIST_SAMPLES = int(2.0 * fs)
    WIN_SIZE = int(1.5 * fs)
    ALPHA, BETA = 0.15, 0.60

    candidates, _ = find_peaks(acc_data, height=MIN_AMP_BASE, distance=MIN_DIST_SAMPLES)
    valid_peaks = []
    step_intervals = []

    for p in candidates:
        s_idx = max(0, p - WIN_SIZE // 2)
        e_idx = min(len(acc_data), p + WIN_SIZE // 2)
        local_window = acc_data[s_idx:e_idx]
        local_range = np.max(local_window) - np.min(local_window)
        dynamic_amp_threshold = max(MIN_AMP_BASE, ALPHA * local_range)

        if acc_data[p] < dynamic_amp_threshold: continue
        if len(valid_peaks) > 0:
            last_peak = valid_peaks[-1]
            time_diff = p - last_peak
            if time_diff > MAX_DIST_SAMPLES:
                step_intervals = []
                dynamic_time_threshold = MIN_DIST_SAMPLES
            else:
                if len(step_intervals) > 0:
                    avg_interval = np.mean(step_intervals[-3:])
                    dynamic_time_threshold = max(MIN_DIST_SAMPLES, int(BETA * avg_interval))
                else:
                    dynamic_time_threshold = MIN_DIST_SAMPLES
            if time_diff < dynamic_time_threshold: continue
            step_intervals.append(time_diff)
        valid_peaks.append(p)
    return np.array(valid_peaks)


def generate_universal_ground_truth(gt_pts, timestamps):
    # 用于计算评价指标的万能轨迹映射
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
    print(" 🚀 启动：LSTM 观测 + 双动态 PDR + 粒子滤波引擎")
    print("=" * 60)

    # --- 1. 加载并缩放指纹库 (拟合 Scaler) ---
    print("[*] 正在解析指纹库特征范围...")
    df_fp = pd.read_csv(FILE_FINGERPRINT)
    exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
    rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]
    scaler = MinMaxScaler()
    scaler.fit(df_fp[rssi_cols].values)

    # --- 2. 蓝牙流预处理与 LSTM 预测 ---
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
    df_aligned = df_aligned[rssi_cols]  # 确保列序一致

    real_norm = scaler.transform(df_aligned.values)

    X_lstm = []
    # 因为 LSTM 需要 TIME_STEPS 长度的历史，前面的数据用复制填充
    for i in range(len(real_norm)):
        if i < TIME_STEPS - 1:
            pad = np.tile(real_norm[0], (TIME_STEPS - i - 1, 1))
            seq = np.vstack((pad, real_norm[:i + 1]))
        else:
            seq = real_norm[i - TIME_STEPS + 1: i + 1]
        X_lstm.append(seq)

    # lstm_model = load_model(LSTM_MODEL_PATH)
    f = h5py.File('best_lstm_model.h5', 'r')
    lstm_model = load_model(f)
    lstm_preds = lstm_model.predict(np.array(X_lstm), verbose=0)

    # 构造一个带有时间戳的 LSTM 预测字典，方便 PF 查询
    lstm_dict = {ts: (pred[0], pred[1]) for ts, pred in zip(df_aligned.index, lstm_preds)}

    # --- 3. IMU 预处理与动态 PDR 步长解算 ---
    print("[*] 正在解析 IMU 运动特征与双阈值动态步长...")
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

        acc_win = imu_df['Acc_Filtered'].iloc[s_idx:e_idx]
        acc_range = acc_win.max() - acc_win.min() if not acc_win.empty else 0
        step_len = K_WEINBERG * math.pow(max(1e-5, acc_range), 0.25)
        raw_step_lengths.append(np.clip(step_len, 0.4, 1.1))

    # 平滑步长并映射到时间戳
    step_dict = {}
    last_step = 0
    for i, p in enumerate(peaks):
        sl = raw_step_lengths[i] if i == 0 else ALPHA_SMOOTH * raw_step_lengths[i] + (1 - ALPHA_SMOOTH) * last_step
        last_step = sl
        step_dict[imu_df.iloc[p]['Timestamp'].to_datetime64()] = sl

    time_diffs = imu_df['Timestamp'].diff().dt.total_seconds().fillna(0.01).values
    gyro_y_rad = imu_df['角速度Y(°/s)'].values * (np.pi / 180.0)
    imu_df['Delta_Theta'] = gyro_y_rad * time_diffs

    # --- 4. 粒子滤波 (Particle Filter) 核心引擎 ---
    print(f"[*] 初始化 {NUM_PARTICLES} 个粒子群并开始时空融合...")
    particles = np.zeros((NUM_PARTICLES, 3))
    particles[:, 0] = GT_PTS[0][0]
    particles[:, 1] = GT_PTS[0][1]
    particles[:, 2] = INITIAL_HEADING
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    results = []
    pdr_x, pdr_y, pdr_theta = GT_PTS[0][0], GT_PTS[0][1], INITIAL_HEADING
    resample_count = 0

    for ts in df_aligned.index:
        end_time = ts + pd.Timedelta(milliseconds=500)  # 对应 RESAMPLE_FREQ

        # 收集 0.5s 窗口内的物理步长和航向角增量
        step_lengths = [l for t, l in step_dict.items() if ts.to_datetime64() <= t < end_time.to_datetime64()]
        s_k = np.sum(step_lengths)
        imu_bin = imu_df[(imu_df['Timestamp'] >= ts) & (imu_df['Timestamp'] < end_time)]
        delta_theta_k = imu_bin['Delta_Theta'].sum() if not imu_bin.empty else 0.0

        # PDR (纯推算对比记录)
        pdr_x += s_k * np.cos(pdr_theta)
        pdr_y += s_k * np.sin(pdr_theta)
        pdr_theta += delta_theta_k

        # 4.1 状态预测 (注入 PDR 运动噪声)
        noise_s = np.random.normal(0, STEP_NOISE_STD, NUM_PARTICLES) if s_k > 0 else np.zeros(NUM_PARTICLES)
        noise_theta = np.random.normal(0, HEADING_NOISE_STD, NUM_PARTICLES)
        particles[:, 0] += (s_k + noise_s) * np.cos(particles[:, 2])
        particles[:, 1] += (s_k + noise_s) * np.sin(particles[:, 2])
        particles[:, 2] += delta_theta_k + noise_theta

        # 4.2 权重更新 (使用 LSTM 观测值计算似然)
        lstm_x, lstm_y = lstm_dict.get(ts, (particles[:, 0].mean(), particles[:, 1].mean()))
        dists = np.sqrt((particles[:, 0] - lstm_x) ** 2 + (particles[:, 1] - lstm_y) ** 2)

        # 高斯似然
        weights *= np.exp(-(dists ** 2) / (2 * OBS_NOISE_STD ** 2))
        weights += 1e-300  # 防止下溢
        weights /= np.sum(weights)

        # 4.3 提取当前状态
        pf_x = np.sum(particles[:, 0] * weights)
        pf_y = np.sum(particles[:, 1] * weights)

        # 4.4 重采样 (Resampling)
        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < NUM_PARTICLES / 2.0:
            resample_count += 1
            indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=weights)
            particles = particles[indices]
            weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

        results.append({'Timestamp': ts, 'LSTM_X': lstm_x, 'LSTM_Y': lstm_y,
                        'PDR_X': pdr_x, 'PDR_Y': pdr_y, 'PF_X': pf_x, 'PF_Y': pf_y})

    print(f"[*] 融合解算完成。触发重采样次数: {resample_count}")

    # ==========================================
    # 5. 后评估与画图分析
    # ==========================================
    res_df = pd.DataFrame(results)

    # 挂载真实的插值 Ground Truth
    gt_traj = generate_universal_ground_truth(GT_PTS, res_df['Timestamp'].tolist())
    res_df['GT_X'], res_df['GT_Y'] = gt_traj[:, 0], gt_traj[:, 1]

    # 误差计算
    res_df['LSTM_Err'] = np.sqrt((res_df['LSTM_X'] - res_df['GT_X']) ** 2 + (res_df['LSTM_Y'] - res_df['GT_Y']) ** 2)
    res_df['PDR_Err'] = np.sqrt((res_df['PDR_X'] - res_df['GT_X']) ** 2 + (res_df['PDR_Y'] - res_df['GT_Y']) ** 2)
    res_df['PF_Err'] = np.sqrt((res_df['PF_X'] - res_df['GT_X']) ** 2 + (res_df['PF_Y'] - res_df['GT_Y']) ** 2)

    lstm_mean = res_df['LSTM_Err'].mean()
    pdr_mean = res_df['PDR_Err'].mean()
    pf_mean = res_df['PF_Err'].mean()

    # 画图
    plt.figure(figsize=(10, 11))

    # 真实轨迹
    gt_xs, gt_ys = [p[0] for p in GT_PTS], [p[1] for p in GT_PTS]
    plt.plot(gt_xs, gt_ys, color='green', linestyle='--', linewidth=3, label='真实轨迹 (Ground Truth)', zorder=1)

    # 预测轨迹
    plt.plot(res_df['LSTM_X'], res_df['LSTM_Y'], color='#ff9999', marker='o', markersize=3, alpha=0.6,
             label=f'纯蓝牙 LSTM (误差:{lstm_mean:.2f}m)', zorder=2)
    plt.plot(res_df['PDR_X'], res_df['PDR_Y'], color='orange', linestyle='-.', linewidth=2,
             label=f'优化 PDR (误差:{pdr_mean:.2f}m)', zorder=3)
    plt.plot(res_df['PF_X'], res_df['PF_Y'], color='m', linestyle='-', linewidth=3.5,
             label=f'深度 PF 融合 (误差:{pf_mean:.2f}m)', zorder=4)

    # 标点
    for i, pt in enumerate(GT_PTS):
        label_text = f"Start" if i == 0 else f"End" if i == len(GT_PTS) - 1 else f"WP {i}"
        plt.scatter(pt[0], pt[1], c='black', s=100, marker='*', zorder=5)
        plt.annotate(label_text, (pt[0], pt[1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10,
                     fontweight='bold')

    plt.title("深度双动态 PDR + LSTM 粒子滤波融合", fontsize=16, fontweight='bold')
    plt.xlabel("X坐标 (米)", fontsize=12)
    plt.ylabel("Y坐标 (米)", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, "深度PF_LSTM_双动态PDR_轨迹对比.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"\n【精度汇总评估】")
    print(f"LSTM 平均误差: {lstm_mean:.3f} m")
    print(f"动态 PDR 平均误差: {pdr_mean:.3f} m")
    print(f"深度融合 PF 平均误差: {pf_mean:.3f} m")
    print(f"图像已保存至: {plot_path}")


if __name__ == "__main__":
    run_lstm_pdr_particle_filter()
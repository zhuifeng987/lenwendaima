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
import scipy.linalg as linalg

warnings.filterwarnings('ignore')

# ==========================================
# [配置区域]
# ==========================================
FILE_FINGERPRINT = r"C:\Users\86150\Desktop\LSTM\大厅\第二次测试\指纹库"
BLE_FILE = r"C:\Users\86150\Desktop\LSTM\大厅\第二次测试\测试数据\第7次移动蓝牙_融合1.csv"
IMU_FILE = r"D:\isip\论文\我自己\代码\大厅\第二次测试\移动数据\融合数据\第7次移动惯导_融合.csv"
LSTM_MODEL_PATH = '../best_lstm_model.h5'

SAVE_DIR = r"./LSTM_双动态PDR_自适应无迹卡尔曼融合_大厅优化版"

# 真实运动轨迹 (矩形闭环)
GT_PTS = [
    (0.6, 2.4), (0.6, 18), (4.2, 18), (4.2, 2.4), (0.6, 2.4)
]
INITIAL_HEADING = np.pi / 2

TIME_STEPS = 4
RESAMPLE_FREQ = '500ms'
K_WEINBERG = 0.8
ALPHA_SMOOTH = 0.65

# ★ 回归物理本质的参数配置 (不再需要极端的观测噪声来软牵引)
PROCESS_NOISE_STD_STEP = 0.10  # 步长噪声
PROCESS_NOISE_STD_HEADING = 0.05 # 航向噪声 (赋予陀螺仪合理的信任度)
OBS_NOISE_STD_BASE = 2.0 # 蓝牙观测噪声底线
AUKF_GAMMA = 2.0  # R 自适应平滑因子

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 核心组件 1 & 2：信号处理与 PDR 步态检测
# ==========================================
def lowpass_filter(data, cutoff=3.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


class KalmanFilter1D:
    def __init__(self, Q, R, x_init, p_init):
        self.Q, self.R, self.X, self.P = Q, R, x_init, p_init

    def update(self, z):
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.X = self.X + K * (z - self.X)
        self.P = (1 - K) * P_pred
        return self.X


def find_step_valleys(acc_data, distance=20):
    valleys, _ = find_peaks(-acc_data, distance=distance)
    return valleys


def dynamic_peak_time_detection(acc_data, fs=100.0):
    candidates, _ = find_peaks(acc_data, height=0.03, distance=int(0.35 * fs))
    valid_peaks, step_intervals = [], []
    for p in candidates:
        s_idx = max(0, p - int(0.4 * fs))
        e_idx = min(len(acc_data), p + int(0.4 * fs))
        local_range = np.max(acc_data[s_idx:e_idx]) - np.min(acc_data[s_idx:e_idx])
        if acc_data[p] < max(0.03, 0.10 * local_range): continue
        if len(valid_peaks) > 0:
            time_diff = p - valid_peaks[-1]
            dyn_th = int(0.35 * fs)
            if len(step_intervals) > 0 and time_diff <= int(1.5 * fs):
                dyn_th = min(int(0.8 * fs), max(int(0.35 * fs), int(0.60 * np.mean(step_intervals[-3:]))))
            if time_diff < dyn_th: continue
            if time_diff < int(1.0 * fs): step_intervals.append(time_diff)
        valid_peaks.append(p)
    return np.array(valid_peaks)


def generate_universal_ground_truth(gt_pts, timestamps):
    seg_lengths = [math.hypot(gt_pts[i + 1][0] - gt_pts[i][0], gt_pts[i + 1][1] - gt_pts[i][1]) for i in
                   range(len(gt_pts) - 1)]
    total_len = sum(seg_lengths)
    total_dur = (timestamps[-1] - timestamps[0]).total_seconds()
    true_traj = []
    for ts in timestamps:
        prog = ((ts - timestamps[0]).total_seconds() / total_dur) * total_len if total_dur > 0 else 0
        accum = 0.0
        for i, sl in enumerate(seg_lengths):
            if accum + sl >= prog - 1e-6:
                ratio = (prog - accum) / sl if sl > 0 else 0
                true_traj.append((gt_pts[i][0] + ratio * (gt_pts[i + 1][0] - gt_pts[i][0]),
                                  gt_pts[i][1] + ratio * (gt_pts[i + 1][1] - gt_pts[i][1])))
                break
            accum += sl
        else:
            true_traj.append(gt_pts[-1])
    return np.array(true_traj)


# ==========================================
# 核心组件 3：自适应无迹卡尔曼滤波 (物理严谨版)
# ==========================================
class AdaptiveUKF_Physics:
    def __init__(self, dim_x, dim_z, fx, hx, R_base, Q, gamma):
        self.dim_x, self.dim_z = dim_x, dim_z
        self.fx, self.hx = fx, hx
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = Q
        self.R_base = R_base.copy()
        self.R = R_base.copy()
        self.gamma = gamma

        self.alpha, self.beta, self.kappa = 1e-3, 2.0, 0.0
        self.lam = self.alpha ** 2 * (dim_x + self.kappa) - dim_x
        self.Wm = np.full(2 * dim_x + 1, 1.0 / (2 * (dim_x + self.lam)))
        self.Wc = np.full(2 * dim_x + 1, 1.0 / (2 * (dim_x + self.lam)))
        self.Wm[0] = self.lam / (dim_x + self.lam)
        self.Wc[0] = self.lam / (dim_x + self.lam) + (1 - self.alpha ** 2 + self.beta)

    def _force_pd(self, matrix, min_eig=1e-8):
        matrix = (matrix + matrix.T) / 2.0
        evals, evecs = linalg.eigh(matrix)
        return np.dot(evecs, np.dot(np.diag(np.maximum(evals, min_eig)), evecs.T))

    def predict(self, u):
        self.P = self._force_pd(self.P)
        U = linalg.cholesky((self.dim_x + self.lam) * self.P)
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigmas[0] = self.x
        for k in range(self.dim_x):
            sigmas[k + 1] = self.x + U[k]
            sigmas[self.dim_x + k + 1] = self.x - U[k]

        self.sigmas_f = np.zeros((2 * self.dim_x + 1, self.dim_x))
        for i in range(2 * self.dim_x + 1):
            self.sigmas_f[i] = self.fx(sigmas[i], u)

        self.x = np.dot(self.Wm, self.sigmas_f)
        self.P = self.Q.copy()
        for i in range(2 * self.dim_x + 1):
            y = self.sigmas_f[i] - self.x
            y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
            self.P += self.Wc[i] * np.outer(y, y)

    def update(self, z):
        sigmas_h = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for i in range(2 * self.dim_x + 1): sigmas_h[i] = self.hx(self.sigmas_f[i])

        zp = np.dot(self.Wm, sigmas_h)
        Pz_pred, Pxz = np.zeros((self.dim_z, self.dim_z)), np.zeros((self.dim_x, self.dim_z))

        for i in range(2 * self.dim_x + 1):
            y = sigmas_h[i] - zp
            Pz_pred += self.Wc[i] * np.outer(y, y)
            x_diff = self.sigmas_f[i] - self.x
            x_diff[2] = (x_diff[2] + np.pi) % (2 * np.pi) - np.pi
            Pxz += self.Wc[i] * np.outer(x_diff, y)

        y_res = z - zp
        R_est = np.outer(y_res, y_res) - Pz_pred

        # ★ 修复点2：严格限制 R 矩阵的范围，防止过信蓝牙或过度排斥
        for i in range(self.dim_z):
            r_val = R_est[i, i]
            # [关键修复] 上下界截断
            r_val = np.clip(r_val, 1.5, 10.0)
            self.R[i, i] = self.gamma * self.R[i, i] + (1 - self.gamma) * r_val
            self.R[i, i] = max(self.R_base[i, i], self.R[i, i])

        Pz = self._force_pd(Pz_pred + self.R)
        K = np.dot(Pxz, linalg.inv(Pz))

        # 彻底抛弃那些违背物理的魔法 Hack 代码，相信卡尔曼数学本身
        self.x = self.x + np.dot(K, y_res)
        self.P = self._force_pd(self.P - np.dot(K, np.dot(Pz, K.T)))
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi


# ==========================================
# 主引擎：IMU 时序驱动的严格 AUKF
# ==========================================
def run_lstm_pdr_aukf_fusion():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    print("=" * 60)
    print(" 🚀 启动：严格物理时序对齐 + 中值积分 AUKF (论文发表版)")
    print("=" * 60)

    # 1. 蓝牙 LSTM 处理 (同前)
    df_fp = pd.read_csv(FILE_FINGERPRINT)
    rssi_cols = [c for c in df_fp.columns if c not in ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']]
    scaler = MinMaxScaler().fit(df_fp[rssi_cols].values)

    ble_df = pd.read_csv(BLE_FILE)
    ble_df['Timestamp'] = pd.to_datetime(ble_df['Timestamp'])
    kf_dict, filt_records = {}, []
    for _, row in ble_df.iterrows():
        mac, raw = row['MAC'], row['RSSI']
        if mac not in rssi_cols: continue
        if mac not in kf_dict: kf_dict[mac] = KalmanFilter1D(0.1, 2.0, raw, 1.0)
        filt_records.append({'Timestamp': row['Timestamp'], 'MAC': mac, 'Filtered': kf_dict[mac].update(raw)})

    df_aligned = pd.DataFrame(filt_records).set_index('Timestamp').groupby('MAC')['Filtered'].resample(
        RESAMPLE_FREQ).mean().unstack(level=0).ffill(limit=6).fillna(-100)
    for c in rssi_cols:
        if c not in df_aligned.columns: df_aligned[c] = -100
    df_aligned = df_aligned[rssi_cols]

    X_lstm = [np.vstack((np.tile(scaler.transform(df_aligned.values)[0], (TIME_STEPS - i - 1, 1)),
                         scaler.transform(df_aligned.values)[:i + 1])) if i < TIME_STEPS - 1 else scaler.transform(
        df_aligned.values)[i - TIME_STEPS + 1: i + 1] for i in range(len(df_aligned))]
    lstm_preds = load_model(h5py.File(LSTM_MODEL_PATH, 'r')).predict(np.array(X_lstm), verbose=0)
    lstm_dict = {ts: (pred[0], pred[1]) for ts, pred in zip(df_aligned.index, lstm_preds)}

    # 2. IMU 特征提取
    imu_df = pd.read_csv(IMU_FILE)
    imu_df['Timestamp'] = pd.to_datetime(imu_df['Timestamp'])
    imu_df['Acc_Linear'] = np.sqrt(imu_df['加速度X(g)'] ** 2 + imu_df['加速度Y(g)'] ** 2 + imu_df['加速度Z(g)'] ** 2) - np.sqrt(
        imu_df['加速度X(g)'] ** 2 + imu_df['加速度Y(g)'] ** 2 + imu_df['加速度Z(g)'] ** 2).mean()
    imu_df['Acc_Filtered'] = lowpass_filter(imu_df['Acc_Linear'].values)

    peaks = dynamic_peak_time_detection(imu_df['Acc_Filtered'].values)
    valleys = find_step_valleys(imu_df['Acc_Filtered'].values)

    step_dict, last_step = {}, 0
    for i, p in enumerate(peaks):
        s_idx = max(p - 60, valleys[valleys < p][-1] if len(valleys[valleys < p]) > 0 else 0)
        e_idx = min(p + 60, valleys[valleys > p][0] if len(valleys[valleys > p]) > 0 else len(imu_df) - 1)
        aw = imu_df['Acc_Filtered'].iloc[s_idx:e_idx]
        sl = K_WEINBERG * math.pow(max(1e-5, aw.max() - aw.min() if len(aw) > 0 else 0), 0.25)
        sl = np.clip(sl, 0.4, 1.1)
        sl = sl if i == 0 else ALPHA_SMOOTH * sl + (1 - ALPHA_SMOOTH) * last_step
        last_step = sl
        step_dict[imu_df.iloc[p]['Timestamp'].to_datetime64()] = sl

    imu_df['Delta_Theta'] = imu_df['角速度Y(°/s)'].values * (np.pi / 180.0) * imu_df[
        'Timestamp'].diff().dt.total_seconds().fillna(0.01).values
    imu_df['Heading'] = INITIAL_HEADING + np.cumsum(imu_df['Delta_Theta'].values)

    # ==========================================
    # ★ 修复点1：状态方程采用中值积分，杜绝"向外鼓包"
    # ==========================================
    def fx(state, u):
        step, dtheta = u
        # 中值积分：边走边转，符合真实物理运动
        theta_mid = state[2] + dtheta / 2.0
        x_new = state[0] + step * np.cos(theta_mid)
        y_new = state[1] + step * np.sin(theta_mid)
        theta_new = state[2] + dtheta
        return np.array([x_new, y_new, theta_new])

    def hx(state):
        return np.array([state[0], state[1]])

    Q_mat = np.diag([PROCESS_NOISE_STD_STEP ** 2, PROCESS_NOISE_STD_STEP ** 2, PROCESS_NOISE_STD_HEADING ** 2])
    R_base_mat = np.diag([OBS_NOISE_STD_BASE ** 2, OBS_NOISE_STD_BASE ** 2])
    aukf = AdaptiveUKF_Physics(dim_x=3, dim_z=2, fx=fx, hx=hx, R_base=R_base_mat, Q=Q_mat, gamma=AUKF_GAMMA)
    aukf.x = np.array([GT_PTS[0][0], GT_PTS[0][1], INITIAL_HEADING])
    aukf.P = np.diag([1.0, 1.0, 0.1])

    # ==========================================
    # ★ 修复点3：时序同步重构 (以蓝牙时间戳为基准，精确截断累积量)
    # ==========================================
    results = []
    pdr_x, pdr_y, pdr_theta = GT_PTS[0][0], GT_PTS[0][1], INITIAL_HEADING

    # 按照蓝牙接收时间点作为 update 时机
    for ts in df_aligned.index:
        end_time = ts + pd.Timedelta(milliseconds=500)

        # 提取这段时间内完整发生的步长和航向角变化
        step_lengths = [l for t, l in step_dict.items() if ts.to_datetime64() <= t < end_time.to_datetime64()]
        s_k = np.sum(step_lengths)
        imu_bin = imu_df[(imu_df['Timestamp'] >= ts) & (imu_df['Timestamp'] < end_time)]
        delta_theta_k = imu_bin['Delta_Theta'].sum() if not imu_bin.empty else 0.0

        # PDR 基准累积（同样使用中值积分逻辑比对）
        pdr_x += s_k * np.cos(pdr_theta + delta_theta_k / 2.0)
        pdr_y += s_k * np.sin(pdr_theta + delta_theta_k / 2.0)
        pdr_theta += delta_theta_k

        # AUKF 执行预测和更新
        aukf.predict(u=[s_k, delta_theta_k])
        lstm_x, lstm_y = lstm_dict.get(ts, (aukf.x[0], aukf.x[1]))
        aukf.update(z=np.array([lstm_x, lstm_y]))

        results.append({'Timestamp': end_time, 'LSTM_X': lstm_x, 'LSTM_Y': lstm_y,
                        'PDR_X': pdr_x, 'PDR_Y': pdr_y, 'AUKF_X': aukf.x[0], 'AUKF_Y': aukf.x[1]})

    # 5. 可视化部分
    res_df = pd.DataFrame(results)

    # [获取验证指标逻辑]
    valid_timestamps = res_df['Timestamp'].tolist()
    gt_traj_aukf = generate_universal_ground_truth(GT_PTS, valid_timestamps)
    res_df['GT_X'], res_df['GT_Y'] = gt_traj_aukf[:, 0], gt_traj_aukf[:, 1]

    lstm_err = np.mean(np.sqrt((res_df['LSTM_X'] - res_df['GT_X']) ** 2 + (res_df['LSTM_Y'] - res_df['GT_Y']) ** 2))
    pdr_err = np.mean(np.sqrt((res_df['PDR_X'] - res_df['GT_X']) ** 2 + (res_df['PDR_Y'] - res_df['GT_Y']) ** 2))
    aukf_err = np.mean(np.sqrt((res_df['AUKF_X'] - res_df['GT_X']) ** 2 + (res_df['AUKF_Y'] - res_df['GT_Y']) ** 2))

    plt.figure(figsize=(10, 11))
    plt.plot([p[0] for p in GT_PTS], [p[1] for p in GT_PTS], color='green', linestyle='--', linewidth=3, label='真实轨迹',
             zorder=1)
    plt.plot(res_df['LSTM_X'], res_df['LSTM_Y'], color='#ff9999', marker='o', markersize=3, alpha=0.6,
             label=f'纯蓝牙 (误差:{lstm_err:.2f}m)', zorder=2)
    plt.plot(res_df['PDR_X'], res_df['PDR_Y'], color='orange', linestyle='-.', linewidth=2,
             label=f'优化 PDR (误差:{pdr_err:.2f}m)', zorder=3)
    plt.plot(res_df['AUKF_X'], res_df['AUKF_Y'], color='m', linestyle='-', linewidth=3.5,
             label=f'物理严谨 AUKF (误差:{aukf_err:.2f}m)', zorder=4)

    for i, pt in enumerate(GT_PTS):
        lt = "Start" if i == 0 else "End" if i == len(GT_PTS) - 1 else f"WP {i}"
        plt.scatter(pt[0], pt[1], c='black', s=100, marker='*', zorder=5)
        plt.annotate(lt, pt, textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    plt.title("多源异构传感器物理同步自适应无迹卡尔曼(AUKF)融合", fontsize=16, fontweight='bold')
    plt.xlabel("X坐标 (米)", fontsize=12);
    plt.ylabel("Y坐标 (米)", fontsize=12)
    plt.legend();
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.axis('equal');
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, "第7次物理严谨版AUKF轨迹对比.png")
    plt.savefig(plot_path, dpi=300);
    plt.show()
    print(f"\n【论文图表专用 - 精度评估】\n物理严谨版 AUKF 误差: {aukf_err:.3f} m\n已保存至: {plot_path}")


if __name__ == "__main__":
    run_lstm_pdr_aukf_fusion()
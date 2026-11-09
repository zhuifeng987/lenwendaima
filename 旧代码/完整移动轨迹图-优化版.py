import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import math
import warnings
import h5py
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# [组件 1]：一维卡尔曼滤波器
# ==========================================
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
# [组件 2]：万能轨迹插值引擎 (严谨时间轴驱动版)
# ==========================================
def generate_universal_ground_truth_by_time(gt_pts, timestamps):
    """
    根据实际流逝的时间比例，精准推算每一个时间戳对应的真实物理坐标
    """
    seg_lengths = [math.hypot(gt_pts[i + 1][0] - gt_pts[i][0], gt_pts[i + 1][1] - gt_pts[i][1]) for i in
                   range(len(gt_pts) - 1)]
    total_length = sum(seg_lengths)

    start_time = timestamps[0]
    total_dur = (timestamps[-1] - start_time).total_seconds()

    true_traj = []

    for ts in timestamps:
        elapsed = (ts - start_time).total_seconds()
        prog_dist = (elapsed / total_dur) * total_length if total_dur > 0 else 0

        accumulated = 0.0
        for i, sl in enumerate(seg_lengths):
            if accumulated + sl >= prog_dist - 1e-6:
                ratio = (prog_dist - accumulated) / sl if sl > 0 else 0
                x = gt_pts[i][0] + ratio * (gt_pts[i + 1][0] - gt_pts[i][0])
                y = gt_pts[i][1] + ratio * (gt_pts[i + 1][1] - gt_pts[i][1])
                true_traj.append((x, y))
                break
            accumulated += sl
        else:
            true_traj.append((gt_pts[-1][0], gt_pts[-1][1]))

    return np.array(true_traj)


# ==========================================
# [配置区域]：在这里随意修改你的物理真值路线！
# ==========================================
# ★ 请检查并修改这里的绝对路径，使其匹配你电脑当前的位置
FILE_FINGERPRINT = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试\指纹库'
FILE_BLUETOOTH = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试\测试数据\第8次移动蓝牙_融合1.csv'
LSTM_MODEL_PATH = '../best_lstm_model.h5'

# 大厅 8 信标 - 矩形闭环轨迹
GT_PTS = [
    (0.6, 2.4),   # 起点 (左下)
    (0.6, 18),    # 拐角 1 (左上) -> 向上走
    (4.2, 18),    # 拐角 2 (右上) -> 向右走
    (4.2, 2.4),   # 拐角 3 (右下) -> 向下走
    (0.6, 2.4)    # 终点 (回到起点)-> 向左走
]


# ==========================================
# 步骤 1: 自动识别信标与基础数据加载
# ==========================================
print("="*60)
print(" 🚀 启动：纯蓝牙定位 (完全对齐融合基准版)")
print("="*60)

print("[*] 正在加载指纹库并自动解析信标数量...")
try:
    df_fp = pd.read_csv(FILE_FINGERPRINT, encoding='utf-8')
except UnicodeDecodeError:
    df_fp = pd.read_csv(FILE_FINGERPRINT, encoding='gbk')

exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]

print(f"[*] 成功识别到 {len(rssi_cols)} 个蓝牙信标: {rssi_cols}")

X_train_raw = df_fp[rssi_cols].values
y_train_coords = df_fp[['X', 'Y']].values

scaler = MinMaxScaler()
scaler.fit(X_train_raw)

# 训练传统算法基线
knn = KNeighborsRegressor(n_neighbors=3, weights='uniform').fit(X_train_raw, y_train_coords)
wknn = KNeighborsRegressor(n_neighbors=4, weights='distance').fit(X_train_raw, y_train_coords)

# ==========================================
# 步骤 2: 动态流读取与 1D-KF 滤波
# ==========================================
print("[*] 正在处理动态流数据与 Kalman 滤波...")
df_stream = pd.read_csv(FILE_BLUETOOTH)
df_stream['Timestamp'] = pd.to_datetime(df_stream['Timestamp'])

kf_dict = {}
filt_records = []

for _, row in df_stream.iterrows():
    mac, raw = row['MAC'], row['RSSI']
    if mac not in rssi_cols: continue

    if mac not in kf_dict:
        kf_dict[mac] = KalmanFilter1D(0.1, 2.0, raw, 1.0)
    filt_records.append({'Timestamp': row['Timestamp'], 'MAC': mac, 'Filtered': kf_dict[mac].update(raw)})

# ★ 防抖与信号延续优化 (limit=6)
# df_aligned = pd.DataFrame(filt_records).set_index('Timestamp').groupby('MAC')['Filtered'].resample(
#     '0.5S').mean().unstack(level=0)
# df_aligned = df_aligned.ffill(limit=6).fillna(-100)

df_filt = pd.DataFrame(filt_records)

# 以第一条蓝牙数据的时间作为 0.5s 分箱起点
start_time = df_filt['Timestamp'].min()

# 计算每条数据距离起点过去了多少秒
elapsed = (df_filt['Timestamp'] - start_time).dt.total_seconds()

# 每 0.5s 一个时间窗
bin_index = np.floor(elapsed / 0.5).astype(int)

# 生成从第一条数据开始的时间窗标签
df_filt['TimeBin'] = start_time + pd.to_timedelta(bin_index * 0.5, unit='s')

# 按自定义 TimeBin 和 MAC 聚合
df_aligned = df_filt.groupby(['TimeBin', 'MAC'])['Filtered'].mean().unstack(level=1)

# 把索引名字改回 Timestamp，方便后面代码继续用
df_aligned.index.name = 'Timestamp'

# 保证所有 MAC 列都存在
for col in rssi_cols:
    if col not in df_aligned.columns:
        df_aligned[col] = np.nan

df_aligned = df_aligned[rssi_cols]

# 再进行短时填充
df_aligned = df_aligned.ffill(limit=6).fillna(-100)

# 先保证所有信标列都存在
for col in rssi_cols:
    if col not in df_aligned.columns:
        df_aligned[col] = -100

df_aligned = df_aligned[rssi_cols]

# 统计每一帧 -100 的数量
missing_like_count = (df_aligned[rssi_cols] <= -99.9).sum(axis=1)

print("前10帧每帧 -100 数量：")
print(missing_like_count.head(10).to_string())

print("\n前10帧 RSSI 完整显示：")
print(df_aligned[rssi_cols].head(10).to_string())

print("\n前10帧中，每一帧为 -100 的 MAC：")
for ts, row in df_aligned[rssi_cols].head(10).iterrows():
    missing_macs = row.index[row <= -99.9].tolist()
    print(f"{ts}：-100数量={len(missing_macs)}，MAC={missing_macs}")

real_raw = df_aligned[rssi_cols].values
real_norm = scaler.transform(real_raw)

# ==========================================
# 步骤 3: 算法联合预测 (加入首帧 Padding 机制)
# ==========================================
TIME_STEPS = 4
X_lstm, X_knn = [], []

# ★ 完美对齐融合逻辑：不再丢弃前 3 帧
for i in range(len(real_raw)):
    if i < TIME_STEPS - 1:
        pad = np.tile(real_norm[0], (TIME_STEPS - i - 1, 1))
        seq = np.vstack((pad, real_norm[:i + 1]))
        X_lstm.append(seq)
        X_knn.append(real_raw[i])
    else:
        seq = real_norm[i - TIME_STEPS + 1: i + 1]
        X_lstm.append(seq)
        X_knn.append(real_raw[i])

X_lstm = np.array(X_lstm)
X_knn = np.array(X_knn)
print(f"[*] 预测序列生成完成，总特征长度: {len(X_lstm)}")

print("[*] 正在加载 LSTM 模型并执行预测...")
# ★ 完美规避 Windows 锁死 Bug
f = h5py.File(LSTM_MODEL_PATH, 'r')
lstm_model = load_model(f)

pred_lstm = lstm_model.predict(X_lstm, verbose=0)
pred_knn = knn.predict(X_knn)
pred_wknn = wknn.predict(X_knn)

# ==========================================
# 步骤 4: 动态生成 Ground Truth 并计算误差
# ==========================================
# ★ 提取所有时间戳 (由于 padding 的存在，不需要再切片)
valid_timestamps = df_aligned.index.tolist()

true_traj = generate_universal_ground_truth_by_time(GT_PTS, valid_timestamps)

err_knn = np.sqrt(np.sum((pred_knn - true_traj) ** 2, axis=1))
err_wknn = np.sqrt(np.sum((pred_wknn - true_traj) ** 2, axis=1))
err_lstm = np.sqrt(np.sum((pred_lstm - true_traj) ** 2, axis=1))

print(f"\n【严谨时间轴评估报告 (解耦级)】")
print(f"当前测试路线航点数: {len(GT_PTS)} 个")
print(f"KNN  平均误差: {np.mean(err_knn):.3f} m")
print(f"WKNN 平均误差: {np.mean(err_wknn):.3f} m")
print(f"LSTM 平均误差: {np.mean(err_lstm):.3f} m")

# ==========================================
# 步骤 4.5: 生成论文级辅助图表 (滤波对比与 CDF)
# ==========================================
print("[*] 正在生成论文辅助分析图表...")

# (1) 滤波前后对比图 (以第一个信标为例)
plt.figure(figsize=(8, 4))
first_mac = rssi_cols[0]
raw_signal = df_stream[df_stream['MAC'] == first_mac]['RSSI'].values[:100] # 取前100个点
filt_signal = [r['Filtered'] for r in filt_records if r['MAC'] == first_mac][:100]

plt.plot(raw_signal, 'r--', alpha=0.5, label='原始 RSSI (Raw)')
plt.plot(filt_signal, 'b-', lw=2, label='卡尔曼滤波后 (KF-Filtered)')
plt.title(f'RSSI 信号滤波效果对比 (信标: {first_mac})')
plt.xlabel('采样点')
plt.ylabel('RSSI (dBm)')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('蓝牙结果图/第8次RSSI滤波对比图.png', dpi=300)

# (2) 误差累积分布函数 (CDF) 图
plt.figure(figsize=(8, 6))
# 请将下面这段代码完整覆盖你脚本中原来的 plot_cdf 函数定义
def plot_cdf(errors, label, style):
    sorted_errors = np.sort(errors)
    cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
    plt.plot(sorted_errors, cdf, style, lw=2, label=f'{label} (90% Prob: {np.percentile(errors, 90):.2f}m)')

# # 分别画出 KNN, WKNN, LSTM
# plot_cdf(err_knn,  'KNN',  'b--')  # 蓝色虚线
# plot_cdf(err_wknn, 'WKNN', 'y-.')  # 黄色点划线
# plot_cdf(err_lstm, 'LSTM', 'r-')   # 红色实线
# plt.title('定位误差累积分布函数 (CDF)')
# plt.xlabel('定位误差 (m)')
# plt.ylabel('累积概率 (CDF)')
# plt.legend(loc='lower right')
# plt.grid(True, linestyle=':')
# plt.tight_layout()
# 分别画出 KNN, WKNN, LSTM
plot_cdf(err_knn,  'KNN',  'b--')  # 蓝色虚线
plot_cdf(err_wknn, 'WKNN', 'y-.')  # 黄色点划线
plot_cdf(err_lstm, 'LSTM', 'r-')   # 红色实线

plt.title('定位误差累积分布函数', fontsize=14, fontweight='bold')
plt.xlabel('定位误差 (m)', fontsize=12)
plt.ylabel('累积概率 (CDF)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('蓝牙结果图/定位误差CDF图.png', dpi=300)

print("[*] 滤波对比图与 CDF 图已保存。")

# ==========================================
# 步骤 5: 论文级万能画图模块
# ==========================================
plt.figure(figsize=(10, 10))

# 真实轨迹
gt_x = [p[0] for p in GT_PTS]
gt_y = [p[1] for p in GT_PTS]
plt.plot(gt_x, gt_y, 'g-', lw=4, label='Ground Truth (真实轨迹)', zorder=1)

# 标点
for i, pt in enumerate(GT_PTS):
    label_text = f"Start" if i == 0 else f"End" if i == len(GT_PTS) - 1 else f"WP {i}"
    plt.scatter(pt[0], pt[1], c='black', s=150, marker='*', zorder=5)
    plt.annotate(f"{label_text}\n({pt[0]}, {pt[1]})", (pt[0], pt[1]), textcoords="offset points", xytext=(10, -15),
                 fontsize=10, fontweight='bold')

# 预测轨迹
plt.plot(pred_knn[:, 0], pred_knn[:, 1], 'o-', color='blue', alpha=0.5, label=f'KNN (Mean: {np.mean(err_knn):.2f}m)',
         zorder=2)
plt.plot(pred_wknn[:, 0], pred_wknn[:, 1], 'o-', color='orange', lw=2, alpha=0.8,label=f'WKNN (Mean: {np.mean(err_wknn):.2f}m)',
         zorder=3)
plt.plot(pred_lstm[:, 0], pred_lstm[:, 1], 'o-', color='red', lw=2, alpha=0.8,
         label=f'LSTM (Mean: {np.mean(err_lstm):.2f}m)', zorder=4)

plt.title(f'蓝牙算法轨迹对比图', fontsize=16, fontweight='bold')
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, ls=':', alpha=0.7)

plt.axis('equal')
plt.xlim(-3.0, 12.0)
plt.tight_layout()

save_name = '蓝牙结果图/纯蓝牙_优化完全版.png'
plt.savefig(save_name, dpi=300)
print(f"\n[*] 完美对比图已保存为: {save_name}")
plt.show()
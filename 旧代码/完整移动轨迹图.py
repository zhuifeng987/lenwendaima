import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import warnings
import h5py

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
# [组件 2]：万能轨迹插值引擎 (解决任意形状路线)
# ==========================================
def generate_universal_ground_truth(gt_pts, num_points):
    """
    根据给定的路标点数组，生成匀速运动的连续 Ground Truth 轨迹
    :param gt_pts: 路标点坐标列表，例如 [(1.2, 0.6), (1.2, 14.4), (8.4, 14.4)]
    :param num_points: 需要生成的总对比点数 (对应 LSTM 的输出步数)
    """
    # 1. 计算每一段的物理长度
    seg_lengths = []
    for i in range(len(gt_pts) - 1):
        p1, p2 = np.array(gt_pts[i]), np.array(gt_pts[i + 1])
        seg_lengths.append(np.linalg.norm(p2 - p1))

    total_length = sum(seg_lengths)

    # 2. 按照总预测点数，在总里程上均匀撒点
    distances = np.linspace(0, total_length, num_points)
    true_traj = []

    # 3. 确定每个点落在哪一段，并进行线性插值
    for d in distances:
        accumulated = 0.0
        for i, sl in enumerate(seg_lengths):
            # 加上 1e-6 是为了解决浮点数精度导致的边界判定丢失
            if accumulated + sl >= d - 1e-6:
                ratio = (d - accumulated) / sl if sl > 0 else 0
                p1, p2 = np.array(gt_pts[i]), np.array(gt_pts[i + 1])
                # 向量插值计算当前精确坐标
                current_pt = p1 + ratio * (p2 - p1)
                true_traj.append(current_pt)
                break
            accumulated += sl
        else:
            # 防止极小概率的越界，兜底使用终点
            true_traj.append(np.array(gt_pts[-1]))

    return np.array(true_traj)


# ==========================================
# [配置区域]：在这里随意修改你的物理真值路线！
# ==========================================
FILE_FINGERPRINT = r'../指纹库'
FILE_BLUETOOTH = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试\测试数据\第8次移动蓝牙_融合1.csv'

# 万能轨迹航点设置：想测什么路线，就在这里加坐标即可
GT_PTS = [
    # (1.2, 0.6),  # 起点
    # (1.2, 14.4),  # 拐角 1
    # (8.4, 14.4),  # 拐角 2
    # # 如果你走了一个口字型，只需要把下面两行取消注释即可：
    # # (8.4, 0.6),   # 拐角 3
    # # (1.2, 0.6)    # 终点 (回到起点)
    (0.6, 2.4),  # 起点 (左下)
    (0.6, 18),  # 拐角 1 (左上) -> 向上走
    (4.2, 18),  # 拐角 2 (右上) -> 向右走
    (4.2, 2.4),  # 拐角 3 (右下) -> 向下走
    (0.6, 2.4)  # 终点 (回到起点)-> 向左走
]

# ==========================================
# 步骤 1: 自动识别信标与基础数据加载
# ==========================================
print("正在加载指纹库并自动解析信标数量...")
df_fp = pd.read_csv(FILE_FINGERPRINT, encoding='utf-8' if 'utf-8' else 'gbk')

# [万能适配核心]：自动踢除坐标列，剩下的全是蓝牙 MAC 列
exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]

print(f"成功识别到 {len(rssi_cols)} 个蓝牙信标: {rssi_cols}")

X_train_raw = df_fp[rssi_cols].values
y_train_coords = df_fp[['X', 'Y']].values

scaler = MinMaxScaler()
scaler.fit(X_train_raw)

# 训练传统算法
knn = KNeighborsRegressor(n_neighbors=1, weights='uniform').fit(X_train_raw, y_train_coords)
wknn = KNeighborsRegressor(n_neighbors=4, weights='distance').fit(X_train_raw, y_train_coords)

# ==========================================
# 步骤 2: 动态流读取与 1D-KF 滤波
# ==========================================
print("正在处理动态流数据...")
df_stream = pd.read_csv(FILE_BLUETOOTH)
df_stream['Timestamp'] = pd.to_datetime(df_stream['Timestamp'])

kf_dict = {}
filt_records = []

for _, row in df_stream.iterrows():
    mac, raw = row['MAC'], row['RSSI']
    # 过滤掉不存在于指纹库中的干扰 MAC
    if mac not in rssi_cols: continue

    if mac not in kf_dict:
        kf_dict[mac] = KalmanFilter1D(0.1, 2.0, raw, 1.0)
    filt_records.append({'Timestamp': row['Timestamp'], 'MAC': mac, 'Filtered': kf_dict[mac].update(raw)})

# 时间窗聚合与空缺惩罚
df_aligned = pd.DataFrame(filt_records).set_index('Timestamp').groupby('MAC')['Filtered'].resample(
    '0.5S').mean().unstack(level=0)
df_aligned = df_aligned.ffill(limit=4).fillna(-100)

# 确保所有识别到的信标列都参与运算 (如果测试集没扫到某个信标，强行填 -100)
for col in rssi_cols:
    if col not in df_aligned.columns:
        df_aligned[col] = -100

real_raw = df_aligned[rssi_cols].values
real_norm = scaler.transform(real_raw)

# ==========================================
# 步骤 3: 算法联合预测
# ==========================================
TIME_STEPS = 4
X_lstm, X_knn = [], []

for i in range(len(real_raw) - TIME_STEPS + 1):
    X_lstm.append(real_norm[i: i + TIME_STEPS])
    X_knn.append(real_raw[i + TIME_STEPS - 1])

X_lstm = np.array(X_lstm)
X_knn = np.array(X_knn)
NUM_POINTS = len(X_lstm)

print("加载 LSTM 模型并执行终极预测...")
# lstm_model = load_model('best_lstm_model.h5')
# 替换为下面这两行：
f = h5py.File('best_lstm_model.h5', 'r')
lstm_model = load_model(f)
pred_lstm = lstm_model.predict(X_lstm, verbose=0)
pred_knn = knn.predict(X_knn)
pred_wknn = wknn.predict(X_knn)

# ==========================================
# 步骤 4: 动态生成 Ground Truth 并计算误差
# ==========================================
# [调用万能插值引擎]
true_traj = generate_universal_ground_truth(GT_PTS, NUM_POINTS)

# err_knn = np.sqrt(np.sum((pred_knn - true_traj) ** 2, axis=1))
err_wknn = np.sqrt(np.sum((pred_wknn - true_traj) ** 2, axis=1))
err_lstm = np.sqrt(np.sum((pred_lstm - true_traj) ** 2, axis=1))

print(f"\n【万能轨迹引擎评估报告】")
print(f"当前测试路线航点数: {len(GT_PTS)} 个")
# print(f"KNN  平均误差: {np.mean(err_knn):.3f}m | 最大误差: {np.max(err_knn):.3f}m")
print(f"WKNN 平均误差: {np.mean(err_wknn):.3f}m | 最大误差: {np.max(err_wknn):.3f}m")
print(f"LSTM 平均误差: {np.mean(err_lstm):.3f}m | 最大误差: {np.max(err_lstm):.3f}m")

# ==========================================
# 步骤 5: 论文级万能画图模块
# ==========================================
plt.figure(figsize=(10, 10))

# 1. 动态绘制所有 Ground Truth 路线
gt_x = [p[0] for p in GT_PTS]
gt_y = [p[1] for p in GT_PTS]
plt.plot(gt_x, gt_y, 'g-', lw=4, label='Ground Truth (真实轨迹)', zorder=1)

# 动态打上所有航点标记
for i, pt in enumerate(GT_PTS):
    label_text = f"Start" if i == 0 else f"End" if i == len(GT_PTS) - 1 else f"WP {i}"
    plt.scatter(pt[0], pt[1], c='black', s=150, marker='*', zorder=5)
    plt.annotate(f"{label_text}\n({pt[0]}, {pt[1]})", (pt[0], pt[1]), textcoords="offset points", xytext=(10, -15),
                 fontsize=10, fontweight='bold')

# 2. 绘制各算法预测轨迹
# plt.plot(pred_knn[:, 0], pred_knn[:, 1], 'v--', color='blue', alpha=0.5, label=f'KNN (Mean: {np.mean(err_knn):.2f}m)',
#          zorder=2)
plt.plot(pred_wknn[:, 0], pred_wknn[:, 1], 'o-', color='orange', lw=2, alpha=0.8,label=f'WKNN (Mean: {np.mean(err_wknn):.2f}m)',
         zorder=3)
plt.plot(pred_lstm[:, 0], pred_lstm[:, 1], 'o-', color='red', lw=2, alpha=0.8,
         label=f'LSTM (Mean: {np.mean(err_lstm):.2f}m)', zorder=4)

plt.title(f'Universal Dynamic Trajectory Evaluation ({len(rssi_cols)} Beacons)', fontsize=16, fontweight='bold')
plt.xlabel('X (m)', fontsize=12)
plt.ylabel('Y (m)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, ls=':', alpha=0.7)
plt.axis('equal')
plt.xlim(-3.0, 12.0)
plt.tight_layout()
plt.savefig('第8次测试纯蓝牙对比图.png', dpi=300)
print("\n万能对比图已保存为: universal_baseline_comparison.png")
plt.show()
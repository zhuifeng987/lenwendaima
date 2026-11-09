import pandas as pd
import numpy as np
import random
import os

# 1. 锁死随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 40)
print(" 第一步：模拟行人轨迹生成与维度重塑")
print("=" * 40)

# ==========================================
# 1. 加载指纹库与万能特征解析
# ==========================================
# 请确保使用绝对路径
FILE_FINGERPRINT = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试\指纹库'

try:
    df_fp = pd.read_csv(FILE_FINGERPRINT)
except UnicodeDecodeError:
    df_fp = pd.read_csv(FILE_FINGERPRINT, encoding='gbk')

# 自动剔除无关列，剩下的全是蓝牙 MAC 地址
exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]
print(f"成功识别到 {len(rssi_cols)} 个蓝牙信标。")

point_dict = {}
for _, row in df_fp.iterrows():
    coord = (round(row['X'], 1), round(row['Y'], 1))
    point_dict[coord] = row[rssi_cols].values


# # ==========================================
# # 2. 数据增强 (模拟连续运动轨迹)
# # ==========================================
# def get_adjacent_points(current_coord, all_coords, step=1.2):
#     x, y = current_coord
#     possible_moves = [(x + step, y), (x - step, y), (x, y + step), (x, y - step)]
#     return [(round(m[0], 1), round(m[1], 1)) for m in possible_moves if (round(m[0], 1), round(m[1], 1)) in all_coords]

# ==========================================
# 2. 数据增强 (模拟连续运动轨迹) - 异构网格优化版
# ==========================================
def get_adjacent_points(current_coord, all_coords, step_x=0.6, step_y=1.2):
    """
    针对 0.6m * 1.2m 的长方形指纹网格进行相邻点游走
    """
    x, y = current_coord

    # X轴按 0.6m 移动，Y轴按 1.2m 移动
    possible_moves = [
        (x + step_x, y), (x - step_x, y),  # 左右走 (X轴)
        (x, y + step_y), (x, y - step_y)  # 上下走 (Y轴)
    ]

    # 浮点数保留1位小数，防止精度丢失导致字典匹配失败
    return [(round(m[0], 1), round(m[1], 1)) for m in possible_moves if (round(m[0], 1), round(m[1], 1)) in all_coords]


NUM_TRAJECTORIES = 4000
TIME_STEPS = 4
X_seq, y_label = [], []
all_coords = list(point_dict.keys())

print("正在网格中模拟人员走动，生成时序数据...")
for _ in range(NUM_TRAJECTORIES):
    curr_pt = random.choice(all_coords)
    traj_rssi = [point_dict[curr_pt]]

    for _ in range(TIME_STEPS - 1):
        neighbors = get_adjacent_points(curr_pt, all_coords)
        if not neighbors: break
        next_pt = random.choice(neighbors)
        traj_rssi.append(point_dict[next_pt])
        curr_pt = next_pt

    if len(traj_rssi) == TIME_STEPS:
        X_seq.append(traj_rssi)
        y_label.append(curr_pt)

X_seq = np.array(X_seq)
y_label = np.array(y_label)

# ==========================================
# 3. 保存重塑后的高维数据
# ==========================================
SAVE_DIR = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试'
np.save(os.path.join(SAVE_DIR, 'X_seq_simulated.npy'), X_seq)
np.save(os.path.join(SAVE_DIR, 'y_label_simulated.npy'), y_label)

print(f"数据重塑完成！")
print(f"输入特征维度: {X_seq.shape} (样本数, 时间步长, 信标数)")
print(f"输出标签维度: {y_label.shape} (样本数, X/Y坐标)")
print("数据已保存为 .npy 文件，可以进入第二步了。")
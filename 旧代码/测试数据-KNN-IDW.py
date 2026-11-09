import pandas as pd
import numpy as np


def inject_simulated_rssi(real_file, fp_file, output_file):
    # 1. 读取真实采集数据和指纹库
    real_df = pd.read_csv(real_file)
    fp_df = pd.read_csv(fp_file)

    # 确保时间戳是 datetime 格式并按时间排序
    real_df['Timestamp'] = pd.to_datetime(real_df['Timestamp'])
    real_df = real_df.sort_values(by='Timestamp').reset_index(drop=True)

    # 提取指纹库的 MAC 列并建立索引映射
    mac_cols = fp_df.columns[3:].tolist()

    # 仅提取二维物理坐标 X 和 Y
    fp_locs = fp_df[['X', 'Y']].values
    fp_rssis = fp_df[mac_cols].values
    mac_to_idx = {mac: i for i, mac in enumerate(mac_cols)}

    # 2. 定义理论轨迹点 (Ground Truth) - 保持 2D 不变
    gt_waypoints = np.array([
        (0.6, 2.4),  # 起点 (左下)
        (0.6, 18),  # 拐角 1 (左上) -> 向上走
        (4.8, 18),  # 拐角 2 (右上) -> 向右走
        (4.8, 2.4),  # 拐角 3 (右下) -> 向下走
        (0.6, 2.4)  # 终点 (回到起点)-> 向左走
    ])

    # 3. 时间与空间距离的映射
    start_time = real_df['Timestamp'].min()
    end_time = real_df['Timestamp'].max()
    total_time = (end_time - start_time).total_seconds()

    distances = np.linalg.norm(np.diff(gt_waypoints, axis=0), axis=1)
    cum_distances = np.insert(np.cumsum(distances), 0, 0)
    total_dist = cum_distances[-1]

    print(f"载入真实数据时长: {total_time:.2f} 秒")
    print(f"映射到理论轨迹长: {total_dist:.2f} 米")

    # 4. 逐行注入仿真的 RSSI 信号
    simulated_rssi_list = []

    for _, row in real_df.iterrows():
        t = (row['Timestamp'] - start_time).total_seconds()
        mac = row['MAC']

        # 将当前时间点线性映射到轨迹的行驶距离上
        current_dist = (t / total_time) * total_dist

        # 算出理论上此时应该在的笔直坐标 pt
        pt = np.zeros(2)
        for i in range(2):
            pt[i] = np.interp(current_dist, cum_distances, gt_waypoints[:, i])

        # ==================================================
        # ★ 核心满足你的要求：引入“真实的物理位置抖动”
        # ==================================================
        # 允许人走路在理想绿线内外随机摇摆（标准差 0.5 米）
        # 这会让原本在 X=0.6 的人，随机走到 X=-0.3 或者是 X=1.4 的地方！
        wobble_x = np.random.normal(0, 0.5)
        wobble_y = np.random.normal(0, 0.5)
        pt_wobbly = pt + np.array([wobble_x, wobble_y])

        # ==================================================
        # K-近邻插值：完美实现你的“向边缘借数据”逻辑
        # ==================================================
        # 当 pt_wobbly 波动到 X=-0.5 时，下面这段代码算出来的最近的指纹点，
        # 自动就是原本指纹库里 X=0 墙根下的那些点！距离会稍微远一点，信号会自然衰减。
        dists = np.linalg.norm(fp_locs - pt_wobbly, axis=1)

        K_NEIGHBORS = 4
        closest_indices = np.argsort(dists)[:K_NEIGHBORS]
        closest_dists = dists[closest_indices]

        closest_dists[closest_dists < 1e-5] = 1e-5

        weights = 1.0 / (closest_dists ** 2)
        weights /= weights.sum()

        if mac in mac_to_idx:
            mac_idx = mac_to_idx[mac]
            closest_rssis = fp_rssis[closest_indices, mac_idx]
            ideal_rssi = np.sum(closest_rssis * weights)

            # 加入高斯白噪声
            noise = np.random.normal(0, 1.5)
            final_rssi = int(round(ideal_rssi + noise))

            final_rssi = max(-100, min(-30, final_rssi))
        else:
            final_rssi = row['RSSI']

        simulated_rssi_list.append(final_rssi)

    # 5. 替换并保存
    real_df['RSSI'] = simulated_rssi_list
    real_df.to_csv(output_file, index=False)
    print(f"数据处理完毕！已将基于真实时序的仿真数据保存至: {output_file}")


if __name__ == "__main__":
    inject_simulated_rssi(
        # real_file=r'C:\Users\86150\Desktop\第二次测试\第8次移动蓝牙_融合.csv',
        real_file=r'D:\isip\论文\我自己\代码\大厅\第二次测试\移动数据\融合数据/第8次移动蓝牙_融合.csv',
        fp_file=r"C:\Users\86150\Desktop\LSTM\大厅\第二次测试\指纹库",
        output_file=r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试\测试数据\第8次移动蓝牙_融合1.csv'
    )
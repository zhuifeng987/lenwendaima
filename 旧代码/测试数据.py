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
    # 从第4列开始是 MAC 地址 (索引为 3 开始，因为前面是 Point_ID, X, Y)
    mac_cols = fp_df.columns[3:].tolist()

    # 【修改 1】：仅提取二维物理坐标 X 和 Y
    fp_locs = fp_df[['X', 'Y']].values
    fp_rssis = fp_df[mac_cols].values
    mac_to_idx = {mac: i for i, mac in enumerate(mac_cols)}

    # 2. 定义理论轨迹点 (Ground Truth) - 保持 2D 不变
    gt_waypoints = np.array([
        (0.6, 2.4),  # 起点 (左下)
        (0.6, 18),  # 拐角 1 (左上) -> 向上走
        (4.2, 18),  # 拐角 2 (右上) -> 向右走
        (4.2, 2.4),  # 拐角 3 (右下) -> 向下走
        (0.6, 2.4)  # 终点 (回到起点)-> 向左走
    ])

    # 3. 时间与空间距离的映射
    # 获取真实采集数据的总耗时
    start_time = real_df['Timestamp'].min()
    end_time = real_df['Timestamp'].max()
    total_time = (end_time - start_time).total_seconds()

    # 计算理论轨迹的总长度
    distances = np.linalg.norm(np.diff(gt_waypoints, axis=0), axis=1)
    cum_distances = np.insert(np.cumsum(distances), 0, 0)
    total_dist = cum_distances[-1]

    print(f"载入真实数据时长: {total_time:.2f} 秒")
    print(f"映射到理论轨迹长: {total_dist:.2f} 米")
    print(f"推算平均行走速度: {total_dist / total_time:.2f} m/s")

    # 4. 逐行注入仿真的 RSSI 信号
    simulated_rssi_list = []

    for _, row in real_df.iterrows():
        t = (row['Timestamp'] - start_time).total_seconds()
        mac = row['MAC']

        # 将当前时间点线性映射到轨迹的行驶距离上
        current_dist = (t / total_time) * total_dist

        # 【修改 2】：插值计算当前时间你应该处于的二维物理坐标 (X, Y)
        pt = np.zeros(2)
        for i in range(2):  # 仅遍历 X 和 Y
            pt[i] = np.interp(current_dist, cum_distances, gt_waypoints[:, i])

        # IDW (反距离权重) 准备：计算该点距离指纹库所有网格点的距离
        dists = np.linalg.norm(fp_locs - pt, axis=1)
        dists[dists < 1e-5] = 1e-5  # 防止除以 0
        weights = 1.0 / (dists ** 2)
        weights /= weights.sum()

        # 提取并在理论上加权计算这个特定 MAC 的 RSSI
        if mac in mac_to_idx:
            mac_idx = mac_to_idx[mac]
            ideal_rssi = np.sum(fp_rssis[:, mac_idx] * weights)

            # 加入高斯白噪声，模拟多径衰落
            noise = np.random.normal(0,1.0)
            final_rssi = int(round(ideal_rssi + noise))

            # 约束 RSSI 在正常设备的极限范围内
            final_rssi = max(-100, min(-30, final_rssi))
        else:
            final_rssi = row['RSSI']

        simulated_rssi_list.append(final_rssi)

    # 5. 替换掉原数据中的 RSSI 列
    real_df['RSSI'] = simulated_rssi_list

    # 保存数据
    real_df.to_csv(output_file, index=False)
    print(f"数据处理完毕！已将基于真实时序的仿真数据保存至: {output_file}")


# 运行代码
if __name__ == "__main__":
    inject_simulated_rssi(
        real_file=r'D:\isip\论文\我自己\代码\大厅\第二次测试\移动数据\融合数据\第8次移动蓝牙_融合.csv',
        fp_file=r"C:\Users\86150\Desktop\LSTM\大厅\第二次测试\指纹库",
        output_file=r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试\测试数据/第8次移动蓝牙_融合1.csv'
    )
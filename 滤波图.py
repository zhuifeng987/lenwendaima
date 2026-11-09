import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ==========================================
# 配置参数
# ==========================================
# 修改为需要处理的 IMU CSV 文件路径
DATA_FILE = r"C:\Users\86150\Desktop\第二次测试\测试数据\第8次移动惯导_融合.csv"

# 图像保存目录
BASE_SAVE_DIR = r"PDR滤波前后对比图"

# 是否显示生成的图像
SHOW_FIGURE = True


# ==========================================
# 绘图全局设置
# ==========================================
plt.rcParams["font.sans-serif"] = ["Times New Roman", "Arial"]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 低通滤波
# ==========================================
def lowpass_filter(data, cutoff=3.0, fs=100.0, order=4):
    """对一维加速度数据进行 Butterworth 低通滤波。"""
    nyquist_frequency = 0.5 * fs
    normalized_cutoff = cutoff / nyquist_frequency
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


# ==========================================
# 仅生成图1：滤波前后对比图
# ==========================================
def generate_filtering_comparison(file_path, save_dir):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    os.makedirs(save_dir, exist_ok=True)

    imu_df = pd.read_csv(file_path)

    required_columns = ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]
    missing_columns = [
        column for column in required_columns
        if column not in imu_df.columns
    ]
    if missing_columns:
        raise KeyError(f"CSV 缺少必要列: {missing_columns}")

    # 计算三轴合加速度，并去除均值分量
    imu_df["Acc_Mag"] = np.sqrt(
        imu_df["加速度X(g)"] ** 2
        + imu_df["加速度Y(g)"] ** 2
        + imu_df["加速度Z(g)"] ** 2
    )
    imu_df["Acc_Linear"] = (
        imu_df["Acc_Mag"] - imu_df["Acc_Mag"].mean()
    )

    # 低通滤波
    imu_df["Acc_Filtered"] = lowpass_filter(
        imu_df["Acc_Linear"].to_numpy()
    )

    # 单位由 g 转换为 m/s²
    acc_raw_ms2 = imu_df["Acc_Linear"] * 9.80665
    acc_filtered_ms2 = imu_df["Acc_Filtered"] * 9.80665
    samples = np.arange(len(imu_df))

    # 1 行 2 列：左右分布
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharex=True
    )

    ax1.plot(
        samples,
        acc_raw_ms2,
        color="blue",
        linewidth=1
    )
    ax1.set_title("Raw Acceleration", fontsize=14)
    ax1.set_xlabel("Sample Index", fontsize=12)
    ax1.set_ylabel("Acceleration (m/s$^2$)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(
        samples,
        acc_filtered_ms2,
        color="red",
        linewidth=1
    )
    ax2.set_title("Filtered Acceleration", fontsize=14)
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.set_ylabel("Acceleration (m/s$^2$)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    save_path = os.path.join(
        save_dir,
        "Fig1_Filtering_Comparison.png"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"图1已保存至: {save_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    generate_filtering_comparison(
        DATA_FILE,
        BASE_SAVE_DIR
    )
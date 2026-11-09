import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取指纹库数据
df = pd.read_csv('指纹库')

# 自定义信标数组
beacons = [
    (0.0, 0),      # 信标 1
    (0, 7.2),     # 信标 2
    (0, 15.6),    # 信标 3
    (0, 24)  ,    # 信标 4
    (5.4, 0),  # 信标 1
    (5.4, 7.2),  # 信标 2
    (5.4, 15.6),  # 信标 3
    (5.4, 24)  # 信标 4
]

# ==========================================
# 🌟 修改 1：在这里新增你的轨迹点数组 🌟
# ==========================================
trajectory = [
    (0.6, 2.4),   # 起点 (左下)
    (0.6, 18),    # 拐角 1 (左上) -> 向上走
    (4.2, 18),    # 拐角 2 (右上) -> 向右走
    (4.2, 2.4),   # 拐角 3 (右下) -> 向下走
    (0.6, 2.4)    # 终点 (回到起点)-> 向左走
]

# 2. 创建画布
plt.figure(figsize=(8, 12))

# 3. 绘制指纹库参考点 (黑色的圆点)
# plt.scatter(df['X'], df['Y'], color='black', marker='o', s=40, zorder=3, label='Reference Points')

# 绘制信标 (红色的三角形)
if beacons:
    beacon_x = [b[0] for b in beacons]
    beacon_y = [b[1] for b in beacons]
    plt.scatter(beacon_x, beacon_y, color='black', marker='o', s=150, zorder=4,  clip_on=False)


# 4. 动态设置刻度与网格线
x_ticks = np.sort(df['X'].unique())
y_ticks = np.sort(df['Y'].unique())

plt.xticks(x_ticks)
plt.yticks(y_ticks)

# 启用网格
plt.grid(True, which='major', linestyle='-', color='dimgray', linewidth=0.5, zorder=0)

# 5. 设置坐标轴标签和图例
plt.xlabel('x / m', fontsize=12, fontfamily='serif')
plt.ylabel('y / m', fontsize=12, fontfamily='serif')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# 6. 设置坐标轴比例一致，确保物理距离分布不形变
plt.gca().set_aspect('auto')

# 7. 严格限制坐标轴显示范围，紧贴数据点边缘，彻底去除网格外的留白
buffer = 0.2
plt.xlim([x_ticks.min(), x_ticks.max()])
plt.ylim([y_ticks.min(), y_ticks.max()])

# 8. 调整边距并保存
# pad=0 和 pad_inches=0 用于去除图表外的多余白边
plt.tight_layout(pad=3)
# plt.savefig('指纹图.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()
plt.savefig('指纹图.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# ==========================================
# 🌟 第二步：在原有画布上，继续叠加绘制轨迹 🌟
# ==========================================
if trajectory:
    traj_x = [t[0] for t in trajectory]
    traj_y = [t[1] for t in trajectory]
    plt.plot(traj_x, traj_y, color='blue', linestyle='-', linewidth=2, marker='*', markersize=8, zorder=5, label='Trajectory')
    plt.scatter(traj_x[0], traj_y[0], color='green', marker='s', s=100, zorder=6, label='Start')
    plt.scatter(traj_x[-1], traj_y[-1], color='purple', marker='X', s=100, zorder=6, label='End')

# 因为新加了轨迹的 label，我们需要在这里生成图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# ==========================================
# 🌟 第三步：保存带有轨迹的图 🌟
# ==========================================
plt.savefig('指纹图_带轨迹.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# 最后才调用 show 显示最终画面 (show 完之后画布就会被清空)
plt.show()
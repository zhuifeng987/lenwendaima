import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(42)

print("="*40)
print(" 第二步：构建并训练轻量级 LSTM 模型")
print("="*40)

WORK_DIR = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试'

# ==========================================
# 1. 加载第一步生成的数据与指纹库
# ==========================================
print("正在加载模拟时序数据...")
X_seq = np.load(os.path.join(WORK_DIR, 'X_seq_simulated.npy'), allow_pickle=True)
y_label = np.load(os.path.join(WORK_DIR, 'y_label_simulated.npy'), allow_pickle=True)

FILE_FINGERPRINT = os.path.join(WORK_DIR, '指纹库')
try:
    df_fp = pd.read_csv(FILE_FINGERPRINT)
except:
    df_fp = pd.read_csv(FILE_FINGERPRINT, encoding='gbk')

exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]
TIME_STEPS = X_seq.shape[1]
NUM_FEATURES = len(rssi_cols)

# ==========================================
# 2. 严格的归一化 (基于真实指纹库尺度)
# ==========================================
scaler = MinMaxScaler()
scaler.fit(df_fp[rssi_cols].values) # 必须基于指纹库拟合

X_seq_flat = X_seq.reshape(-1, NUM_FEATURES)
X_seq_norm = scaler.transform(X_seq_flat).reshape(X_seq.shape)

X_train, X_test, y_train, y_test = train_test_split(X_seq_norm, y_label, test_size=0.2, random_state=42)

# ==========================================
# 3. 搭建万能自适应 LSTM
# ==========================================
model = Sequential([
    LSTM(units=32, activation='tanh', input_shape=(TIME_STEPS, NUM_FEATURES)),
    Dropout(0.2),
    Dense(units=2)
])
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_path = os.path.join(WORK_DIR, 'best_lstm_model.h5')
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)

print("开始训练网络...")
model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, callbacks=[early_stop, checkpoint], verbose=1)

print(f"\n✅ 训练完成！最佳模型已保存至: {model_path}")


# #带损耗图
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt  # <--- 新增导入 matplotlib 用于绘图
# import os
# import warnings
#
# # ==========================================
# # [新增核心修复]：强行关闭 HDF5 的文件锁机制
# # ==========================================
# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
#
# warnings.filterwarnings('ignore')
# tf.random.set_seed(42)
#
# print("="*40)
# print(" 第二步：构建并训练轻量级 LSTM 模型")
# print("="*40)
#
# WORK_DIR = r'C:\Users\86150\Desktop\LSTM\大厅\第二次测试'
#
# # ==========================================
# # 1. 加载第一步生成的数据与指纹库
# # ==========================================
# print("正在加载模拟时序数据...")
# X_seq = np.load(os.path.join(WORK_DIR, 'X_seq_simulated.npy'), allow_pickle=True)
# y_label = np.load(os.path.join(WORK_DIR, 'y_label_simulated.npy'), allow_pickle=True)
#
# FILE_FINGERPRINT = os.path.join(WORK_DIR, '指纹库')
# try:
#     df_fp = pd.read_csv(FILE_FINGERPRINT)
# except:
#     df_fp = pd.read_csv(FILE_FINGERPRINT, encoding='gbk')
#
# exclude_cols = ['X', 'Y', 'Point_ID', 'Timestamp', 'Unnamed: 0']
# rssi_cols = [col for col in df_fp.columns if col not in exclude_cols]
# TIME_STEPS = X_seq.shape[1]
# NUM_FEATURES = len(rssi_cols)
#
# # ==========================================
# # 2. 严格的归一化 (基于真实指纹库尺度)
# # ==========================================
# scaler = MinMaxScaler()
# scaler.fit(df_fp[rssi_cols].values) # 必须基于指纹库拟合
#
# X_seq_flat = X_seq.reshape(-1, NUM_FEATURES)
# X_seq_norm = scaler.transform(X_seq_flat).reshape(X_seq.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X_seq_norm, y_label, test_size=0.2, random_state=42)
#
# # ==========================================
# # 3. 搭建万能自适应 LSTM
# # ==========================================
# model = Sequential([
#     LSTM(units=32, activation='tanh', input_shape=(TIME_STEPS, NUM_FEATURES)),
#     Dropout(0.2),
#     Dense(units=2)
# ])
# model.compile(optimizer='adam', loss='mse')
#
# early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# model_path = os.path.join(WORK_DIR, 'best_lstm_model.h5')
# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=0)
#
# print("开始训练网络...")
# # <--- 修改点：用 history 变量接收训练过程数据
# history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.2, callbacks=[early_stop, checkpoint], verbose=1)
#
# print(f"\n✅ 训练完成！最佳模型已保存至: {model_path}")
#
# # ==========================================
# # 4. 绘制并保存训练损失曲线
# # ==========================================
# print("正在绘制训练曲线图...")
# plt.figure(figsize=(10, 6), dpi=150) # 设置画布大小和清晰度
#
# # 提取 history 中的数据进行绘制
# plt.plot(history.history['loss'], label='Train Loss (MSE)', linewidth=1.5)
# plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', linewidth=1.5)
#
# # 设置图表标题、标签和样式，严格对齐你的参考图
# plt.title('LSTM Training Curve (Loss vs Epochs)', fontsize=14)
# plt.xlabel('Epochs', fontsize=12)
# plt.ylabel('Loss (MSE)', fontsize=12)
# plt.legend(fontsize=11)
# plt.grid(True)
#
# # 保存图片到工作目录并展示
# plot_path = os.path.join(WORK_DIR, 'lstm_training_curve.png')
# plt.savefig(plot_path, bbox_inches='tight')
# print(f"✅ 训练曲线图已保存至: {plot_path}")
#
# plt.show() # 在窗口中展示图片
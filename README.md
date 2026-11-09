# 蓝牙与PDR粒子滤波融合室内定位

本项目实现了一套面向室内场景的蓝牙与惯性导航融合定位方法。系统利用LSTM从蓝牙RSSI序列中估计二维位置，利用PDR完成步态检测、步长估计和航向推算，并通过事件驱动的异步粒子滤波融合两类定位结果。

## 主要功能

- 蓝牙RSSI卡尔曼滤波与时间对齐
- KNN、WKNN和LSTM蓝牙定位对比
- 自适应双约束PDR计步与Weinberg步长估计
- LSTM、PDR与粒子滤波异步融合
- 有效路径段约束和定位误差CDF分析
- 中英文结果图生成

## 运行环境

- Python 3.8
- TensorFlow 2.13.0
- NumPy、Pandas、SciPy、Matplotlib、scikit-learn和h5py

安装依赖：

```bash
pip install -r requirements.txt
```

## 推荐入口

```bash
python 中文-蓝牙结果图.py
python 中文-PDR.py
python 中文-融合定位代码-active_segment.py
```

中文脚本使用项目相对路径读取指纹库、实测数据和模型。部分历史英文脚本仍保留原始绝对路径配置，换用其他计算机时需要修改相应路径。

## 主要数据与模型

- `指纹库`：蓝牙RSSI指纹及二维坐标
- `实测数据/`：蓝牙与IMU实测数据
- `best_lstm_model.h5`：训练完成的LSTM定位模型
- `X_seq_simulated.npy`、`y_label_simulated.npy`：模型训练数据

## 融合方法概述

粒子状态由二维位置和航向角组成。IMU步态事件用于执行粒子运动预测，LSTM蓝牙位置作为观测更新粒子权重；当有效粒子数下降到阈值以下时进行重采样。有效路径段版本还会降低偏离当前候选路线段粒子的权重。

## 说明

生成的结果图和IDE本地配置已通过 `.gitignore` 排除。项目中的实测数据和训练模型仅用于科研与算法验证。

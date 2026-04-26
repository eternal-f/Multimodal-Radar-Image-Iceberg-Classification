# Chinese After English [后附中文]
# Multimodal Radar Image Iceberg Classification - Kaggle Competition

This repository contains a complete solution for the **Statoil/C-CORE Iceberg Detection Challenge** on Kaggle. The goal is to classify radar images as either iceberg or ship, using a combination of CNN ensembles, KNN clustering, and Gradient Boosting. (Competition Link: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/leaderboard)

## Leaderboard Performance

- **Our Team Final LogLoss**: 0.08415  
- **Top 1 LogLoss**: 0.08227  
- **Top 2 LogLoss**: 0.08555  

## Approach Overview

The solution is inspired by top-performing teams and uses a multi-stage pipeline:

1. **CNN Ensemble** – Multiple convolutional neural networks trained with data augmentation.
2. **KNN Clustering** – Uses incident angle (`inc_angle`) similarity to refine predictions.
3. **Gradient Boosting** – Combines CNN and KNN outputs for final prediction.

## Key Insights

- CNN alone achieves only **LogLoss ~0.19**.
- KNN alone achieves **LogLoss ~0.20**.
- **Boosting (CNN + KNN + GBM)** brings LogLoss down to **~0.01** on CV, with final test score **0.084**.
- `inc_angle` is a critical feature – identical values (to 4 decimal places) share the same label 97% of the time.
- Avoid extreme probability thresholds (e.g., mapping to 0.99/0.001) – simple clipping to `[0.001, 0.999]` works best.

## Repository Structure
.
├── Data/ # Training and test JSON files (not included)
├── save_model/ # Saved Keras models per fold
├── train_predict/ # CNN predictions for train/test sets
├── knn/ # Output features from KNN + CNN
├── Dataprocess.py # Data loading, preprocessing, augmentation
├── Model.py # CNN architecture (without inc_angle)
├── Train.py # Training with cross-validation and augmentation
├── KNN.py # KNN clustering using inc_angle and CNN predictions
├── GBM.py # LightGBM boosting on CNN + KNN features
└── README.md


## Method Evolution

### 1. Pure CNN
- **LogLoss**: 0.24  
- Dual-band images (75×75×2) + `inc_angle` fed into CNN  
- Heavy use of batch norm and dropout  
- Early stopping & learning rate scheduling

### 2. CNN + Data Augmentation
- **LogLoss**: 0.19  
- Removed `inc_angle` from CNN input (better focus on image features)  
- Used DataAugmentation with rotation, shift, and flip  
- Reduced model depth to 3 conv layers (32/64/128) to avoid overfitting

### 3. KNN
- **LogLoss**: 0.20  
- KNN uses only `inc_angle` and CNN pseudo-labels on test set  
- Identifies highly similar samples by `inc_angle` (4 decimal places)

### 4. CNN + KNN + Boosting
- **LogLoss**: ~0.01   
- LightGBM on features:  cnn's predict and knn's predict  

### 5. CNN Voting + KNN + Boosting
- **LogLoss**: 0.096  
- Ensemble of 4 CNNs with different augmentation & dropout settings  

### 6. Fine-tuning + 9-model Voting
- **LogLoss**: 0.084  
- KNN with n=30 neighbors  
- Final clipping of probabilities to `[0.001, 0.999]`

# 中文版
# 多模态雷达图像冰山分类 - Kaggle竞赛

本仓库包含**Statoil/C-CORE冰山探测挑战赛**的完整解决方案。竞赛目标是通过CNN集成、KNN聚类和梯度提升的组合，对雷达图像进行冰山或船舶的二分类。（竞赛链接：https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/leaderboard） 

## 排行榜表现
- **本组最终对数损失**：0.08415

- **第1名对数损失**：0.08227

- **第2名对数损失**：0.08555

## 方案概述
本方案受前几名团队启发，采用多阶段流程： 
 
1. **CNN集成** – 使用数据增强训练的多个卷积神经网络。 
2. **KNN聚类** – 利用入射角（`inc_angle`）相似性优化预测。 
3. **梯度提升** – 融合CNN和KNN的输出进行最终预测。 

## 关键发现
- 仅用CNN只能达到对数损失约**0.19**。
- 仅用KNN只能达到对数损失约**0.20**。
- 提升方法（**CNN + KNN + GBM**） 在交叉验证中将对数损失降至约**0.01**，最终测试得分为**0.084**。
- `inc_angle`是关键特征 – 在小数点后4位相同的情况下，97%的样本标签一致。
- 避免极端概率映射（例如映射到0.99/0.001） – 简单的裁剪到`[0.001, 0.999]`效果最佳。
 
## 文件结构
.
├── Data/ # 训练和测试JSON文件（未包含）
├── save_model/ # 每折保存的Keras模型
├── train_predict/ # 训练集/测试集的CNN预测结果
├── knn/ # KNN + CNN 输出的特征
├── Dataprocess.py # 数据加载、预处理、增强
├── Model.py # CNN架构（不含inc_angle）
├── Train.py # 使用交叉验证和增强的训练
├── KNN.py # 基于inc_angle和CNN预测的KNN聚类
├── GBM.py # 基于CNN+KNN特征的LightGBM提升
└── README.md

## 方法演进
### 1. 纯CNN
- **对数损失**：0.24
- 双波段图像（75×75×2）+ inc_angle 输入CNN
- 大量使用批归一化和dropout
- 早停与学习率调度

### 2. CNN + 数据增强
**对数损失**：0.19
- 从CNN输入中移除`inc_angle`（让网络更专注于图像特征）
- 使用带旋转、平移、翻转的数据增强
- 将模型深度减少到3个卷积层（32/64/128），避免过拟合

### 3. KNN
**对数损失**：0.20
KNN仅使用`inc_angle`和测试集上的CNN伪标签
通过`inc_angle`（小数点后4位）识别高度相似的样本

### 4. CNN + KNN + 提升
**对数损失**：~0.01
使用LightGBM，特征包括：CNN预测值和KNN预测值

### 5. CNN投票 + KNN + 提升
**对数损失**：0.096
集成4个CNN，采用不同的增强和dropout设置

### 6. 微调 + 9模型投票
**对数损失**：0.084
KNN调整为n=30
最终将概率裁剪到`[0.001, 0.999]`


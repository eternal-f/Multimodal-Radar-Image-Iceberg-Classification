# Multimodal Radar Image Iceberg Classification - Kaggle Competition

This repository contains a complete solution for the **Statoil/C-CORE Iceberg Detection Challenge** on Kaggle. The goal is to classify radar images as either iceberg or ship, using a combination of CNN ensembles, KNN clustering, and Gradient Boosting. (Competition Link: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/leaderboard)

## Leaderboard Performance

- **Final LogLoss**: 0.08415  
- **Top 1 LogLoss**: 0.08227  
- **Top 2 LogLoss**: 0.08555  

## Approach Overview

The solution is inspired by top-performing teams (Top 1, 2, 4) and uses a multi-stage pipeline:

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

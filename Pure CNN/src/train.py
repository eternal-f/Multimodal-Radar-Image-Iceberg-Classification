import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.utils.class_weight import compute_class_weight
from model import *

def train_model(x_train, inc_train, y_train, x_val, inc_val, y_val):
    # 处理类别不平衡
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    # 初始化模型
    model = create_model()
    optimizer = Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    # 训练
    history = model.fit(
        [x_train, inc_train], y_train,
        validation_data=([x_val, inc_val], y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    return model, history
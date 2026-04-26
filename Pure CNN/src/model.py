from tensorflow.keras import layers, models

def create_model():
    """创建多输入模型"""
    img_input = layers.Input(shape=(75, 75, 2), name='image')
    # 归一化
    x = layers.BatchNormalization()(img_input)
    # 卷积块1
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 卷积块2
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 卷积块3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # 入射角输入分支
    inc_input = layers.Input(shape=(1,), name='incidence_angle')
    inc = layers.Dense(16, activation='relu')(inc_input)
    inc = layers.Dropout(0.3)(inc)

    # 融合
    combined = layers.Concatenate()([x, inc])

    # 全连接层
    combined = layers.Dense(32, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(16, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)

    # 输出
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = models.Model(
        inputs=[img_input, inc_input],
        outputs=output
    )

    return model
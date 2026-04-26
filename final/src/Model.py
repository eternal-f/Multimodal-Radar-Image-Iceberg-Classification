from keras import layers
from keras import models

def create_model_without_inc():
    """
    不加入inc
    根据前五个队伍的经验，专注于图像的鲁棒性能更强
    """
    img_input = layers.Input(shape=(75, 75, 2), name='image')
    x = layers.BatchNormalization(momentum=0.0)(img_input)
    # 卷积块1
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    # 卷积块2
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    # 卷积块3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # 全连接层
    fc1 = layers.Dense(128, activation='relu')(x)
    fc1 = layers.BatchNormalization()(fc1)
    fc1 = layers.Dropout(0.5)(fc1)
    # 输出
    output = layers.Dense(1, activation='sigmoid')(fc1)

    model = models.Model(
        inputs=[img_input],
        outputs=output
    )

    return model

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_process(data):
    x = []
    inc_angles = []
    y = []

    for item in data:
        # 数据重新写为75x75的图片，并堆叠在一起
        band1 = np.array(item['band_1']).reshape(75, 75)
        band2 = np.array(item['band_2']).reshape(75, 75)
        image = np.stack([band1, band2], axis=-1)
        x.append(image)
        # 处理入射角
        inc = item.get('inc_angle')
        if inc == 'na' or inc is None:
            inc_angles.append(-999)  # 用-999替换缺失
        else:
            inc_angles.append(float(inc))
        if 'is_iceberg' in item:
            y.append(item['is_iceberg'])

    x = np.array(x)
    inc_angles = np.array(inc_angles).reshape(-1, 1)
    # 处理缺失值：用中位数填充
    median_inc = np.median(inc_angles[inc_angles != -999])
    inc_angles[inc_angles == -999] = median_inc
    # 标准化入射角
    scaler = StandardScaler()
    inc_angles = scaler.fit_transform(inc_angles)

    return x, inc_angles, np.array(y)

def create_augmentation(x, y, s):
    # 创建数据增强生成器，每批次64个，循环无限生成，batchsize对应train的超参
    batch_size = 64
    augmentation = ImageDataGenerator(
        rotation_range=10,           # 旋转0-30度
        width_shift_range=0.12,       # 水平平移
        height_shift_range=0.12,      # 垂直平移
        # horizontal_flip=True,        # 水平翻转
        vertical_flip=True,
        # fill_mode='reflect'
    )
    result = augmentation.flow(x, y, batch_size=batch_size, seed=s)
    while True:
        yield next(result)

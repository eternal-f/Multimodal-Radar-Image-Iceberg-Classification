import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsRegressor
def data_load(path, flag):
    """
    path: 训练集、测试集路径
    flag: 标记当前读入数据条是来自于train(True)还是test
    """
    result = pd.read_json(path, dtype={'inc_angle': 'str'})    # 读入数据，inc转为str文本，用于后续替换空数据
    result = result.drop(columns=['band_1', 'band_2'])        # 用inc_angle进行聚类，band_1和2用不着
    result = result.set_index('id')                           # 用id作为索引，方便后续定位行
    result['inc_angle'] = result.inc_angle.replace('na', '0.0')   # 替换空数据
    result['fake'] = result.inc_angle.apply(lambda x: len(x.split('.')[1]) > 5) # inc大于4位的fake
    result['inc_angle'] = result.inc_angle.astype('float64')  # 转回float
    result['train'] = flag
    return result

def add_predict(in_put):
    train_predict = np.load('./train_predict/train_predict_520491080')
    test_predict = np.load('./train_predict/test_predict_520491080')

    in_put.loc[in_put.train, 'cnn_predict'] = train_predict
    in_put.loc[~in_put.train, 'cnn_predict'] = test_predict
    in_put.loc[~in_put.train, 'is_iceberg'] = test_predict

    return in_put

def group_stats(gro):
    best = gro.is_iceberg.values
    predict = gro.cnn_predict.values

    med = np.zeros(len(gro))
    mea = np.zeros(len(gro))
    cnt = np.zeros(len(gro))

    for idx in range(len(gro)):
        # 创建临时数组，不修改原数组
        temp_array = best.copy()
        temp_array[idx] = predict[idx]

        med[idx] = np.median(temp_array)
        mea[idx] = np.mean(temp_array)
        cnt[idx] = len(temp_array)

    gro = gro.copy()
    gro['med'] = med
    gro['mea'] = mea
    gro['cnt'] = cnt
    return gro

def create_knn(in_put):
    for key in in_put.index:
        mask = (~in_put.fake) & (in_put.index != key)
        knn = KNeighborsRegressor(n_neighbors=20, weights='distance', algorithm='brute')
        knn.fit(in_put[mask].inc_angle.values.reshape((-1, 1)), in_put[mask].cnn_predict.values)
        angle_value = in_put.loc[key, 'inc_angle']
        in_put.loc[key, 'knn_predict'] = knn.predict([[angle_value]])

    print('KNN CV loss: {}'.format(log_loss(in_put[in_put.train].is_iceberg, in_put[in_put.train].knn_predict)))
    return in_put

if __name__ == '__main__':
    content = pd.concat([
        data_load('./Data/train.json', True),
        data_load('./Data/test.json', False),
    ])
    data = add_predict(content)
    data = data.groupby('inc_angle').apply(lambda x: group_stats(x))
    data = data.reset_index()
    data = create_knn(data)
    data.to_csv('./knn/feature.csv', header=True)

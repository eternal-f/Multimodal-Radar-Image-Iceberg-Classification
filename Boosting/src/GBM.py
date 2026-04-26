import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm

def lgbm_cv(df):
    x = df[df.train][['cnn_predict', 'med', 'mea', 'cnt', 'knn_predict']].values
    y = df[df.train][['is_iceberg']].values.squeeze()
    # CV for LightGBM.
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    cv = np.zeros(len(x))
    for fold_id, (train_idx, test_idx) in enumerate(kf.split(x, y)):
        cl = lightgbm.LGBMClassifier(max_depth=3, n_iter=70, learning_rate=0.1, min_child_samples=40, verbose=-1)
        cl.fit(
            x[train_idx], y[train_idx],
            eval_set=[(x[test_idx], y[test_idx])],
            eval_metric='logloss'
        )
        cv[test_idx] = cl.predict_proba(x[test_idx])[:, 1]
    print('LightGBM CV loss: {}'.format(log_loss(df[df.train].is_iceberg, np.clip(cv, 0.001, 0.999))))

def lgbm(df):
    x = df[df.train][['cnn_predict', 'med', 'mea', 'cnt', 'knn_predict']].values
    y = df[df.train][['is_iceberg']].values.squeeze()
    z = df[~df.train][['cnn_predict', 'med', 'mea', 'cnt', 'knn_predict']]

    cl = lightgbm.LGBMClassifier(max_depth=3, n_iter=70, learning_rate=0.1, min_child_samples=40)
    cl.fit(x, y)
    return np.clip(cl.predict_proba(z)[:,1], 0.001, 0.999)

def main():
    features = pd.read_csv('./knn/feature.csv', index_col='id')
    lgbm_cv(features)
    predictions = lgbm(features)
    pd.DataFrame(
        data=predictions, index=features[~features.train].index, columns=['is_iceberg']
    ).to_csv('result.csv', header=True)


if __name__ == '__main__':
    main()

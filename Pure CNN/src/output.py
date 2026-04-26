import json
from dataprocess1 import *
from train import *
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
# 读取数据
with open('./Data/train.json', 'r') as f:
    train_data = json.load(f)
# 划分训练测试
X, inc_angles, y = data_process(train_data)
X_train, X_val, inc_train, inc_val, y_train, y_val = train_test_split(
    X, inc_angles, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# 训练
model, history = train_model(
    X_train, inc_train, y_train,
    X_val, inc_val, y_val
)
# 预测结果
with open('./Data/test.json', 'r') as f:
    test_data = json.load(f)

X_test, inc_test, _ =data_process(test_data)  # test无标签
model_pre = load_model('best_model.h5')
predictions = model_pre.predict([X_test, inc_test])
# 生成提交文件
submission = pd.DataFrame({
    'id': [item['id'] for item in test_data],
    'is_iceberg': predictions.flatten()
})
submission.to_csv('submission.csv', index=False)

import pandas as pd
import json
from Dataprocess import data_process
from tensorflow.keras.models import load_model

with open('./Data/test.json', 'r') as f:
    test_data = json.load(f)

X_test, inc_test, _ =data_process(test_data)  # test无标签
for i in range(5):
    model = load_model('./save_model/best_model_{}.keras'.format(i))
    predictions = model.predict(X_test)

    # 8. 生成提交文件
    submission = pd.DataFrame({
        'id': [item['id'] for item in test_data],
        'is_iceberg': predictions.flatten()
    })
    submission.to_csv('submission_{}.csv'.format(i), index=False)
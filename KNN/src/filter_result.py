import pandas as pd

df = pd.read_csv('./knn/feature.csv')

# 删除 train 列为 True 的行
df_filtered = df[df['train'] != True]
df_filtered.to_csv('feature_filtered.csv', index=False)

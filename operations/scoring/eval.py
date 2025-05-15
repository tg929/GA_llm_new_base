import pandas as pd
import numpy as np

def normalize_ds(ds_series):
    """
    归一化: DS_hat = -clip(DS, -20, 0) / 20
    """
    clipped = np.clip(ds_series, -20, 0)
    return -clipped / 20

def normalize_sa(sa_series):
    """
    归一化: SA_hat = (10 - SA) / 9
    """
    return (10 - sa_series) / 9

def calculate_composite_score(df):
    """
    输入DataFrame,包含列:'smiles', 'DS', 'QED', 'SA'
    输出DataFrame,增加一列'composite_score'，即 y = DS_hat * QED * SA_hat
    其中DS_hat和SA_hat为文献归一化公式。
    """
    df = df.copy()
    df['DS_hat'] = normalize_ds(df['DS'])
    df['SA_hat'] = normalize_sa(df['SA'])
    df['composite_score'] = df['DS_hat'] * df['QED'] * df['SA_hat']
    return df

# 示例用法：
# df = pd.read_csv('your_file.csv')
# df = calculate_composite_score(df)
# df.to_csv('scored_file.csv', index=False)

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features_from_date

def load_dataset(
    datadir: str,
    dataname: str,
    split_rate: list,
    seq_len: int,
    features: str = 'S',
    time_embedding: list = [True, 'h'],
    del_feature: list = None
):
    
    file_path = os.path.join(datadir, f"{dataname}.csv")
    df_raw = pd.read_csv(file_path)

    if del_feature is not None:
        df_raw = df_raw.drop(columns=del_feature)

    # ETTh1(1시간 단위)라면 1개월 = 30일 * 24시간 = 720
    # ETTm1(15분 단위)라면 1개월 = 30일 * 24시간 * 4번 = 2880
    if dataname.startswith('ETTh'):
        months_to_rows = 30 * 24 
    elif dataname.startswith('ETTm'):
        months_to_rows = 30 * 24 * 4 
    else:
        raise ValueError(f"Unknown dataset name: {dataname} (expected to start with 'ETTh' or 'ETTm')")
    
    total_len = 20 * months_to_rows
    df_raw = df_raw.iloc[:total_len]

    if features == 'M' or features == 'MS': # 다변량
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features == 'S': # 단변량
        cols_data = df_raw.columns[-1:]
        df_data = df_raw[cols_data]

    var = df_data.columns.tolist()

    num_train = int(total_len * split_rate[0])
    num_val = int(total_len * split_rate[1])

    border1s = [0, num_train - seq_len, num_train + num_val - seq_len]
    border2s = [num_train, num_train + num_val, total_len] # 12/4/4개월
       
    data = df_data.values # (14400, 7) or (14400, 1)

    timeenc_val = time_embedding[0]
    freq_val = time_embedding[1]
    
    data_stamp_raw = time_features_from_date(
        date_series=df_raw['date'], 
        timeenc=timeenc_val, 
        freq=freq_val
    )

    if isinstance(data_stamp_raw, pd.DataFrame):
        data_stamp = data_stamp_raw.values 
    else:
        data_stamp = data_stamp_raw

    trn = data[border1s[0] : border2s[0]] # x차원만 영향을 받음
    val = data[border1s[1] : border2s[1]]
    tst = data[border1s[2] : border2s[2]]

    trn_ts = data_stamp[border1s[0] : border2s[0]]
    val_ts = data_stamp[border1s[1] : border2s[1]]
    tst_ts = data_stamp[border1s[2] : border2s[2]]

    return trn, trn_ts, val, val_ts, tst, tst_ts, var

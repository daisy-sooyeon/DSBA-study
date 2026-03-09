import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features_from_date


def load_dataset(
    datadir: str,
    dataname: str,
    val_split_rate: list,
    time_embedding: list = [0, 'd'],
    del_feature: list = None
):
    
    dataset_path = os.path.join(datadir, dataname)
    
    train_path = os.path.join(dataset_path, 'train.csv')
    test_path = os.path.join(dataset_path, 'test.csv')
    label_path = os.path.join(dataset_path, 'test_label.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    label_df = pd.read_csv(label_path)

    train_df = train_df.interpolate(method='linear', limit_direction='both')
    test_df = test_df.interpolate(method='linear', limit_direction='both')

    if del_feature is not None:
        train_df = train_df.drop(columns=del_feature)
        test_df = test_df.drop(columns=del_feature)

    use_time_features = time_embedding[0]
    freq = time_embedding[1]

    if dataname.startswith('PSM'):
        trn_ts = None
        val_ts = None
        tst_ts = None

    else:
        train_time_features = time_features_from_date(train_df.iloc[:, 0], timeenc=use_time_features, freq=freq)
        test_time_features = time_features_from_date(test_df.iloc[:, 0], timeenc=use_time_features, freq=freq)

        if isinstance(train_time_features, pd.DataFrame):
            trn_ts = train_time_features.values
        if isinstance(test_time_features, pd.DataFrame):
            tst_ts = test_time_features.values

    train_data = train_df.iloc[:, 1:].values
    tst = test_df.iloc[:, 1:].values
    label = label_df.iloc[:, -1].values

    var = train_data.shape[1]

    trn, val = train_test_split(train_data, test_size=val_split_rate, shuffle=False)

    return trn, trn_ts, val, val_ts, tst, tst_ts, var, label
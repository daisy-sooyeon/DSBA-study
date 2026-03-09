from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def apply_scaling(scaler: str, trn: np.ndarray, val: np.ndarray, tst: np.ndarray):

    if scaler == 'minmax':
        scaler_obj = MinMaxScaler()
    elif scaler == 'standard':
        scaler_obj = StandardScaler()
    elif scaler == 'minmax_square':
        scaler_obj = MinMaxScaler()
    elif scaler == 'minmax_m1p1':
        scaler_obj = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown scaler type: {scaler} (expected 'minmax', 'standard', 'minmax_square', or 'minmax_m1p1')")

    scaler_obj.fit(trn)
    trn_scaled = scaler_obj.transform(trn)
    val_scaled = scaler_obj.transform(val)
    tst_scaled = scaler_obj.transform(tst)

    if scaler == 'minmax_square':
        trn_scaled = np.square(trn_scaled)
        val_scaled = np.square(val_scaled)
        tst_scaled = np.square(tst_scaled)

    return trn_scaled, val_scaled, tst_scaled
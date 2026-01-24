import numpy as np
from sklearn.metrics import r2_score

def MAE(pred, true):
    return np.mean(np.abs(true-pred))

def MSE(pred, true):
    return np.mean((true-pred)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def R2(pred, true):
    return r2_score(true, pred)

def AE(pred, true):
    pass

def SE(pred, true):
    pass

def metric(pred, true, pred_len):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    if pred_len == 1:
        r2 = R2(pred, true)
        return r2, mae, mse, rmse
    return mae, mse, rmse
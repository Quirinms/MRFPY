import numpy as np
from scipy.optimize import minimize
# Use internal functions

def int_mae(arr_true, arr_pred):
    len_mae = len(arr_pred)
    mae = 0
    for i in range(len_mae):
        mae += abs(arr_true[i] - arr_pred[i])
    return mae/len_mae

def arr_mae(arr_true, arr_pred):
    len_mae = len(arr_pred)
    arr_mae = np.zeros(len_mae)
    for i in range(len_mae):
        arr_mae[i] = abs(arr_true[i] - arr_pred[i])
    return arr_mae












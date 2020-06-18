import numpy as np
import itertools
# Use internal functions
from .RollingWindow import rolling_window_one_step, rolling_window_multi_step

def coeff_choice(max_coeffs, max_level):
    x = np.arange(1, max_coeffs+1)
    length = max_level
    lst_cc = [p for p in itertools.product(x, repeat=length)]
    return lst_cc

def aic_onestep(data, numLevel = 1, numCoeff = 2, window = 1, model = "r"):
    """
    data        np.array - contains only values of a time series
    numLevel    int - Number of wavelet levels. Trend level is accounted automatically
    numCoeff    int - Number of coefficients per scale
    window      int - Window, over which the onestep is created
    model       String - Choose the forecast model. r = regression, nn = neural network
    """
    dct_CCPS_P = {}            # Parameter setting ccps -> AIC & APE
    numLevel = numLevel + 1    # +1 to account for trend level
    numCoeff = numCoeff + 1    #
    for level in range(2, numLevel):    # Create all levels from 1 to numLevel
        # Combination setting
        lst_coeff_choice = coeff_choice(numCoeff, level)    # Create all combinations for this level
        for ccps in lst_coeff_choice:
            # Cast
            arr_ccps = np.array(ccps)
            # Adjust
            scales = len(arr_ccps) - 1       # Wavelet levels, dont account for the one smooth level
            # Method application
            arr_mae, arr_mase, arr_fco = rolling_window_one_step(data, window, scales, arr_ccps, model = "r")
            # Performance measures: Akaikes Information Criterion + Accumulative Prediction Error
            ftMae = np.sum(arr_mae)/len(arr_mae)                           # MAE Mean Absolute Error
            ftAIC = len(data)*np.log(ftMae**2) + 2*(np.sum(arr_ccps)+1)    # AIC
            ftAPE = np.sum(arr_mae)                                        # APE
            dct_CCPS_P.update({ccps : {"AIC": ftAIC, "APE": ftAPE}})       # Dictionary update
    return dct_CCPS_P

def aic_multistep(data, steps = 2, numLevel = 1, numCoeff = 2, window = 1, model = "r", strategy = "r"):
    """
    data        np.array - contains only values of a time series
    steps       int - Number of steps to forecast in to the future
    numLevel    int - Number of wavelet levels. Trend level is accounted automatically
    numCoeff    int - Number of coefficients per scale
    window      int - Window, over which the onestep is created
    model       String - Choose the forecast model. r = regression, nn = neural network
    """
    dct_CCPS_P = {}            # Parameter setting ccps -> AIC & APE
    numLevel = numLevel + 1
    numCoeff = numCoeff + 1
    for level in range(2, numLevel):    # Create all levels from 1 to numLevel
        # Combination setting
        lst_coeff_choice = coeff_choice(numCoeff, level)    # Create all combinations for this level
        for ccps in lst_coeff_choice:
            # Cast
            arr_ccps = np.array(ccps)
            # Adjust
            scales = len(arr_ccps) - 1       # Wavelet levels, dont account for the one smooth level
            # Method application
            arr_mae, arr_mase = rolling_window_multi_step(data, window, steps, scales,
                                                          arr_ccps, model = "r", strategy = strategy)
            # Performance measures: Akaikes Information Criterion + Accumulative Prediction Error
            ftSum = np.sum(arr_mae)                                        # Sum of MAE's
            ftNum = arr_mae.shape[0] * arr_mae.shape[1]                    # Number of measured MAE's
            ftMae = ftSum/ftNum                                            # Mean Absolute Error
            ftAIC = len(data)*np.log(ftMae**2) + 2*(np.sum(arr_ccps)+1)    # AIC
            ftAPE = ftSum                                                  # APE
            dct_CCPS_P.update({ccps : {"AIC": ftAIC, "APE": ftAPE}})       # Dictionary update
    return dct_CCPS_P










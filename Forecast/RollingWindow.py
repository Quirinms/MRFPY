import numpy as np
from scipy.optimize import minimize
# Use internal functions
from .Regression import onestep as ros, recursive_multistep as rrms, far_horizon_multistep as rfms
from .NeuralNet import onestep as nnos, recursive_multistep as nnrms, far_horizon_multistep as nnfms
from .KnowledgeDiscovery import int_mae, arr_mae

def rolling_window_one_step(data, window, scales, ccps, model="r"):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    window    Window size for rolling window procedure
    scales    Number of resolution levels to create. Remember that there are scales many wavelet
              levels plus one level for the smooth coefficients
    ccps      Contains the number of coefficients to use on each scale individually. Remember
              the plus one entry for the smooth level.
    model     Choose auto regressive or neural network

    IMPORTANT: Rolling window procedure only for ONE STEP computations
    """
    if model in ["r", "R", "ar", "Ar", "AR", "regression", "Regression"]:
        model = "r"
    elif model in ["n", "nn", "NN", "nN", "Nn", "neural network", "Neural Network"]:
        model = "n"
    else:
        model = "n"
        print("Neural network is chose as underlying forecasting procedure")

    arr_MAE_Forecast = np.zeros(window)  # Computes error (MAE) on forecasted
    arr_MASE_Forecast = np.zeros(window)  # Computes error (MAE) on forecasted
    arr_Forecast_One_Step = np.zeros(window)  # One-Step Forecast

    for i in range(window):  # Rolling forecast procedure
        data_window = data[:-window + i]  # Data for current position
        if model == "r":  # Choice of method
            fc = ros(data_window, scales, ccps)
        else:
            fc = nnos(data_window, scales, ccps)

        arr_Forecast_One_Step[i] = fc  # Save forecast - only for one step
        arr_T = data[-window + i]  # True value - only for one step
        ft_MAE = int_mae(np.array([arr_T]), np.array([fc]))  # MAE

        n = 10  # Preparation MASE: MAE Randowm Walk
        if (i == (window - 1)):  # True values length n for random walk
            arr_T = data[-n:]  # For last step
        else:
            arr_T = data[:-window + i + 1][-n:]  # For all steps before the last

        arr_RW = data[:-window + i][-10:]  # Forecast length n for random walk
        ft_MAE_rw = int_mae(np.array(arr_T), np.array(arr_RW))  # MAE random walk
        ft_MASE = ft_MAE / ft_MAE_rw  # MASE

        arr_MAE_Forecast[i] = ft_MAE  # Save MAE results
        arr_MASE_Forecast[i] = ft_MASE  # Save MASE results
    return arr_MAE_Forecast, arr_MASE_Forecast, arr_Forecast_One_Step


def rolling_window_multi_step(data, window, horizon, scales, ccps, model="r", strategy="r"):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    window    Window size for rolling window procedure
    horizon   Forecast horizon
    scales    Number of resolution levels to create. Remember that there are scales many wavelet
              levels plus one level for the smooth coefficients
    ccps      Contains the number of coefficients to use on each scale individually. Remember
              the plus one entry for the smooth level.
    model     choose auto regressive or neural network

    IMPORTANT: Rolling window procedure only for MULTI STEP computations
    """
    if model in ["r", "R", "ar", "Ar", "AR", "regression", "Regression"]:
        model = "r"
    elif model in ["n", "nn", "NN", "nN", "Nn", "neural network", "Neural Network"]:
        model = "n"
    else:
        model = "n"
        print("Neural network is chose as underlying forecasting procedure")

    if strategy in ["r", "R", "rec", "Rec", "recursive", "Recursive"]:
        strategy = "r"
        print("A recursive approach is used for multi step forecast")
    else:
        strategy = "f"
        print("A far horizon approach is used for multi step forecast")

    arr_MAE_Forecast = np.zeros((window, horizon))  # Save MAE
    arr_MASE_Forecast = np.zeros(window)  # Save MASE
    for i in range(window):  # Rolling window
        data_window = data[:-window - horizon + 1 + i]  # Data for current position - only multi step
        if model == "r" and strategy == "r":  # Choice of method
            fc = rrms(data_window, horizon, scales, ccps)
        elif model == "r" and strategy == "f":
            fc = rfms(data_window, horizon, scales, ccps)
        elif model == "n" and strategy == "r":
            fc = nnrms(data_window, horizon, scales, ccps)
        else:
            fc = nnfms(data_window, horizon, scales, ccps)

        # If else for index problems (last case would be 0, indexing not possible)
        if (i == (window - 1)):
            arr_T = data[-horizon:]  # For last step
        else:
            arr_T = data[:-window + 1 + i][-horizon:]  # For all horizon before the last

        ft_MAE = int_mae(arr_T, fc)  # Overall MAE
        arr_MAE = arr_mae(arr_T, fc)  # Timewise MAE
        n = 10  # Preparation MASE: Length MAE Randowm Walk
        arr_T = data[:-window - horizon + 2 + i][-n:]  # True value for position at forecast
        arr_RW = data[:-window - horizon + 1 + i][-n:]  # Using last value as forecast for next
        ft_MAE_rw = int_mae(arr_T, arr_RW)  # MAE random walk
        ft_MASE = ft_MAE / ft_MAE_rw  # MASE

        arr_MAE_Forecast[i] = arr_MAE  # Save MAE results
        arr_MASE_Forecast[i] = ft_MASE  # Save MASE results
    return arr_MAE_Forecast, arr_MASE_Forecast



















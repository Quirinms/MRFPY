import numpy as np
import pandas as pd
#from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from .MR import decomposition, training_scheme, far_horizon_training_scheme


def lsm_optimization(points_in_future, lsmatrix):
    """
    points_in_future    homogenous target vector b of training scheme Ax=b

    lsmatrix            matrix A for the training scheme Ax=b. contains the equations
                        according to a forecasting scheme
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Least Square Optimization Method for training
    # ----------------------------------------------------------------------------------------#
    weights, res, rank, s = np.linalg.lstsq(lsmatrix, points_in_future)
    return weights


def mle(weights, points_in_future, lsmatrix):
    """
    points_in_future    homogenous target vector b of training scheme Ax=b

    lsmatrix            matrix A for the training scheme Ax=b. contains the equations
                        according to a forecasting scheme

    weights             solution vector x to Ax=b
    """
    # y = A*x + eps => eps = y - A*x
    y, A, x = points_in_future, lsmatrix, weights
    arr_eps = points_in_future - np.dot(lsmatrix, weights)
    ft_MLE = np.sum(np.log(abs(arr_eps)))
    # ft_MLE = np.exp(-arr_eps)
    return ft_MLE


def mle_optimization(points_in_future, lsmatrix):
    """
    points_in_future    homogenous target vector b of training scheme Ax=b

    lsmatrix            matrix A for the training scheme Ax=b. contains the equations
                        according to a forecasting scheme
    """
    x0 = np.ones(lsmatrix.shape[1])
    MLE = minimize(fun=mle,
                   x0=x0,
                   args=(points_in_future, lsmatrix),
                   method="L-BFGS-B",  # Nelder-Mead L-BFGS-B
                   tol=10 ** (-5),
                   options={"disp": True})
    return MLE.x


def prediction_scheme(weights, wmatrix, rhwtCoeff, scales, ccps):
    """
    weights      np.darray containing training weights
    wmatrix      matrix which contains the wavelet coefficients
    rhwtCoeff    matrix which contains the transformed scales
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    """
    time = wmatrix.shape[1] - 1
    future_point = 0
    counter = 0
    # arrLSM = np.zeros(sum(ccps))

    for scale in range(scales + 1):
        for k in range(ccps[scale]):
            if scale != scales:
                index = time - ((k) * (2 ** (scale + 1)))
                future_point += weights[counter] * wmatrix[scale, index]
                # arrLSM[counter] = wmatrix[scale][index]
                counter += 1
            else:
                index = time - ((k) * (2 ** (scale + 1)))
                future_point += weights[counter] * rhwtCoeff[scale - 1, index]
                # arrLSM[counter] = rhwtCoeff[scale-1, index]
                counter += 1
    return future_point


def onestep(data, scales, ccps, opt="lsm"):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    opt          optimization method. Possible choices are "lsm" and "mle"
    """
    # Decomposition
    arrValues, wmatrix, rhwtCoeff, scales = decomposition(data, scales=scales)
    # Construct Wavelet Coefficient Selection Scheme for Training
    points_in_future, lsmatrix = training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps)
    # Optimization - training
    if opt == "mle":
        weights = mle_optimization(points_in_future, lsmatrix)
    else:
        weights = lsm_optimization(points_in_future, lsmatrix)

    # Use prediction sheme
    ftForecast = prediction_scheme(weights, wmatrix, rhwtCoeff, scales, ccps)
    return ftForecast


def far_horizon_onestep(data, step, scales, ccps, opt="lsm"):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    steps        forecast horizon
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    opt          optimization method. Possible choices are "lsm" and "mle"
    """
    # Decomposition
    arrValues, wmatrix, rhwtCoeff, scales = decomposition(data, scales=scales)
    # Construct Wavelet Coefficient Selection Scheme for Training
    if step == 1:
        points_in_future, lsmatrix = training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps)
    else:
        points_in_future, lsmatrix = far_horizon_training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps, step)
    # Optimization - training
    if opt == "mle":
        weights = mle_optimization(points_in_future, lsmatrix)
    else:
        weights = lsm_optimization(points_in_future, lsmatrix)

    # Use prediction sheme
    ftForecast = prediction_scheme(weights, wmatrix, rhwtCoeff, scales, ccps)
    return ftForecast

def recursive_multistep(data, steps, scales, ccps, opt = "lsm"):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    steps        forecast horizon
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    opt          optimization method. Possible choices are "lsm" and "mle"
    """
    # Data structures
    arrForecast = np.zeros(steps)
    # Call one step prediction multiple times
    for i in range(steps):
        ftForecast = onestep(data, scales, ccps, opt)
        arrForecast[i] = ftForecast           # Save all results
        data = np.append(data, ftForecast)    # Append recent forecast and update data
    return arrForecast

def far_horizon_multistep(data, steps, scales, ccps, opt = "lsm"):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    steps        forecast horizon
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    opt          optimization method. Possible choices are "lsm" and "mle"
    """
    # Data structures
    arrForecast = np.zeros(steps)
    # Call one step prediction multiple times
    for i in range(1, steps+1):
        ftForecast = far_horizon_onestep(data, i, scales, ccps, opt)
        arrForecast[i-1] = ftForecast           # Save all results
        data = np.append(data, ftForecast)    # Append recent forecast and update data
    return arrForecast






















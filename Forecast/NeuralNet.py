import numpy as np
from sklearn.neural_network import MLPRegressor
from .MR import decomposition, training_scheme, far_horizon_training_scheme


def prediction_scheme(wmatrix, rhwtCoeff, scales, ccps):
    """
    wmatrix      matrix which contains the wavelet coefficients
    rhwtCoeff    matrix which contains the transformed scales
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    """
    # Error capturing
    if len(ccps) != scales + 1:
        raise ValueError("Length of ccps must be number of scales from decomposition + 1")

    # Highest range of coefficients for constructing model equations
    minCut = np.zeros(scales+1)
    for i in range(scales+1):
        if i != scales:
            minCut[i] = ccps[i]*(2**(i+1))
        else:
            minCut[i] = ccps[i]*(2**(i))
    maxMinCut = np.max(minCut)    # Range needed for constructing model

    #----------------------------------------------------------------------------------------#
    #---- Create matrix for lsm Optimization
    #----------------------------------------------------------------------------------------#
    swt_requirement = rhwtCoeff.shape[1]                   # Length of swt dec
    # Consider need for coefficients to construct model and zero coefficients at start
    startTraining = int(2*maxMinCut)                       # Cut from start two times
    intNumEquations = int(swt_requirement-startTraining-1) # Number of equations for training
    numberWeights = sum(ccps)                              # Number of parameters
    lst_forecast_coeff = []

    time = wmatrix.shape[1]-1
    counter = 0
    for scale in range(scales + 1):
        for k in range(ccps[scale]):
            if scale != scales:
                index = time - ((k)*(2**(scale+1)))
                lst_forecast_coeff.append(wmatrix[scale, index])
                counter += 1
            else:
                index = time - ((k)*(2**(scale)))
                lst_forecast_coeff.append(rhwtCoeff[scale-1, index])
                counter += 1

    return lst_forecast_coeff

def onestep(data, scales, ccps):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    scales    Number of resolution levels to create. Remember that there are scales many wavelet
              levels plus one level for the smooth coefficients
    ccps      Contains the number of coefficients to use on each scale individually. Remember
              the plus one entry for the smooth level.
    """
    # Decomposition
    arrValues, wmatrix, rhwtCoeff, scales = decomposition(data, scales)
    # Wavelet Training Scheme
    points_in_future, lsmatrix = training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps)
    # Multi Layer Perceptron initiation (SKLearn)
    mra_MLP = MLPRegressor(hidden_layer_sizes=(10),
                           activation='relu',
                           solver='adam',
                           alpha=1e-5,
                           random_state=0)
    # Training
    mra_MLP.fit(lsmatrix, points_in_future)
    # Wavelet Forecast Scheme
    nn_onestep = prediction_scheme(wmatrix, rhwtCoeff, scales, ccps)
    # Onestep Forecast
    ftForecast = mra_MLP.predict([nn_onestep])[0]
    return ftForecast

def recursive_multistep(data, steps, scales, ccps):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    steps     forecast horizon
    scales    Number of resolution levels to create. Remember that there are scales many wavelet
              levels plus one level for the smooth coefficients
    ccps      Contains the number of coefficients to use on each scale individually. Remember
              the plus one entry for the smooth level.
    """
    # Data structures
    arrForecast = np.zeros(steps)
    # Call one step prediction multiple times
    for i in range(steps):
        ftForecast = onestep(data, scales, ccps)
        arrForecast[i] = ftForecast           # Save all results
        data = np.append(data, ftForecast)    # Append recent forecast and update data
    return arrForecast

def far_horizon_onestep(data, steps, scales, ccps):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    scales    Number of resolution levels to create. Remember that there are scales many wavelet
              levels plus one level for the smooth coefficients
    ccps      Contains the number of coefficients to use on each scale individually. Remember
              the plus one entry for the smooth level.
    """
    # Decomposition
    arrValues, wmatrix, rhwtCoeff, scales = decomposition(data, scales)
    # Wavelet Training Scheme
    if steps == 1:
        points_in_future, lsmatrix = training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps)
    else:
        points_in_future, lsmatrix = far_horizon_training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps, steps = 2)
    # Multi Layer Perceptron initiation (SKLearn)
    mra_MLP = MLPRegressor(hidden_layer_sizes=(10),
                           activation='relu',
                           solver='adam',
                           alpha=1e-5,
                           random_state=0)
    # Training
    mra_MLP.fit(lsmatrix, points_in_future)
    # Wavelet Forecast Scheme
    nn_onestep = prediction_scheme(wmatrix, rhwtCoeff, scales, ccps)
    # Onestep Forecast
    ftForecast = mra_MLP.predict([nn_onestep])[0]
    return ftForecast

def far_horizon_multistep(data, steps, scales, ccps):
    """
    data      pd.DataFrame or np.ndarray with one dimension, containing only values of time series
    steps     forecast horizon
    scales    Number of resolution levels to create. Remember that there are scales many wavelet
              levels plus one level for the smooth coefficients
    ccps      Contains the number of coefficients to use on each scale individually. Remember
              the plus one entry for the smooth level.
    """
    # Data structures
    arrForecast = np.zeros(steps)
    # Call one step prediction multiple times
    for i in range(1, steps+1):
        ftForecast = far_horizon_onestep(data, i, scales, ccps)
        arrForecast[i-1] = ftForecast           # Save all results
        data = np.append(data, ftForecast)    # Append recent forecast and update data
    return arrForecast





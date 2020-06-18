import numpy as np
import pandas as pd


def decomposition(data, scales=1):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    arrValues = None
    if type(data) == pd.DataFrame:
        if type(data.iloc[0, 1]) != np.float64:
            raise ValueError("If data is pandas.DataFrame, the second column must carry the \
                             data and be of type numpy.float64")
        arrValues = np.array(data.iloc[:, 1])

    elif type(data) == np.ndarray:
        if len(data.shape) != 1:
            raise ValueError("If data is numpy.ndarray, it must be 1-dimensional")
        arrValues = data
    else:
        raise ValueError("Data must be of type pandas.DataFrame or numpy.ndarray")

    # ----------------------------------------------------------------------------------------#
    # ---- Initialization and Error capturing
    # ----------------------------------------------------------------------------------------#
    intLenTS = len(arrValues)  # Length timeseries is N    - np.array 1:N

    # Requirement for training
    numPoints = scales  # Number of points lost for building each level
    numMaxRange = 2 ** scales  # Number of points needed for training biggest scale
    rangeScales = numMaxRange + numPoints  # Required number of swt coefficients on lowest scale

    if (intLenTS < rangeScales):
        raise ValueError("Length of Data is no sufficient for given level")

    arrLB = np.zeros(scales)  # Array for indicating sizes needed for each level
    for i in range(scales):
        arrLB[i] = 2 ** i

    start = 2 ** scales  # Time points needed for construction coeffs
    rhwtCoeff = np.zeros((scales, intLenTS))  # Smooth coefficients
    wmatrix = np.zeros((scales, intLenTS))  # Wavelet coefficients (Differences)
    # ----------------------------------------------------------------------------------------#
    # ---- Redundant haar wavelet transform for prediction
    # ----------------------------------------------------------------------------------------#
    for level in range(0, scales):  # Build all smooth level
        for time in range(0, intLenTS):  # Consider all possible time points
            if time >= int(sum(arrLB[1:level + 1]) + 1):  # Care for loss of time points f.e. level
                lvl = level - 1
                idx = time - (2 ** level)
                if level == 0:
                    rhwtCoeff[level, time] = 0.5 * (arrValues[idx] + arrValues[time])
                else:
                    rhwtCoeff[level, time] = 0.5 * (rhwtCoeff[lvl, idx] + rhwtCoeff[lvl, time])

    for time in range(intLenTS):  # Consider all possible time points
        for scale in range(scales):  # Build all difference level
            if scale == 0:
                wmatrix[scale, time] = arrValues[time] - rhwtCoeff[scale, time]
            else:
                wmatrix[scale, time] = rhwtCoeff[scale - 1, time] - rhwtCoeff[scale, time]
    # Following equation must be true
    # sum(wmatrix[:, idx]) + rhwtCoeff[scales-1, idx] = signal[idx] - idx = 0, ..., len(arrValues)-1
    return arrValues, wmatrix, rhwtCoeff, scales


def training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps):
    """
    arrValues    pd.DataFrame or np.ndarray with one dimension
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
    minCut = np.zeros(scales + 1)
    for i in range(scales + 1):
        if i != scales:
            minCut[i] = ccps[i] * (2 ** (i + 1))
        else:
            minCut[i] = ccps[i] * (2 ** (i))
    maxMinCut = np.max(minCut)  # Range needed for constructing model

    # ----------------------------------------------------------------------------------------#
    # ---- Create matrix with equations according to forecasting scheme
    # ----------------------------------------------------------------------------------------#
    swt_requirement = rhwtCoeff.shape[1]  # Length of swt dec
    # Consider need for coefficients to construct model and zero coefficients at start
    startTraining = int(2 * maxMinCut)  # Cut from start two times
    intNumEquations = int(swt_requirement - startTraining - 1)  # Number of equations for training
    numberWeights = sum(ccps)  # Number of parameters
    lsmatrix = np.zeros((intNumEquations, numberWeights), dtype='float')  # Matrix for equations

    for i in range(intNumEquations):
        time = startTraining + i
        counter = 0
        for scale in range(scales + 1):
            for k in range(ccps[scale]):
                if scale != scales:
                    index = time - ((k) * (2 ** (scale + 1)))
                    lsmatrix[i, counter] = wmatrix[scale, index]
                    counter += 1
                else:
                    index = time - ((k) * (2 ** (scale)))
                    lsmatrix[i, counter] = rhwtCoeff[scale - 1, index]
                    counter += 1

    points_in_future = arrValues[startTraining + 1:]

    return points_in_future, lsmatrix


def far_horizon_training_scheme(arrValues, wmatrix, rhwtCoeff, scales, ccps, steps=2):
    """
    arrValues    pd.DataFrame or np.ndarray with one dimension
    wmatrix      matrix which contains the wavelet coefficients
    rhwtCoeff    matrix which contains the transformed scales
    scales       Number of resolution levels to create. Remember that there are scales many wavelet
                 levels plus one level for the smooth coefficients
    ccps         Contains the number of coefficients to use on each scale individually. Remember
                 the plus one entry for the smooth level.
    steps        forecast horizon
    """
    # Error capturing
    if len(ccps) != scales + 1:
        raise ValueError("Length of ccps must be number of scales from decomposition + 1")

    # Highest range of coefficients for constructing model equations
    minCut = np.zeros(scales + 1)
    for i in range(scales + 1):
        if i != scales:
            minCut[i] = ccps[i] * (2 ** (i + 1))
        else:
            minCut[i] = ccps[i] * (2 ** (i))
    maxMinCut = np.max(minCut)  # Range needed for constructing model

    # ----------------------------------------------------------------------------------------#
    # ---- Create matrix with equations according to forecasting scheme
    # ----------------------------------------------------------------------------------------#
    swt_requirement = rhwtCoeff.shape[1]  # Length of swt dec
    # Consider need for coefficients to construct model and zero coefficients at start
    startTraining = int(2 * maxMinCut)  # Cut from start two times
    intNumEquations = int(swt_requirement - startTraining - 1)  # Number of equations for training
    intNumEquations = intNumEquations - steps + 1
    numberWeights = sum(ccps)  # Number of parameters
    lsmatrix = np.zeros((intNumEquations, numberWeights), dtype='float')  # Matrix for equations

    for i in range(intNumEquations):
        time = startTraining + i
        counter = 0
        for scale in range(scales + 1):
            for k in range(ccps[scale]):
                if scale != scales:
                    index = time - ((k) * (2 ** (scale + 1)))
                    lsmatrix[i, counter] = wmatrix[scale, index]
                    counter += 1
                else:
                    index = time - ((k) * (2 ** (scale)))
                    lsmatrix[i, counter] = rhwtCoeff[scale - 1, index]
                    counter += 1

    points_in_future = arrValues[-intNumEquations:]

    return points_in_future, lsmatrix




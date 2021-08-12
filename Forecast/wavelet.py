import numpy as np
import pandas as pd


def decomposition(data, aggregation = np.array([2,4]), threshold=False, thresholdStrategy="hard", thresholdLambda=0.05):
    """
    data            np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
                    second column with values
    aggregation     Number of resolution levels to create. Remember that there are scales many wavelet
                    levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    if type(data) == pd.DataFrame:
        print("Given data is pd.DataFrame. Values of time series are assumed to be in second column."
              "First column is meant for time")
        data = np.array(data.iloc[:, 1])

    if type(data) is not np.ndarray:
        print("Given data is not of type ndarray.")
        return
    if type(aggregation) is not np.ndarray:
        print("Given aggregation is not of type ndarray.")
        return
    if len(aggregation.shape) != 1:
        print("Dimension of given aggregations does not equal one.")
        return
    # ----------------------------------------------------------------------------------------#
    # ---- Decomposition
    # ----------------------------------------------------------------------------------------#
    intLenTS = len(data)
    scales = len(aggregation)
    smooth_coeff = dynamic_aggregation(data, aggregation)
    wavelet_coeff = compute_wavelet_matrix(data, intLenTS, scales, smooth_coeff)
    if threshold == True:
        my_threshold_array = get_threshold_array(wavelet_coeff, thresholdLambda)
        if thresholdStrategy == "hard":
            wavelet_coeff = hard_thresholding(wavelet_coeff, my_threshold_array)
        else:
            wavelet_coeff = soft_thresholding(wavelet_coeff, my_threshold_array)
    return wavelet_coeff, smooth_coeff, scales


def first_level_aggregation(data, time, number_aggregation_points):
    aggregation_sum = 0
    for i in range(number_aggregation_points):  # Keine - 1! => Testen
        aggregation_sum = aggregation_sum + data[time - i]
    averaged_aggregation = aggregation_sum / number_aggregation_points
    return averaged_aggregation


def dynamic_aggregation(data, aggregation):
    intLenTS     = len(data)
    scales       = len(aggregation)
    smooth_coeff = np.zeros((scales, intLenTS))         # Smooth coefficients
    for level in range(scales):                         # Compute smooth approximation for scales many levels
        start = aggregation[level] - 1                  # - 1 for 0-Indexing (Python)
        for time in range(intLenTS):                    #
            if time >= start:                           # Consider the required offset for such smoothing depending on how many points need to be aggregated
                smooth_coeff[level, time] = first_level_aggregation(data, time, aggregation[level])
    return smooth_coeff


def compute_wavelet_matrix(data, intLenTS, scales, smooth_coeff):
    wavelet_matrix = np.zeros((scales, intLenTS))
    for time in range(intLenTS):
        for scale in range(scales):
            if scale == 0:
                wavelet_matrix[0, time] = data[time] - smooth_coeff[0, time]
            else:
                wavelet_matrix[scale, time] = smooth_coeff[scale-1, time] - smooth_coeff[scale, time]
    return wavelet_matrix


def get_threshold_array(wavelet_coeff, thresholdLambda):
    if type(thresholdLambda) != np.ndarray:
        num_scales = np.shape(wavelet_coeff)[0]
        tmp_var = np.zeros(num_scales)
        for i in range(num_scales):
            tmp_var[i] = np.percentile(abs(wavelet_coeff[i,:]), thresholdLambda*100)
            #tmp_var[i] = thresholdLambda * np.max(wavelet_coeff[i,:])
        thresholdLambda = tmp_var
    return thresholdLambda


def hard_thresholding(wavelet_coeff, thresholdLambda):
    first_dim = wavelet_coeff.shape[0]
    second_dim = wavelet_coeff.shape[1]
    for i in range(first_dim):
        for j in range(second_dim):
            wt = abs(wavelet_coeff[i,j])    # Get absolute value of the value coefficient
            if wt < thresholdLambda[i]:     # Absolute value of WT coefficient smaller than threshold?
                wavelet_coeff[i,j] = 0      # Hard threshold: if below threshold, make it zero
    return wavelet_coeff


def soft_thresholding(wavelet_coeff, thresholdLambda):
    first_dim = wavelet_coeff.shape[0]
    second_dim = wavelet_coeff.shape[1]
    for i in range(first_dim):
        for j in range(second_dim):
            wt = wavelet_coeff[i,j]        # Get value of the value coefficient
            if abs(wt) < thresholdLambda[i]:# Absolute value of WT coefficient smaller than threshold?
                wavelet_coeff[i,j] = 0      # Soft threshold: if below threshold, make it zero
            else:                           # Soft threshold: if above threshold, subtract a lambda
                wavelet_coeff[i, j] = np.sign(wt)*(np.abs(wt)-thresholdLambda[i])
    return wavelet_coeff


def training(UnivariateData, wavelet_coeff, smooth_coeff, scales, coeff_selection, aggregation,
             MultivariateData=None, NumMV=1):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    if ((type(UnivariateData) != np.ndarray) or (UnivariateData.ndim != 1)):
        print("Given univariate data needs to be a 1 dimensional Numpy array.")
    if ((type(MultivariateData) == np.ndarray) and (type(MultivariateData) == pd.DataFrame)):
        MultivariateData = np.matrix(MultivariateData)

    coeff_selection = coeff_selection.astype("int")
    # ----------------------------------------------------------------------------------------#
    # ---- Training
    # ----------------------------------------------------------------------------------------#
    maxConLen     = np.max(aggregation)-1                                               # Req. to construct decompositon
    tmpVAR        = np.append(aggregation, aggregation[len(aggregation) - 1])
    maxReqLen     = np.max(coeff_selection * tmpVAR)                                    # Req. to construct model
    startTraining = maxReqLen + maxConLen                                               # Offset at start of time series
    len_data      = len(UnivariateData)                                                 # Length of swt dec
    intNumEq      = int(round((len_data - startTraining - 1), 0))                       # Number of equations available for training
    numberWeights = np.sum(coeff_selection)                                             # Number of equations needed
    if numberWeights > intNumEq:                                                        # There must be enough
        raise ValueError("There are not enough equations for training."                 # equations, otherwise no
                         "Your time series is too short!")                              # optimization is performed
    # training_X:
    # Rows are number of equations necessary for the model
    # Columns are selected coefficients
    training_X = np.zeros((intNumEq, numberWeights))
    for i in range(intNumEq):
        time = startTraining + i                                                          # Start from offset, see above
        counter = 0                                                                       # 0-Indexing (Python)
        for s in range(len(coeff_selection)):                                             # Compute equations for all necessary scales
            for k in range(coeff_selection[s]):                                           # Enable as many coefficients per scale as desired
                if s != (len(coeff_selection)-1):                                         # Treat all wavelet coefficients the same, but smooth part (the last coefficient) different
                    index = int(time - k * aggregation[s])
                    training_X[i, counter] = wavelet_coeff[s, index]
                else:
                    index = int(time - k * aggregation[s-1])
                    training_X[i, counter] = smooth_coeff[s-1, index]
                counter = counter + 1
    # training_X:
    # Columns must be added or rows must be extended
    # training_X:
    # Columns must be added or rows must be extended
    if type(MultivariateData) == np.ndarray:
        if NumMV == 1:
            multivariateEquations = np.array(MultivariateData[-intNumEq:, :])
        else:
            multivariateEquations = np.array(MultivariateData[-intNumEq:, :])
            for i in range(1,NumMV):
                xDayData = MultivariateData[(-intNumEq-i):-i, :]
                multivariateEquations = np.concatenate((multivariateEquations, xDayData), axis = 1)
        training_X = np.concatenate((training_X, multivariateEquations), axis = 1)
    #training_forecasts = data[(len(data)-intNumEq):(len(data)-1)]
    training_Y = UnivariateData[(len_data-intNumEq):]      # Vector containing the predicted values for training
    return training_X, training_Y


def prediction_scheme(wavelet_coeff, smooth_coeff, coeff_selection, aggregation, MultivariateData=None, NumMV=1):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    if ((type(MultivariateData) == np.ndarray) and (type(MultivariateData) == pd.DataFrame)):
        MultivariateData = np.matrix(MultivariateData)
    coeff_selection = coeff_selection.astype("int")
    # ----------------------------------------------------------------------------------------#
    # ---- Prediction for regression
    # ----------------------------------------------------------------------------------------#
    time         = wavelet_coeff.shape[1] - 1
    len_future_point = 0
    #for i in range(len(coeff_selection)):
    #    for j in range(coeff_selection[i]):
    #        len_future_point = len_future_point + 1
    len_future_point = np.sum(coeff_selection)
    if type(MultivariateData) == np.ndarray:
        if type(NumMV) != int:
            NumMV = 1
        if NumMV < 1:
            NumMV = 1
        len_future_point = len_future_point + MultivariateData.shape[1]*NumMV
    future_point = np.zeros(len_future_point)
    counter      = 0                           # 0-Indexing (Python)
    scales       = len(aggregation)            #
    for s in range(len(coeff_selection)):      # All wavelet scales + last smooth scale
        for k in range(coeff_selection[s]):    #
            if s != scales:
                index                 = int(time - k * aggregation[s])
                future_point[counter] = wavelet_coeff[s, index]
            else:
                index                 = int(time - k * aggregation[s - 1])
                future_point[counter] = smooth_coeff[s - 1, index]
            counter = counter + 1
    if type(MultivariateData) == np.ndarray:
        numExplanatoryVariables = MultivariateData.shape[1]
        multivariateRegressor   = MultivariateData[-NumMV:, :]
        # Watch out for the order: It must be corresponding to training() (see above)
        # Most recent variables are at the end of the matrix structure, whereas oldest data is at the start
        for i in range(NumMV - 1, -1, -1):
            for j in range(numExplanatoryVariables):
                future_point[counter] = multivariateRegressor[i,j]
                counter = counter + 1
    return future_point
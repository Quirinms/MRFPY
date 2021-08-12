#----------------------------------------------------------------------------------------------------------------------#
# Arrays + DataFrame -> Objects and Operations
#----------------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------------------------------------#
# Parallel computation
#----------------------------------------------------------------------------------------------------------------------#
import multiprocessing
from functools import partial

#----------------------------------------------------------------------------------------------------------------------#
# Wavelet Forecasting Framework
#----------------------------------------------------------------------------------------------------------------------#
from MRFPY.forecast import multistep


"""
Modellselektion MCMC Varian Scott
Modellselektion Warmstart für evolutionäre optimierung
"""


def rollingForecastingOrigin(UnivariateData, CoefficientCombination, Aggregation, Horizon=2, Window=2, Method="r",
                             MultivariateData=None, NumMV=1, Sparse=False, SparseCoverage=0.5, numClusters=1,
                             hidden_layer_sizes=8, activation="relu", solver="lbfgs", alpha=1e-5, n_estimators=100,
                             criterion="mae",
                             max_depth=3, loss="ls", learning_rate=0.1, subsample=0.9, random_state=0, strategy="mean",
                             max_iter=5000, Threshold=False, ThresholdStrategy="hard", ThresholdLambda=0.05):
    """
    DESCRIPTION
    Rolling Forecasting Origin method for multiresolution forecasting

    INPUT
    UnivariateData      np.ndarray
    CoefficientCombination
    Aggregation

    OPTIONAL
    Horizon             Int integer
    Window              Int integer
    Method              Int integer
    MultivariateData    np.ndarray numerical array
    NumMV               Int integer
    Sparse              bool boolean
    SparseCoverage      np.float numerical value
    numClusters         Int integer
    hidden_layer_sizes  Int integer
    activation
    solver
    alpha
    n_estimators
    criterion
    max_depth
    loss
    learning_rate
    subsample
    random_state
    strategy
    max_iter
    Threshold
    ThresholdStrategy
    ThresholdLambda

    OUTPUT
    mat_error           np.ndarray
    mat_forecast        np.ndarray


    """
    InternMethodName = "rollingForecastingOrigin.py"
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    if type(UnivariateData) != np.ndarray:
        Message = ": UnivariateData must be of type np.ndarray!"
        print(InternMethodName + Message)
        return 0, 0

    # ----------------------------------------------------------------------------------------#
    # ---- Error capturing
    # ----------------------------------------------------------------------------------------#
    if Horizon < 1:
        Message = ": Horizon must be greater than 0!"
        print(InternMethodName + Message)
        return 0, 0

    if Window < 2:
        Message = ": Window must be greater or equal 2!"
        print(InternMethodName + Message)
        return 0, 0


    #if type(MultivariateData) != np.ndarray:
    #    return 0, 0
    # ----------------------------------------------------------------------------------------#
    # ---- Rolling Forecasting Origin
    # ----------------------------------------------------------------------------------------#
    if Sparse == False:
        WindowRange = np.linspace(0, Window-1, Window)
    else:             # If Sparse => sample part of the given window => faster computation
        SC = int(SparseCoverage * Window)
        if SC < 2:    # Do not let the sparse window become to small
            WindowRange = np.linspace(0, Window - 1, Window)
        else:         # If too small => use old window (should be small enough then)
            WindowRange = np.linspace(0, Window - 1, SC)
    WindowRange = np.array(np.round(WindowRange), dtype="int")

    LengthRFO = len(WindowRange)
    Model = None
    mat_error = np.zeros((LengthRFO, Horizon))
    mat_forecast = np.zeros((LengthRFO, Horizon))
    # Simple non parallel implementation
    if numClusters == 1:
        intLen = len(UnivariateData)  # Length time series
        counter = 0
        for i in WindowRange:
            int_Index = intLen - Window - Horizon + i  # Current Forecast Position
            dfTrain   = UnivariateData[0:(int_Index+1)]
            dfTest    = UnivariateData[(int_Index+1):(int_Index+1+Horizon)]
            forecast, Model = multistep(UnivariateData=dfTrain, Horizon=Horizon,
                                        CoefficientCombination=CoefficientCombination, Aggregation=Aggregation,
                                        Method=Method, MultivariateData=MultivariateData, NumMV=NumMV,
                                        hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                        alpha=alpha, n_estimators=n_estimators, criterion=criterion,
                                        max_depth=max_depth, loss=loss, learning_rate=learning_rate,
                                        subsample=subsample, random_state=random_state, strategy=strategy,
                                        max_iter=max_iter, Model=Model, threshold=Threshold,
                                        thresholdStrategy=ThresholdStrategy, thresholdLambda=ThresholdLambda)
            arr_Error = np.array(dfTest - forecast)
            mat_error[counter] = arr_Error
            mat_forecast[counter] = np.array(dfTest)
            counter = counter + 1
    # Parallel implementation
    else:
        # Get clusters
        # import os
        numCores = multiprocessing.cpu_count()
        if numCores == None:
            raise ValueError("Number of available cores is None!")
        numCores = numCores - 1                                              # Leave computational power for sys ops
        if((numClusters == "max") or (numClusters > numCores)):
            numClusters = numCores
        pool      = multiprocessing.Pool(numClusters)
        rollingFO = partial(rFO_single,                                      # Rolling Forecasting Origin
                            UnivariateData,                                  # Time Series Forecasting Task (Univariate)
                            CoefficientCombination, Aggregation,             # Model selection part
                            Horizon, Window,                                 # TimeSeriesFc. Task (Multistep + Crossval)
                            Method,                                          # Method for processing features
                            MultivariateData, NumMV,                         # TimeSeriesForecasting Task (Multivariate)
                            hidden_layer_sizes, activation, solver, alpha,   # Hyperparameters
                            n_estimators, criterion, max_depth, loss,        # Hyperparameters
                            learning_rate,                                   # Hyperparameters
                            subsample, random_state, strategy, max_iter,     # Hyperparameters
                            Model,                                           # Model reuse for parallel not possible rn
                            Threshold, ThresholdStrategy, ThresholdLambda    # Wavelet thresholding
                            )                                                # Parallelization over var i

        result    = np.array(pool.map(rollingFO, WindowRange))
        pool.close()

        if len(result) == 0:
            print("Result is of length zero!")
            return 0,0


        mat_error    = result[:, 0]
        mat_forecast = result[:, 1]
        #results = np.array(Parallel
        #                   (n_jobs = numClusters)
        #                   (delayed(rFO_single)
        #                       (data, coeff_selection, aggregation, horizon, window, method, i) for i in range(window)))
        """
    lst_forecast = parallel::parLapply(cl, 1:window_size, rolling_window_single,
                                       data = data, ccps = ccps,
                                       agg_per_lvl = agg_per_lvl,
                                       horizon = horizon,
                                       window_size = window_size,
                                       method = method)
        """
    return mat_error, mat_forecast


def rFO_single(UnivariateData, CoefficientCombination, Aggregation, Horizon, Window, Method, MultivariateData, NumMV,
               hidden_layer_sizes, activation, solver, alpha, n_estimators, criterion, max_depth, loss, learning_rate,
               subsample, random_state, strategy, max_iter, Model, Threshold, ThresholdStrategy, ThresholdLambda, i):
    InternMethodName = "rFO_single.py (subroutine of rollingForecastingOrigin.py for parallel computation)"
    if type(UnivariateData) != np.ndarray:
        Message = ": UnivariateData is not of type np.ndarray!"
        print(InternMethodName + Message)
        return

    if type(CoefficientCombination) != np.ndarray:
        Message = ": CoefficientCombination is not of type np.ndarray!"
        print(InternMethodName + Message)
        return

    if type(Aggregation) != np.ndarray:
        Message = ": Aggregation is not of type np.ndarray!"
        print(InternMethodName + Message)
        return

    if type(Horizon) != np.int:
        Message = ": Horizon is not of type np.int!"
        print(InternMethodName + Message)
        return

    if type(Method) != str:
        Message = ": Method is not of type str!"
        print(InternMethodName + Message)
        return

    if type(MultivariateData) != np.ndarray:
        if str(type(MultivariateData)) != "<class 'NoneType'>":
            Message = ": MultivariateData is neither np.ndarray (multivariate case) nor of type NoneType (univariate case)!"
            print(InternMethodName + Message)
            return

    if type(NumMV) != np.int:
        Message = ": NumMV is not of type np.int!"
        print(InternMethodName + Message)
        return

    intLen    = len(UnivariateData)  # Length time series
    int_Index = intLen - Window - Horizon + i  # Current Forecast Position
    dfTrain   = UnivariateData[0:(int_Index + 1)]
    dfTest    = UnivariateData[(int_Index + 1):(int_Index + 1 + Horizon)]
    forecast, Model  = multistep(UnivariateData=dfTrain,
                                 Horizon=Horizon, CoefficientCombination=CoefficientCombination,
                                 Aggregation=Aggregation, Method=Method, MultivariateData=MultivariateData, NumMV=NumMV,
                                 hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                 alpha=alpha, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                 loss=loss, learning_rate=learning_rate, subsample=subsample, random_state=random_state,
                                 strategy=strategy, max_iter=max_iter, Model=Model, threshold=Threshold,
                                 thresholdStrategy=ThresholdStrategy, thresholdLambda=ThresholdLambda)
    arr_Error = np.array(dfTest - forecast)
    return arr_Error, dfTest


def in_sample_forecast(data, aggregation):
    return

#
#
#
#
#

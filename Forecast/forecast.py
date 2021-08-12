#----------------------------------------------------------------------------------------------------------------------#
# Author: Quirin Stier
# Date: 07.05.2021
# Wavelet Forecasting Framework
# One-step and multi-step forecasts with wavelets
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# Arrays + DataFrame -> Objects and Operations
#----------------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------------------------------------#
# Parallel computation
#----------------------------------------------------------------------------------------------------------------------#
#import multiprocessing
#from functools import partial

#----------------------------------------------------------------------------------------------------------------------#
# Wavelet Framework
#----------------------------------------------------------------------------------------------------------------------#
from MRFPY.wavelet import decomposition
from MRFPY.wavelet import prediction_scheme
from MRFPY.wavelet import training

#----------------------------------------------------------------------------------------------------------------------#
# Regression tools from scikit learn
#----------------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


"""
Modellselektion MCMC Varian Scott
Modellselektion Warmstart für evolutionäre optimierung
"""


def adaBoostRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData=None, NumMV=1, loss="linear",
                      random_state=0, Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    CoefficientCombination = CoefficientCombination.astype("int")
    if loss not in ["linear", "square", "exponential"]:
        print("Given loss is not valid for this method. Setting to default 'linear'.")
        loss = "linear"
    # ----------------------------------------------------------------------------------------#
    # ---- Multilayer Perceptron onestep
    # ----------------------------------------------------------------------------------------#
    scales = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData, NumMV)
    if Model is None:
        ABR = AdaBoostRegressor(loss=loss, random_state=random_state)
    ABR.fit(Training_X, Training_Y)
    predictionInput = prediction_scheme(WaveletCoefficients, SmoothCoefficients, CoefficientCombination,
                                        Aggregation, MultivariateData, NumMV)
    Forecast        = ABR.predict([predictionInput])[0]                                # Onestep Forecast
    return Forecast, Model


def baggingRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData=None, NumMV=1, n_estimators=10,
                     random_state=0, Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    CoefficientCombination = CoefficientCombination.astype("int")
    # ----------------------------------------------------------------------------------------#
    # ---- Multilayer Perceptron onestep
    # ----------------------------------------------------------------------------------------#
    scales = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData, NumMV)
    if Model is None:
        BR = BaggingRegressor(n_estimators=n_estimators, random_state=random_state)
    BR.fit(Training_X, Training_Y)
    predictionInput = prediction_scheme(WaveletCoefficients, SmoothCoefficients, CoefficientCombination,
                                        Aggregation, MultivariateData, NumMV)
    Forecast        = BR.predict([predictionInput])[0]                                # Onestep Forecast
    return Forecast, Model


def dummyRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData=None, NumMV=1, strategy="mean",
                   Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    CoefficientCombination = CoefficientCombination.astype("int")
    # ----------------------------------------------------------------------------------------#
    # ---- Multilayer Perceptron onestep
    # ----------------------------------------------------------------------------------------#
    scales = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData)
    if Model is None:
        DR = DummyRegressor(strategy=strategy)
    DR.fit(Training_X, Training_Y)
    predictionInput = prediction_scheme(WaveletCoefficients, SmoothCoefficients, CoefficientCombination,
                                        Aggregation, MultivariateData)
    Forecast        = DR.predict([predictionInput])[0]                                # Onestep Forecast
    return Forecast, Model


def gradientBoostingRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData=None, NumMV=1, loss="ls",
                              learning_rate=0.1, n_estimators=100, subsample=0.9, criterion="mae", max_depth=2,
                              random_state=0, Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    CoefficientCombination = CoefficientCombination.astype("int")
    if loss not in ["ls", "lad", "huber", "quantile"]:
        print("Given loss is not valid for this method. Setting to default 'ls'.")
        loss = "ls"
    if criterion not in ["friedman_mse", "mae", "mse"]:
        print("Given loss is not valid for this method. Setting to 'mae'.")
        criterion = "mae"
    # ----------------------------------------------------------------------------------------#
    # ---- Multilayer Perceptron onestep
    # ----------------------------------------------------------------------------------------#
    scales = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData)
    if Model is None:
        GBR = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                        subsample=subsample, criterion=criterion, max_depth=max_depth,
                                        random_state=random_state)
    GBR.fit(Training_X, Training_Y)
    predictionInput = prediction_scheme(WaveletCoefficients, SmoothCoefficients, CoefficientCombination,
                                        Aggregation, MultivariateData)
    Forecast        = GBR.predict([predictionInput])[0]                                # Onestep Forecast
    return Forecast, Model


def mlpRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData=None, NumMV=1, hidden_layer_sizes=8,
                 activation="relu", solver="lbfgs", alpha=1e-5, random_state=0, max_iter=5000, Model=None,
                 threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    CoefficientCombination = CoefficientCombination.astype("int")
    # ----------------------------------------------------------------------------------------#
    # ---- Multilayer Perceptron onestep
    # ----------------------------------------------------------------------------------------#
    scales = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData, NumMV)

    # 3 different possibilities of training model and reuse existing results: All the same error performance
    # Runtime is different!
    #
    # Reuse model if possible and train with full set of training data
    # Runtime: 171.7 s (with RFO (single processing) on ENTSOE for Horizon = 14 and Window = 36)
    if Model is None:
        Model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                             max_iter=max_iter, random_state=random_state)
    Model.fit(Training_X, Training_Y)
    # Reuse model if possible
    # IF None then create new model and train with full training data
    # ELSE reuse and only use latest data (one equation) in order to update old model
    # Runtime: 203.3 s (with RFO (single processing) on ENTSOE for Horizon = 14 and Window = 36)
    """
    if Model is None:
        Model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                             max_iter=max_iter, random_state=random_state)
        Model.fit(Training_X, Training_Y)
    else:
        Model.fit(Training_X[-1:, :], Training_Y[-1:])
    """
    # Create new model and train with full set of training data
    # Runtime: 296.8 s (with RFO (single processing) on ENTSOE for Horizon = 14 and Window = 36)
    """
    Model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
                         max_iter=max_iter, random_state=random_state)
    Model.fit(Training_X, Training_Y)
    """
    predictionInput                                 = prediction_scheme(WaveletCoefficients, SmoothCoefficients,
                                                                        CoefficientCombination, Aggregation,
                                                                        MultivariateData, NumMV)
    Forecast = Model.predict([predictionInput])[0]         # Onestep Forecast
    return Forecast, Model


def multiLinearRegression(UnivariateData, CoefficientCombination, Aggregation, MultivariateData = None, NumMV=1,
                          Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
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
    CoefficientCombination = CoefficientCombination.astype("int")
    # ----------------------------------------------------------------------------------------#
    # ---- (Auto-/Multilinear-) Regression One-Step Forecast
    # ----------------------------------------------------------------------------------------#
    scales                                          = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData, NumMV)
    Weights, res, rank, s                           = np.linalg.lstsq(Training_X, Training_Y, rcond=None)
    #Forecast = prediction(Weights, WaveletCoefficients, SmoothCoefficients,
    #                      CoefficientCombination, Aggregation, MultivariateData)
    predscheme = prediction_scheme(WaveletCoefficients, SmoothCoefficients,
                                   CoefficientCombination, Aggregation, MultivariateData, NumMV)
    Forecast   = np.dot(predscheme, Weights)
    return Forecast, Model


def randomForestRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData=None, NumMV=1, n_estimators=100,
            criterion="mae", max_depth=3, random_state=0, Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.2):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#
    CoefficientCombination = CoefficientCombination.astype("int")
    if criterion not in ["mae", "mse"]:
        print("Given loss is not valid for this method. Setting to 'mae'.")
        criterion = "mae"
    # ----------------------------------------------------------------------------------------#
    # ---- Multilayer Perceptron onestep
    # ----------------------------------------------------------------------------------------#
    scales = len(Aggregation)
    WaveletCoefficients, SmoothCoefficients, Scales = decomposition(UnivariateData, Aggregation, threshold,
                                                                    thresholdStrategy, thresholdLambda)
    Training_X, Training_Y                          = training(UnivariateData, WaveletCoefficients, SmoothCoefficients,
                                                               Scales, CoefficientCombination, Aggregation,
                                                               MultivariateData)
    if Model is None:
        randForest = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                           random_state=random_state)
    randForest.fit(Training_X, Training_Y)
    predictionInput = prediction_scheme(WaveletCoefficients, SmoothCoefficients, CoefficientCombination,
                                        Aggregation, MultivariateData)
    Forecast        = randForest.predict([predictionInput])[0]        # Onestep Forecast
    return Forecast, Model


def onestep(UnivariateData, CoefficientCombination, Aggregation, Method="r", MultivariateData=None, NumMV=1,
            hidden_layer_sizes=8, activation="relu", solver="lbfgs", alpha=1e-5, n_estimators=100, criterion="mae",
            max_depth=3, loss="ls", learning_rate=0.1, subsample=0.9, random_state=0, strategy="mean", max_iter=5000,
            Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.05):
    if Method in ["r", "R", "Regression", "regression"]:
        OneStepForecast, Model = multiLinearRegression(UnivariateData, CoefficientCombination, Aggregation,
                                                       MultivariateData, NumMV, threshold=threshold,
                                                       thresholdStrategy=thresholdStrategy,
                                                       thresholdLambda=thresholdLambda)
    elif Method in ["mlp", "MLP", "MultilayerPerceptron"]:
        OneStepForecast, Model = mlpRegressor(UnivariateData, CoefficientCombination, Aggregation,
                                              MultivariateData, NumMV,
                                              hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                              solver=solver, alpha=alpha, max_iter=max_iter, random_state=random_state,
                                              threshold=threshold, thresholdStrategy=thresholdStrategy,
                                              thresholdLambda=thresholdLambda)
    elif Method in ["rf", "RF", "RandomForest"]:
        OneStepForecast, Model = randomForestRegressor(UnivariateData, CoefficientCombination, Aggregation,
                                                       MultivariateData, NumMV, n_estimators=n_estimators,
                                                       criterion=criterion,
                                                       max_depth=max_depth, random_state=random_state, Model=Model,
                                                       threshold=threshold, thresholdStrategy=thresholdStrategy,
                                                       thresholdLambda=thresholdLambda)
    elif Method in ["gbr", "GBR", "GradientBoostingRegressor"]:
        OneStepForecast, Model = gradientBoostingRegressor(UnivariateData, CoefficientCombination, Aggregation,
                                                           MultivariateData, NumMV, loss=loss,
                                                           learning_rate=learning_rate,
                                                           n_estimators=n_estimators, subsample=subsample,
                                                           criterion=criterion, max_depth=max_depth,
                                                           random_state=random_state, threshold=threshold,
                                                           thresholdStrategy=thresholdStrategy,
                                                           thresholdLambda=thresholdLambda)
    elif Method in ["br", "BR", "BaggingRegressor"]:
        OneStepForecast, Model = baggingRegressor(UnivariateData, CoefficientCombination, Aggregation,
                                                  MultivariateData, NumMV,
                                                  n_estimators=n_estimators, random_state=random_state,
                                                  threshold=threshold, thresholdStrategy=thresholdStrategy,
                                                  thresholdLambda=thresholdLambda)
    elif Method in ["abr", "ABR", "adaBoostRegressor"]:
        OneStepForecast, Model = adaBoostRegressor(UnivariateData, CoefficientCombination, Aggregation,
                                                   MultivariateData, NumMV, loss=loss, random_state=random_state,
                                                   threshold=threshold, thresholdStrategy=thresholdStrategy,
                                                   thresholdLambda=thresholdLambda)
    elif Method in ["dr", "DR", "DummyRegressor"]:
        OneStepForecast, Model = dummyRegressor(UnivariateData, CoefficientCombination, Aggregation, MultivariateData, NumMV,
                                                strategy=strategy, threshold=threshold, thresholdStrategy=thresholdStrategy,
                                                thresholdLambda=thresholdLambda)
    else:
        print("Use 'r' for autoregression and 'nn' for neural network (multilayer perceptron)")
        return
    return OneStepForecast, Model


def multistep(UnivariateData, Horizon, CoefficientCombination, Aggregation, Method="r", MultivariateData=None, NumMV=1,
              hidden_layer_sizes=8, activation="relu", solver="lbfgs", alpha=1e-5, n_estimators=100, criterion="mae",
              max_depth=3, loss="ls", learning_rate=0.1, subsample=0.9, random_state=0, strategy="mean",
              max_iter=5000, Model=None, threshold=False, thresholdStrategy="hard", thresholdLambda=0.05):
    """
    data       np.ndarray with dimension 1 or pd.DataFrame with first column for timestamps and
               second column with values
    scales     Number of resolution levels to create. Remember that there are scales many wavelet
               levels plus one level for the smooth coefficients
    """
    # ----------------------------------------------------------------------------------------#
    # ---- Type capturing
    # ----------------------------------------------------------------------------------------#

    # ----------------------------------------------------------------------------------------#
    # ---- Initialization and Error capturing
    # ----------------------------------------------------------------------------------------#
    Model1 = None
    arr_multistep = np.zeros(Horizon)
    for i in range(Horizon):
        forecast, Model = onestep(UnivariateData=UnivariateData, CoefficientCombination=CoefficientCombination,
                                  Aggregation=Aggregation, Method=Method, MultivariateData=MultivariateData, NumMV=NumMV,
                                  hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                  alpha=alpha, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                  loss=loss, learning_rate=learning_rate, subsample=subsample, random_state=random_state,
                                  strategy=strategy, max_iter=max_iter, Model=Model,
                                  threshold=threshold, thresholdStrategy=thresholdStrategy,
                                  thresholdLambda=thresholdLambda)
        if i == 0:
            Model1 = Model
        arr_multistep[i] = forecast
        UnivariateData = np.append(UnivariateData, forecast)
    max_val = np.max(UnivariateData)
    if max_val > 0:
        upper_limit = 1.3 * max_val
    else:
        upper_limit = 0.7 * max_val
    min_val = np.min(UnivariateData)
    if min_val > 0:
        lower_limit = 0.7 * min_val
    else:
        lower_limit = 1.3 * min_val
    for i in range(Horizon):
        if arr_multistep[i] > upper_limit:
            arr_multistep[i] = upper_limit
        if arr_multistep[i] < lower_limit:
            arr_multistep[i] = lower_limit
    return arr_multistep, Model1


#
#
#
#
#
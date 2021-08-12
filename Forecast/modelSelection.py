import numpy as np
import pandas as pd

#from deap import creator, base, tools, algorithms
#from EvoOpt.solvers.DuelistAlgorithm import DuelistAlgorithm
#import EvoOpt
import random
import math
from MRFPY.rollingForecastingOrigin import rollingForecastingOrigin
from MRFPY.forecast import multistep
from scipy.optimize import differential_evolution
import itertools

def nested_model_selection(UnivariateData, Horizon=2, EvaluationLength=2, TestLength=2,
                           Method="r", MultivariateData=None, NumMV=1,
                           numClusters=1, QualityMeasure="MAE", num_optim_iter=1,
                           Sparse=False, SparseCoverage=0.5,
                           init="latinhypercube", hidden_layer_sizes=8, activation="relu", solver="lbfgs", alpha=1e-5,
                           n_estimators=100, criterion="mae", max_depth=3, loss="ls", learning_rate=0.1, subsample=0.9,
                           random_state=0, strategy="mean", max_iter=5000,
                           Threshold=False, ThresholdStrategy="hard", ThresholdLambda=0.05):
    DataLength = len(UnivariateData)

    # Create Test data
    # => There is no need for explicitly defining an evaluation dataset in case of a split in test and evaluation data.
    TestUnivariateData = UnivariateData[0:(DataLength - EvaluationLength)]
    if (len(TestUnivariateData) != (DataLength - EvaluationLength)):
        print("Something went wrong when splitting Data in Test and Evaluation part in modelSelection.py!")
        return
    if (len(TestUnivariateData) == 0):
        print("modelSelection.py: There is no Testdata for computing a model!")
        return
    if type(MultivariateData) == np.ndarray:
        TestMultivariateData = MultivariateData[0:(DataLength - EvaluationLength)]
        if TestMultivariateData.shape[0] != (DataLength - EvaluationLength):
            print(
                "Something went wrong when splitting MultivariateData in Test and Evaluation part in modelSelection.py!")
            return
        if (len(TestUnivariateData) == 0):
            print("modelSelection.py: There is no Testdata (Multivariate data part) for computing a model!")
            return
    else:
        TestMultivariateData = None

    AllAggregations = [np.array([2,4]), np.array([2,4,8]), np.array([2,4,8,16]), np.array([2,4,8,16,32])]
    lst_Results = []
    selectMAE = np.zeros(len(AllAggregations))
    counter = 0
    for Aggregation in AllAggregations:
        NumCoeffs = len(Aggregation) + 1
        CoefficientsMinimum = np.ones(NumCoeffs) * 2
        if Method == "r":
            CoefficientsMaximum = np.ones(NumCoeffs) * 15
        else:
            CoefficientsMaximum = np.ones(NumCoeffs) * 8
        lst_tpls = []
        for i in range(len(CoefficientsMinimum)):
            lst_tpls.append((int(CoefficientsMinimum[i]), int(CoefficientsMaximum[i])))
        my_args = TestUnivariateData, Aggregation, Horizon, TestLength, Method, \
                  TestMultivariateData, NumMV, \
                  Sparse, SparseCoverage, \
                  numClusters, hidden_layer_sizes, activation, solver, alpha, n_estimators, \
                  criterion, max_depth, loss, learning_rate, subsample, random_state, strategy, max_iter, \
                  QualityMeasure, \
                  Threshold, ThresholdStrategy, ThresholdLambda,

        result = differential_evolution(func=evaluation_function, bounds=lst_tpls, args=my_args, maxiter=num_optim_iter,
                                        init=init)
        # Input = np.round(result.x)
        Input = np.array(np.round(result.x), dtype="int")
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=Input, Aggregation=Aggregation,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        tmpError = np.sum(abs(forecast-TestUnivariateData[-Horizon:]))
        selectMAE[counter] = tmpError
        counter = counter + 1
        lst_Results.append(Input)

    Idx = np.argmin(selectMAE)
    Aggregation = AllAggregations[Idx]
    Input = lst_Results[Idx]

    EvaluationError, EvaluationForecast = rollingForecastingOrigin(UnivariateData=UnivariateData,
                                                                   CoefficientCombination=Input,
                                                                   Aggregation=Aggregation, Horizon=Horizon,
                                                                   Window=EvaluationLength,
                                                                   Method=Method,
                                                                   MultivariateData=MultivariateData, NumMV=NumMV,
                                                                   Threshold=Threshold,
                                                                   ThresholdStrategy=ThresholdStrategy,
                                                                   ThresholdLambda=ThresholdLambda,
                                                                   numClusters=numClusters,
                                                                   hidden_layer_sizes=hidden_layer_sizes,
                                                                   activation=activation, solver=solver,
                                                                   alpha=alpha, n_estimators=n_estimators,
                                                                   criterion=criterion, max_depth=max_depth,
                                                                   loss=loss, learning_rate=learning_rate,
                                                                   subsample=subsample,
                                                                   random_state=random_state, strategy=strategy,
                                                                   max_iter=max_iter)
    return Input, EvaluationError, EvaluationForecast




def model_selection(UnivariateData, CoefficientsMinimum, CoefficientsMaximum, Aggregation,
                    Horizon=2, TestLength=2, EvaluationLength=2, Method="r",
                    MultivariateData=None, NumMV=1,
                    Sparse=False, SparseCoverage=0.5,
                    numClusters=1,
                    QualityMeasure="MAE", Optimization="evopt",
                    num_optim_iter=1, init="latinhypercube",
                    hidden_layer_sizes=8, activation="relu", solver="lbfgs", alpha=1e-5, n_estimators=100,
                    criterion="mae",
                    max_depth=3, loss="ls", learning_rate=0.1, subsample=0.9, random_state=0, strategy="mean",
                    max_iter=5000, Threshold=False, ThresholdStrategy="hard", ThresholdLambda=0.05, Verbose=True):

    InternMethodName = "Model selection (model_selection.py)"     # Name for creating messages to the user
    #------------------------------------------------------------------------------------------------------------------#
    # Type capturing
    #------------------------------------------------------------------------------------------------------------------#
    if type(UnivariateData) != np.ndarray:
        Message = ": UnivariateData is not of type np.ndarray!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(CoefficientsMinimum) != np.ndarray:
        Message = ": CoefficientsMinimum is not of type np.ndarray!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(CoefficientsMaximum) != np.ndarray:
        Message = ": CoefficientsMaximum is not of type np.ndarray!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(Aggregation) != np.ndarray:
        Message = ": Aggregation is not of type np.ndarray!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(Horizon) != np.int:
        Message = ": Horizon is not of type np.int!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(TestLength) != np.int:
        Message = ": TestLength is not of type np.int!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(EvaluationLength) != np.int:
        Message = ": EvaluationLength is not of type np.int!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(NumMV) != np.int:
        Message = ": NumMV is not of type np.int!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if numClusters != "max":
        if type(numClusters) != np.int:
            Message = ": numClusters is neither equal to 'max' nor of type np.int!"
            print(InternMethodName + Message)
            return 0, 0, 0, 0, 0, 0, 0

    if type(Method) != str:
        Message = ": Method is not of type str!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(MultivariateData) != np.ndarray:
        if str(type(MultivariateData)) != "<class 'NoneType'>":
            Message = ": MultivariateData is neither np.ndarray (multivariate case) nor of type NoneType (univariate case)!"
            print(InternMethodName + Message)
            return 0, 0, 0, 0, 0, 0, 0

    if type(Sparse) != bool:
        Message = ": Sparse is not of type bool!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if type(SparseCoverage) != float:
        Message = ": SparseCoverage is not of type float!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    if((SparseCoverage > 1) or (SparseCoverage < 0)):
        Message = ": SparseCoverage is not between 0 and 1 (not representing a percentage)!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    #------------------------------------------------------------------------------------------------------------------#
    # Error capturing
    #------------------------------------------------------------------------------------------------------------------#

    if TestLength < 1:
        Message = ": TestLength must be greater than 0!"
        print(InternMethodName + Message)
        return 0, 0, 0, 0, 0, 0, 0

    # CoefficientsMinimum > 1 everywhere
    for i in range(len(CoefficientsMinimum)):
        if CoefficientsMinimum[i] < 1:
            CoefficientsMinimum[i] = 1
            InternPar = "CoefficientsMinimum"
            Message   = " must be bigger than 1 everywhere, setting it to one automatically at position "
            print(InternMethodName + ": Parameter " + InternPar + Message + str(i))

    # Check multivariate case
    if ((type(MultivariateData) == np.ndarray) or (type(MultivariateData) == pd.DataFrame)):
        if NumMV < 1:
            NumMV = 1
            InternPar = "NumMV"
            Message = " must be bigger than 1, setting it to one automatically"
            print(InternMethodName + ": Parameter "+InternPar+Message)
        print(InternMethodName + ": Computing model selection on rolling forecasting origin for multivariate time series!")
    else:
        print(InternMethodName + ": Computing model selection on rolling forecasting origin for univariate time series!")

    if Horizon < 1:
        InternPar = "Horizon"
        Message   = " must be bigger than 1, setting it to one automatically!"
        print(InternMethodName + ": Parameter " + InternPar + Message)

    #------------------------------------------------------------------------------------------------------------------#
    # Print information
    #------------------------------------------------------------------------------------------------------------------#

    # Print information
    if Verbose == True:
        print(InternMethodName+": Input can be successfully used for computations!")
        print(InternMethodName+": Performing a model selection with "+Optimization+" and quality measure "+QualityMeasure+".")
        if type(MultivariateData) != np.ndarray:
            print(InternMethodName + ": Model selection with rolling forecasting origin on univariate time series.")
        else:
            print(InternMethodName + ": Model selection with rolling forecasting origin on multivariate time series.")
        if EvaluationLength > 0:
            print(InternMethodName + ": Performing a model selection for a nested rolling forecasting origin!")
            print(InternMethodName + ": There will be a test error and an evaluation error.")
            print(InternMethodName + ": The test error will determine the model which performance is then measured on the evaluation dataset.")
        else:
            print(InternMethodName + ": Performing a model selection for a NON-nested rolling forecasting origin!")
            print(InternMethodName + ": Choosing the best model obtained on test data.")
        #print(InternMethodName+": Error, Forecast and information log will be saved to file!") #=> Workflow


    #------------------------------------------------------------------------------------------------------------------#
    # Start computations
    #------------------------------------------------------------------------------------------------------------------#
    numCoefficients = len(CoefficientsMinimum)
    lst_x = []
    for i in range(numCoefficients):
        if i != (numCoefficients - 1):
            lst_x.append("w" + str(i + 1))
        else:
            lst_x.append("s" + str(i))
    CoefficientsMinimum.astype(dtype="int")
    CoefficientsMaximum.astype(dtype="int")

    DataLength = len(UnivariateData)

    # Create Test data
    # => There is no need for explicitly defining an evaluation dataset in case of a split in test and evaluation data.
    TestUnivariateData       = UnivariateData[0:(DataLength - EvaluationLength)]
    if (len(TestUnivariateData) != (DataLength - EvaluationLength)):
        print("Something went wrong when splitting Data in Test and Evaluation part in modelSelection.py!")
        return
    if (len(TestUnivariateData) == 0):
        print("modelSelection.py: There is no Testdata for computing a model!")
        return
    if type(MultivariateData) == np.ndarray:
        TestMultivariateData       = MultivariateData[0:(DataLength - EvaluationLength)]
        if TestMultivariateData.shape[0] != (DataLength - EvaluationLength):
            print("Something went wrong when splitting MultivariateData in Test and Evaluation part in modelSelection.py!")
            return
        if (len(TestUnivariateData) == 0):
            print("modelSelection.py: There is no Testdata (Multivariate data part) for computing a model!")
            return
    else:
        TestMultivariateData = None
    # Define en evaluation data set only for univariate
    #if EvaluationLength > 0:
    #    EvaluationUnivariateData = UnivariateData[(DataLength - EvaluationLength):DataLength]
    #    if (len(EvaluationUnivariateData) != EvaluationLength):
    #        print("Something went wrong when splitting Data in Test and Evaluation part in modelSelection.py!")
    #        return


    #------------------------------------------------------------------------------------------------------------------#
    # Model selection on test data
    #------------------------------------------------------------------------------------------------------------------#
    # There must be no evaluation data set for that
    # Model selection is always done on test data!
    # There can be an evaluation on an evaluation dataset (Nested rolling forecasting origin)
    #
    # Get optimal input Input, error Error and quality measure QualityMeasure
    if Optimization == "simulatedannealing":
        Input, Error, QualityMeasure = simulated_annealing(UnivariateData=TestUnivariateData,
                                                           CoefficientsMinimum=CoefficientsMinimum,
                                                           CoefficientsMaximum=CoefficientsMaximum,
                                                           Aggregation=Aggregation, Horizon=Horizon, Window=TestLength,
                                                           Method=Method,
                                                           MultivariateData=TestMultivariateData, NumMV=NumMV,
                                                           Sparse=Sparse, SparseCoverage=SparseCoverage,
                                                           numClusters=numClusters, QualityMeasure=QualityMeasure,
                                                           num_optim_iter=num_optim_iter,
                                                           hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                                           solver=solver, alpha=alpha, n_estimators=n_estimators,
                                                           criterion=criterion, max_depth=max_depth, loss=loss,
                                                           learning_rate=learning_rate, subsample=subsample,
                                                           random_state=random_state,
                                                           strategy=strategy, max_iter=max_iter)
    elif Optimization == "evopt":
        lst_tpls = []
        for i in range(len(CoefficientsMinimum)):
            lst_tpls.append((int(CoefficientsMinimum[i]), int(CoefficientsMaximum[i])))
        my_args = TestUnivariateData, Aggregation, Horizon, TestLength, Method,\
                  TestMultivariateData, NumMV,\
                  Sparse, SparseCoverage,\
                  numClusters, hidden_layer_sizes, activation, solver, alpha, n_estimators,\
                  criterion, max_depth, loss, learning_rate, subsample, random_state, strategy, max_iter,\
                  QualityMeasure,\
                  Threshold, ThresholdStrategy, ThresholdLambda,

        result = differential_evolution(func=evaluation_function, bounds=lst_tpls, args=my_args, maxiter=num_optim_iter,
                                        init=init)
        #Input = np.round(result.x)
        Input = np.array(np.round(result.x), dtype="int")
    elif Optimization == "OneMultiStepEvOpt":



        lst_tpls = []
        for i in range(len(CoefficientsMinimum)):
            lst_tpls.append((int(CoefficientsMinimum[i]), int(CoefficientsMaximum[i])))
        my_args = TestUnivariateData, Aggregation, Horizon, 1, Method, \
                  TestMultivariateData, NumMV, \
                  Sparse, SparseCoverage, \
                  numClusters, hidden_layer_sizes, activation, solver, alpha, n_estimators, \
                  criterion, max_depth, loss, learning_rate, subsample, random_state, strategy, max_iter, \
                  QualityMeasure, \
                  Threshold, ThresholdStrategy, ThresholdLambda,

        result = differential_evolution(func=evaluation_function, bounds=lst_tpls, args=my_args, maxiter=num_optim_iter,
                                        init=init)
        # Input = np.round(result.x)
        Input = np.array(np.round(result.x), dtype="int")
    elif Optimization == "lastBestOneStep":
        # Get the best out of what?
        Agg2 = np.array([2, 4])
        numCoefficients = len(Agg2)
        CoeffComb2 = np.ones(numCoefficients + 1) * 10
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb2, Aggregation=Agg2,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM21 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon
        CoeffComb2 = np.ones(numCoefficients + 1) * 15
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb2, Aggregation=Agg2,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM22 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon
        CoeffComb2 = np.ones(numCoefficients + 1) * 5
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb2, Aggregation=Agg2,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM23 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon

        if QM21 < QM22:
            QM2 = QM21
        elif QM22 < QM23:
            QM2 = QM22
        else:
            QM2 = QM23


        Agg3 = np.array([2, 4, 8])
        numCoefficients = len(Agg3)
        CoeffComb3 = np.ones(numCoefficients + 1) * 10
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb3, Aggregation=Agg3,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM31 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon
        CoeffComb3 = np.ones(numCoefficients + 1) * 15
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb3, Aggregation=Agg3,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM32 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon
        CoeffComb3 = np.ones(numCoefficients + 1) * 5
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb3, Aggregation=Agg3,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM33 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon

        if QM31 < QM32:
            QM3 = QM31
        elif QM32 < QM33:
            QM3 = QM32
        else:
            QM3 = QM33

        Agg4 = np.array([2, 4, 8, 16])
        numCoefficients = len(Agg4)
        CoeffComb4 = np.ones(numCoefficients + 1) * 10
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb4, Aggregation=Agg4,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM41 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon
        CoeffComb4 = np.ones(numCoefficients + 1) * 15
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb4, Aggregation=Agg4,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM42 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon
        CoeffComb4 = np.ones(numCoefficients + 1) * 5
        forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon,
                                CoefficientCombination=CoeffComb4, Aggregation=Agg4,
                                Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
        QM43 = np.sum(abs(forecast - TestUnivariateData[-Horizon:])) / Horizon

        if QM41 < QM42:
            QM4 = QM41
        elif QM42 < QM43:
            QM4 = QM42
        else:
            QM4 = QM43

        if QM4 < QM3 and QM4 < QM2:
            Input = CoeffComb4
            Aggregation = Agg4
        elif QM2 < QM3:
            Input = CoeffComb2
            Aggregation = Agg2
        else:
            Input = CoeffComb3
            Aggregation = Agg3
    elif Optimization == "multistep":
        print("Not finished")
        return
        # Get the best out of what?
        Agg2 = np.array([2, 4])
        numCoefficients = len(Agg2)
        x = np.arange(1, 12 + 1)
        #AllCoefficientCombination2 = np.array(list(itertools.product(x, repeat=numCoefficients + 1)))
        AllCoefficientCombination2 = np.ones(numCoefficients + 1) * 10
        numExperiments = 1#len(AllCoefficientCombination2)
        All2QM = np.zeros(numExperiments)
        counter = 0
        for tmpComb in AllCoefficientCombination2:
            forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon, CoefficientCombination=tmpComb, Aggregation=Agg2,
                                 Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
            All2QM[counter] = np.sum(abs(forecast - TestUnivariateData[-Horizon:]))/Horizon
            counter = counter + 1
        Idx2 = np.argmin(All2QM)
        QM2 = All2QM[Idx2]

        Agg3 = np.array([2, 4, 8])
        numCoefficients = len(Agg3)
        #AllCoefficientCombination3 = np.array(list(itertools.product(x, repeat=numCoefficients + 1)))
        AllCoefficientCombination3 = np.ones(numCoefficients + 1) * 10
        numExperiments = 1# len(AllCoefficientCombination3)
        All3QM = np.zeros(numExperiments)
        counter = 0
        for tmpComb in AllCoefficientCombination3:
            forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon, CoefficientCombination=tmpComb, Aggregation=Agg3,
                                 Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
            All3QM[counter] = np.sum(abs(forecast - TestUnivariateData[-Horizon:]))/Horizon
            counter = counter + 1
        Idx3 = np.argmin(All3QM)
        QM3 = All3QM[Idx3]

        Agg4 = np.array([2, 4, 8, 16])
        numCoefficients = len(Agg4)
        #AllCoefficientCombination4 = np.array(list(itertools.product(x, repeat=numCoefficients + 1)))
        AllCoefficientCombination4 = np.ones(numCoefficients + 1) * 10
        numExperiments = 1#len(AllCoefficientCombination4)
        All4QM = np.zeros(numExperiments)
        counter = 0
        for tmpComb in AllCoefficientCombination4:
            forecast, m = multistep(TestUnivariateData[:-Horizon], Horizon=Horizon, CoefficientCombination=tmpComb, Aggregation=Agg4,
                                 Method=Method, MultivariateData=MultivariateData, NumMV=NumMV)
            All4QM[counter] = np.sum(abs(forecast - TestUnivariateData[-Horizon:]))/Horizon
            counter = counter + 1
        Idx4 = np.argmin(All4QM)
        QM4 = All4QM[Idx4]

        if QM4 < QM3 and QM4 < QM2:
            Input = AllCoefficientCombination4[Idx4]
            Aggregation = np.array([2, 4, 8, 16])
        elif QM2<QM3:
            Input = AllCoefficientCombination2[Idx2]
            Aggregation = np.array([2, 4])
        else:
            Input = AllCoefficientCombination3[Idx3]
            Aggregation = np.array([2,4,8])
    else:
        print("Optimization method not implemented. Available optimization methods are"
              "'simulatedannealing': Simulated Annealing"
              "'evopt': Evolutionary Optimization."
              "Return and retry.")
        return

    #------------------------------------------------------------------------------------------------------------------#
    # Case: Nonnested rolling forecasting origin
    #------------------------------------------------------------------------------------------------------------------#
    # Get rolling forecasting origin on test data
    TestError, TestForecast = rollingForecastingOrigin(UnivariateData=TestUnivariateData,
                                                       CoefficientCombination=Input,
                                                       Aggregation=Aggregation, Horizon=Horizon,
                                                       Window=TestLength,
                                                       Method=Method,
                                                       MultivariateData=TestMultivariateData, NumMV=NumMV,
                                                       Sparse=False,     # Never True! => False results
                                                       Threshold=Threshold,
                                                       ThresholdStrategy=ThresholdStrategy,
                                                       ThresholdLambda=ThresholdLambda,
                                                       numClusters=numClusters,
                                                       hidden_layer_sizes=hidden_layer_sizes,
                                                       activation=activation, solver=solver, alpha=alpha,
                                                       n_estimators=n_estimators, criterion=criterion,
                                                       max_depth=max_depth, loss=loss,
                                                       learning_rate=learning_rate,
                                                       subsample=subsample, random_state=random_state,
                                                       strategy=strategy, max_iter=max_iter)
    LengthTestError = TestError.shape[0]
    if LengthTestError > 0:
        TestQualityMeasure = np.sum(np.sum(abs(TestError))) / (LengthTestError * Horizon)
    else:
        TestQualityMeasure = 0

    EvaluationError = 0
    EvaluationForecast = 0
    EvaluationQualityMeasure = 0

    #------------------------------------------------------------------------------------------------------------------#
    # Case: Nested rolling forecasting origin (NRFO)
    #------------------------------------------------------------------------------------------------------------------#
    # => Use evaluation dataset to create a rolling forecasting origin on evaluation data
    # Model selection was done on test data
    if EvaluationLength > 0:
        EvaluationError, EvaluationForecast = rollingForecastingOrigin(UnivariateData=UnivariateData,
                                                                       CoefficientCombination=Input,
                                                                       Aggregation=Aggregation, Horizon=Horizon,
                                                                       Window=EvaluationLength,
                                                                       Method=Method,
                                                                       MultivariateData=MultivariateData, NumMV=NumMV,
                                                                       Sparse=False,     # Never True! => False results
                                                                       Threshold=Threshold,
                                                                       ThresholdStrategy=ThresholdStrategy,
                                                                       ThresholdLambda=ThresholdLambda,
                                                                       numClusters=numClusters,
                                                                       hidden_layer_sizes=hidden_layer_sizes,
                                                                       activation=activation, solver=solver,
                                                                       alpha=alpha,n_estimators=n_estimators,
                                                                       criterion=criterion, max_depth=max_depth,
                                                                       loss=loss, learning_rate=learning_rate,
                                                                       subsample=subsample,
                                                                       random_state=random_state, strategy=strategy,
                                                                       max_iter=max_iter)
        LengthEvaluationError = EvaluationError.shape[0]
        if LengthEvaluationError > 0:
            EvaluationQualityMeasure = np.sum(np.sum(abs(EvaluationError))) / (LengthEvaluationError * Horizon)
        else:
            EvaluationQualityMeasure = 0

    return Input, TestError, TestForecast, TestQualityMeasure, EvaluationError, EvaluationForecast, EvaluationQualityMeasure


def evaluation_function(CoefficientCombination,
                        UnivariateData, Aggregation, Horizon, Window, Method,
                        MultivariateData, NumMV,
                        Sparse, SparseCoverage,
                        numClusters, hidden_layer_sizes, activation, solver, alpha, n_estimators,
                        criterion, max_depth, loss, learning_rate, subsample, random_state, strategy, max_iter,
                        QualityMeasure,
                        Threshold, ThresholdStrategy, ThresholdLambda):
    InternMethodName="evaluation_function.py (subroutine for modelSelection.py)"
    if type(QualityMeasure) != str:
        Message = ": Method is not of type str!"
        print(InternMethodName + Message)
        return
    Error, Forecast = rollingForecastingOrigin(UnivariateData=UnivariateData,
                                               CoefficientCombination=CoefficientCombination,  # Input -> variable here
                                               Aggregation=Aggregation, Horizon=Horizon, Window=Window,
                                               Method=Method,
                                               MultivariateData=MultivariateData, NumMV=NumMV,
                                               Sparse=Sparse, SparseCoverage=SparseCoverage,
                                               numClusters=numClusters, hidden_layer_sizes=hidden_layer_sizes,
                                               activation=activation, solver=solver, alpha=alpha,
                                               n_estimators=n_estimators, criterion=criterion,
                                               max_depth=max_depth, loss=loss, learning_rate=learning_rate,
                                               subsample=subsample, random_state=random_state,
                                               strategy=strategy, max_iter=max_iter,
                                               Threshold=Threshold,
                                               ThresholdStrategy=ThresholdStrategy, ThresholdLambda=ThresholdLambda)
    qm = 0
    if type(Error) == np.ndarray:
        Length = Error.shape[0]
        Normalization = Length*Horizon
        if QualityMeasure == "MAE":
            qm = np.sum(np.sum(abs(Error)))/Normalization
        elif QualityMeasure == "AIC":
            qm = 0
        elif QualityMeasure == "MRE":
            # Calculate two dimensional qm
            # Use pareto optimum
            # Allow penalization
            qm = 0
        elif QualityMeasure == "RMSE":
            qm = 0
        elif QualityMeasure == "MSE":
            qm = 0
        else:
            print("No implemented Quality Measure was given. Default is Mean Absolute Error.")
            qm = np.sum(np.sum(abs(Error)))/(Length*Horizon)
    return qm


def simulated_annealing(UnivariateData, CoefficientsMinimum, CoefficientsMaximum, Aggregation, Horizon, Window, Method,
                        MultivariateData=None, NumMV=1, numClusters=1, QualityMeasure="MAE", num_optim_iter=1,
                        hidden_layer_sizes=8, activation="relu", solver="lbfgs",
                        alpha=1e-5, n_estimators=100, criterion="mae", max_depth=3, loss="ls",
                        learning_rate=0.1, subsample=0.9, random_state=0, strategy="mean", max_iter=5000):
    # Generate random input initialization
    len_x = len(CoefficientsMinimum)
    old_x = np.zeros(len_x)
    new_x = np.zeros(len_x)
    for i in range(len_x):
        old_x[i] = random.sample(range(CoefficientsMinimum[i], CoefficientsMaximum[i]+1), 1)[0]
    old_x = old_x.astype(dtype = "int")
    # Evaluate random input initialization
    oldError, mat_forecast = rollingForecastingOrigin(UnivariateData=UnivariateData, CoefficientCombination=old_x,
                                                      Aggregation=Aggregation, Horizon=Horizon, Window=Window,
                                                      Method=Method, MultivariateData=MultivariateData, NumMV=NumMV,
                                                      numClusters=numClusters, hidden_layer_sizes=hidden_layer_sizes,
                                                      activation=activation, solver=solver, alpha=alpha,
                                                      n_estimators=n_estimators, criterion=criterion,
                                                      max_depth=max_depth, loss=loss, learning_rate=learning_rate,
                                                      subsample=subsample, random_state=random_state,
                                                      strategy=strategy, max_iter=max_iter)
    if QualityMeasure == "MAE":
        oldQM = np.sum(np.sum(abs(oldError)))/(Window*Horizon)
    else:
        oldQM = np.sum(np.sum(abs(oldError))) / (Window * Horizon)
    # Start exponential cool down system
    for temperature in np.logspace(0,5,num=num_optim_iter)[::-1]:
        # Generate random input
        for i in range(len_x):
            new_x[i] = random.sample(range(CoefficientsMinimum[i], CoefficientsMaximum[i]+1), 1)[0]
        new_x = new_x.astype(dtype = "int")
        # Evaluate random input
        newError, mat_forecast = rollingForecastingOrigin(UnivariateData=UnivariateData, CoefficientCombination=old_x,
                                                          Aggregation=Aggregation, Horizon=Horizon, Window=Window,
                                                          Method=Method, MultivariateData=MultivariateData, NumMV=NumMV,
                                                          numClusters=numClusters,
                                                          hidden_layer_sizes=hidden_layer_sizes,
                                                          activation=activation, solver=solver, alpha=alpha,
                                                          n_estimators=n_estimators, criterion=criterion,
                                                          max_depth=max_depth, loss=loss, learning_rate=learning_rate,
                                                          subsample=subsample, random_state=random_state,
                                                          strategy=strategy, max_iter=max_iter)
        if QualityMeasure == "MAE":
            newQM = np.sum(np.sum(abs(newError))) / (Window * Horizon)
        else:
            newQM = np.sum(np.sum(abs(newError))) / (Window * Horizon)
        # Decide if to take or not to take
        if math.exp((oldQM - newQM) / temperature) > random.random():
            oldError = newError
            oldQM = newQM
            old_x = new_x
    return old_x, oldError, oldQM





#
#
#
#
#

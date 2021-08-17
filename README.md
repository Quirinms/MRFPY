# MRFPY
*MRFPY: Multi Resolution Forecasting in Python*

*MRFR: [Multi Resolution Forecasting in R](https://github.com/Quirinms/MRFR)*

*Multi Resolution Time Series Forecasting (with wavelets)*

This is a python package for time series forecasting and got created for a master thesis, University of Marburg, June, 2020.

## Contents

1. [Description](#description)
2. [Use cases](#use-cases)
3. [Installation](#installation)
4. [Documentation](#documentation)
5. [References](#references)

## Description

Forecasting seasonal univariate time series data is a major challenge. The seasonality
of a time series can be analytically investigated with Fourier or multiresolution theory.
There are forecasting procedures using a Fourier approach or a multiresolution predic-
tion algorithm which use aggregation for that task. Different wavelet algorithms for
time series forecasting were proposed in theory with promising potential. Since it could
be shown that wavelets recognize time localized frequency components, the question
is if they are suitable to model and forecast seasonal time series. This work imple-
ments a wavelet forecasting approach called the multiresolution forecasting approach
and examines its potential to forecast seasonal univariate time series. The performance
of the multiresolution forecasting approach is then compared to a Fourier and an ag-
gregation method. Furthermore, the results from a selection of robust and automated
forecasting procedures can be found in the first appendix to provide a broader overview.
The thesis presents a wavelet preprocessing combined with linear and nonlinear forecast-
ing methods. A special form of the wavelet transformation is used in order to extract
features from a univariate time series. These features carry different frequency infor-
mation over varying time periods. The features are processed in linear and nonlinear
methods to produce one-step forecasts. Multi-step forecasts are computed by applying
the one-step forecast procedure recursively.
The results are important for practical purposes, since there are many use cases for
seasonal time series forecasting, for example for workforce management.
The evaluation is done with appropriate quality measures and visualization techniques.
For comparison, the four forecasting methods are applied to three datasets. The re-
sulting forecasts of the presented multiresolution approach can compete with those of
state-of-the-art methods.

## Use cases

4 use cases: Callcenter, Electricity demand, oil prices and stock values.

![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Callcenter.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Electricity.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Prices.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Stox.png)

![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Callcenter_SMAPE_Reference_Horizon_1.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Callcenter_SMAPE_Reference_From_1_To_14.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Entsoe_SMAPE_Reference_From_1_To_14.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Entsoe_SMAPE_Reference_Horizon_1.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/SAP500_SMAPE_Reference_From_1_To_14.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/SAP500_SMAPE_Reference_Horizon_1.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/SEP_SMAPE_Reference_From_1_To_14.png)
![alt text](https://github.com/Quirinms/MRFPY/blob/master/doc/images/SEP_SMAPE_Reference_Horizon_1.png)


## Installation

Execute following command in a terminal:

pip install MRFPY

## Documentation



## References

Aussem,  A.,  Campbell,  J.,  and  Murtagh,  F.  (1998)  Wavelet-based  feature extraction    and    decomposition    strategies    for    financial    forecasting, International Journal of Computational Intelligence in Finance, 6 (5-12).




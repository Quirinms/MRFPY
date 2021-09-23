# Multiresolution Forecasting Libraries in Python and R
*MRFPY on pypi: [Multi Resolution Forecasting in Python](https://pypi.org/project/MRFPY/)*
*MRFPY on GitHub: [Multi Resolution Forecasting in Python](https://github.com/Quirinms/MRFPY)*

*MRFR on CRAN: [Multi Resolution Forecasting in R](https://CRAN.R-project.org/package=mrf)*
*MRFR on GitHub: [Multi Resolution Forecasting in R](https://github.com/Quirinms/MRFR)*

*Multiresolution Forecasting in Python (with wavelets)*

This is a Python package for univariate time series forecasting with wavelets.
An adaption is also available in R with similar handling and naming.



## Contents

1. [Highlights](#highlights)
2. [Description](#description)
3. [Installation](#installation)
4. [References](#references)


## Highlights

![Highlights](https://github.com/Quirinms/MRFPY/blob/master/doc/images/Highlights.png)

## Description

This package provides an implementation of an algorithm of the workgroup around Renaud, O., Starck, J.-L., and Murtagh, F..
It uses a redundant Haar wavelet transform to decompose a time series in its wavelet and the corresponding smooth approximation features.
Those features are processed in linear or nonlinear methods in order to yield a one-step forecast.
Multi-step forecasts are obtained recursively.
Currently, only univariate time series can be forecasted.
There is ongoing work for multivariate time series forecasting.
Find the theoretical work from the above mentioned workgroup in the references.

You can create one-step forecasts with various linear and nonlinear methods using wavelet features trying out different possibilites.
One-step forecasts can be created by directly accessing the methods specific function call or the abstract method "onestep".
Multi-step forecasts are computed recursively and can be called with the abstract method "multistep".
Evaluation studies of one specific setting can be computed with the rolling window function.
A complete model selection with nested cross validation can be called with the function model_selection.



## Installation

Execute following command in a terminal:

pip install MRFPY==1.0.1 

## References

Aussem,  A.,  Campbell,  J.,  and  Murtagh,  F.  (1998)  Wavelet-based  feature extraction    and    decomposition    strategies    for    financial    forecasting, International Journal of Computational Intelligence in Finance, 6 (5-12).

Aussem, A., Campbell, J., and Murtagh, F.: Waveletbased Feature Extraction and Decomposition Strategies for Financial Forecasting.
International Journal of Computational Intelligence in Finance, 6:5–12. 1998.

Benaouda, D., Murtagh, F., Starck, J.-L., and Renaud, O.: Wavelet-based Nonlinear Multiscale Decomposition Model for Electricity Load
Forecasting. Neurocomputing, 70(1-3):139–154. doi:10.1016/j.neucom.2006.04.005. 2006.

Gonghui, Z., Starck, J.-L., Campbell, J., and Murtagh, F.: The Wavelet Transform for Filtering Financial Data Streams. Journal of Computational Intelligence in Finance, 7(3):18–35. 1999.

Murtagh, F., Starck, J.-L., and Renaud, O.: On Neuro-Wavelet Modeling. Decision Support Systems, 37(4):475–484. doi:10.1016/S0167-9236(03)00092-7. 2004.

Renaud, O., Starck, J.-L., and Murtagh, F.: Prediction based on a Multiscale Decomposition. International Journal of Wavelets, Multiresolution and Information Processing, 1(2):217–232. doi:10.1142/S0219691303000153. 2003.

Renaud, O., Starck, J.-L., and Murtagh, F.: Wavelet-based combined Signal Filtering and Prediction. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 35(6):1241–1251. doi:10.1109/TSMCB.2005.850182. 2005.


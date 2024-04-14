## model-based-seasonality
A number of statistical and ML models that take in financial trading data and output seasonal patterns

# Overview

This job identifies seasonal trading patterns exhibited by clients. It identifies 
seasonality on:
- Month-end dates

## Try for youself
You should be able to run at the bottom of the start_app.py, xgboost_model.py and 
test.py. 
NB. You might need to obtain ING credit file _fx_mosaic_kdb_trades_20200101_20201231.csv_ for the year 2020 and add into the data folder under a name 'trades2020.csv'.

## Data

The data used for the models currently is ING 2020-2021 data.

## Pre-processing

The pre-processing of the data involves several steps, namely:

- Removing null values
- Removing outliers that don't fall in 5-95% percentile.
- Running stationarity tests and returning only non-stationary 
clients

## Methods

For now we only run month-end seasonality jobs for most traded
clients. The models we implemented are:
- AutoReg
- ARIMA
- SARIMA
- XGBoost

However, the hyperparameter search was focused mainly on SARIMA
and hence, the seasonality job applies this model at the moment,
which can be configurable in the future.


#### Filtering out non-seasonal clients

The time series regression models are applied on clients that are
non-stationary (i.e. have either trend or seasonal components).
For this reason we apply Dickey–Fuller test and Kwiatkowski–Phillips–Schmidt–Shin (KPSS) tests.
Which test to use for filtering non-stationary clients is configurable.
Furthermore, the tests have been shown to have contradicting results, so
should be seen as increasing confidence for a client to be non-stationary 
rather than a definite result. 

#### Cross-validation
All the trained models were compared on a validation dataset. The evaluation metrics used include:

- RMSE root_mean_squared_error
- percentage RMSE
- MAE - mean_absolute_percentage_error
- R<sup>2</sup> score
- MAPE - mean_absolute_percentage_error
- Explained Variance Score.

#### The directory structure


- **_evaluation.py_** contains all model metric related functions and calculations
- **_model_seasonality_test.py_** contains stationary tests that are used
- to filter out stationary clients.
- **_sarima.py_**, **_xgboost_model.py_** and **_other_models.py_** contain the models:
AutoReg, ARIMA, SARIMAX from stats library and XGBoost models training code, 
including hyperparameter search.
- **_start_app.py_** is the file to run the seasonality job
- **_preprocess_data.py_** contains all the data preprocessing steps,
from reading, removing NaNs and outliers, filtering stationary clients and 
in case of training, splitting to train, validation and test (80-10-10) datasets.

N.B. throughout the code there might be NUM variable that restricts training data 
on more than NUM clients for faster training and processing of the data. Given 
the computational capacity simply remove the constraint.





 
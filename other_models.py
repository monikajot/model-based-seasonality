import matplotlib.pyplot as plt
import pandas as pd
import random as rand
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from preprocess_data import get_client_data
from evaluation import evaluate
NUM = 1


def autoreg_model(train, val, plot=True, pars = None):
    if pars:
        lags, trend, seasonal, period = pars[0], pars[1], pars[2], pars[3]
    else:
        lags, trend, seasonal, period = (30,'ct',True,90)
    # fit model
    model = AutoReg(train, lags=lags, trend=trend, seasonal=seasonal, period=period)
    model_fit = model.fit()
    # make prediction
    pred = model_fit.predict(len(train),len(val)+len(train)-1,dynamic=False)
    if plot:
        plt.plot(val, color='blue' )
        plt.plot(pred.reset_index(drop=True), '-.',color='red')
        plt.legend()
    return pred


def arima_model(train, test, plot=False):
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        pred = output
        predictions.append(pred)
        true = test[i]
        history.append(pred)
        if plot:
            print('predicted=%f, expected=%f' % (pred, true))
    if plot:
        plt.plot(test)
        plt.plot(predictions, color='red')
    return predictions


def arima_model_all_clients(train_data, val_data, clients_sorted):
    results = []
    for client in clients_sorted['Client_Name'][:NUM]:
        train = get_client_data(client, train_data).Notional.reset_index(drop=True)
        val = get_client_data(client, data=val_data).Notional.reset_index(drop=True)
        if len(val) > 10 and len(train) > 100:
            pred = arima_model(train, val)
            results.append(evaluate(pred, val))
    return results


def autoreg_hyper_search(train, val):
    lags = [1, 5, 10, 30, 50]
    trend = ['n', 'c', 't', 'ct']
    seasonal = [False, True]
    period = [5, 10, 30, 60, 90]
    rand_runs = 20
    results= []
    for run in range(rand_runs):
        l, t, s, p = rand.choice(lags), rand.choice(trend), rand.choice(seasonal), rand.choice(period)
        pred = autoreg_model(train, val, plot=False, pars=[l, t, s, p])
        d = {'lags': l, 'trend': t, 'seasonality': s, 'period': p}
        d.update(evaluate(pred,val))
        results.append(d)
    return pd.DataFrame(results)
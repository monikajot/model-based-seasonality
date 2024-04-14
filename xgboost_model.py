import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from preprocess_data import get_client_data, get_raw_data, preprocess
from evaluation import evaluate
import pickle


class XGBoostModel():
    def __init__(self):
        self.FILE_NAME = 'data/trades2020.csv'
        self.raw_data = get_raw_data(self.FILE_NAME)
        train_data, val_data, clients_sorted, test = preprocess(self.raw_data)
        self.train, self.val, self.clients_sorted = train_data, val_data, clients_sorted
        self.pars = {
            'n_estimators': 1000,
            'subsample': 1,
            'max_depth': 6,
            'eta': 0.01,
            'min_split_loss': 0,
            'alpha': 0,
            'reg_lambda': 1,
        }

    def fit(self):
        predictions, results = self.xgb_model_all_clients(self.train, self.val, self.clients_sorted)
        return predictions, results

    def predict(self, train, test):
        walk_forward_validation(train, test, self.pars, plot=False)

    def xgb_model_all_clients(self, train_data, val_data, clients_sorted):
        return xgb_model_all_clients(train_data, val_data, clients_sorted, self.pars)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    # transform a time series dataset into a supervised learning dataset
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def xgboost_forecast(train, testX, pars):
    if pars:
        n_estimators = pars['n_estimators']
        subsample = pars['subsample']
        max_depth = pars['max_depth']
        eta = pars['eta']
        reg_lambda = pars['reg_lambda']
    else:
        n_estimators=1000
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=n_estimators,
                         subsample=subsample,
                         max_depth=max_depth,
                         eta=eta,
                         reg_lambda= reg_lambda,
                        )
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return model, yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(train, test, pars, plot=False):
    predictions = list()
    # split dataset
#     train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        model, yhat = xgboost_forecast(history, testX, pars)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
#         print('>expected=%.1f, predicted=%.1f' % (testy, yhat))

    if plot:
        plt.plot(test)
        plt.plot(predictions, color='red')
    return model, predictions


def xgb_model_all_clients(train_data, val_data, clients_sorted, pars):
    results = []
    predictions = []
    for client in clients_sorted['Client_Name'][:NUM_CLIENTS_TRAIN]:
        train = get_client_data(client, train_data).Notional.reset_index(drop=True)
        val = get_client_data(client,data=val_data).Notional.reset_index(drop=True)
        if len(val)>10 and len(train)>100:
            train_xgb = series_to_supervised(train)
            val_xgb = series_to_supervised(val)
            _, pred = walk_forward_validation(train_xgb, val_xgb, pars)
            predictions.append(pred)
            results.append(evaluate(pred, val))
    return predictions, results

if __name__ == '__main__':
    NUM_CLIENTS_TRAIN =2
    my_model = XGBoostModel()
    # NUM_CLIENTS_TRAIN = 5
    # with open('data/my_model_init', 'rb') as f:
    #     my_model = pickle.load(f)
    model, pred = my_model.fit()
    print(pred)
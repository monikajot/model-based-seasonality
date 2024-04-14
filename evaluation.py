import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))) * 100


def evaluate(true, pred):
    rmse = mean_squared_error(true, pred, squared=False)
    rmse_p = rmse_percentage_error(true, pred,)
    mae = mean_absolute_error(true, pred)
    R2 = r2_score(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    exl_var = explained_variance_score(true, pred)
    return {
        'rmse': rmse,
        'rmse_p':rmse_p,
        'mae': mae,
        'R2': R2,
        'mape': mape,
        'exl_var': exl_var
           }


def visualise_entire_plot(train, val, pred):
    # plot full series train + predicted
    full = np.concatenate([train.to_numpy(), val.to_numpy()], axis=0)
    full_pred = np.concatenate([train.to_numpy(), pred.to_numpy()], axis=0)
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.plot(full, 'blue')
    plt.plot(full_pred, '-.',color='red')


def get_xgb_pars():
    pars = dict()
    n_estimators = [1,20,100,200,1000]
    subsample = [.5, 1]
    max_depth = [4,6,9,15]
    eta = [0.01, 0.1, 0.2, 0.3]
    min_split_loss = [0]
    alpha = [0]
    reg_lambda = [0.5,1,2]
    pars['n_estimators'] = rand.choice(n_estimators)
    pars['subsample'] = rand.choice(subsample)
    pars['max_depth'] = rand.choice(max_depth)
    pars['eta'] = rand.choice(eta)
    pars['reg_lambda']=rand.choice(reg_lambda)
    return pars


def get_sarimax_pars():
    """
    Function description
    Parameters
    ----------
    transaction_data
        The pre-processed transaction data

    Returns
    -------
    pd.DataFrame
        A dataframe with the percentage of the total volum
    """
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['t', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    s_params = 12
    pars = {}

    pars['p'], pars['d'], pars['q'] = rand.choice(p_params), rand.choice(d_params), rand.choice(q_params)
    pars['P'], pars['D'], pars['Q'] = rand.choice(P_params), rand.choice(D_params), rand.choice(Q_params)
    pars['t'] = rand.choice(t_params)
    pars['s'] = s_params

    return pars
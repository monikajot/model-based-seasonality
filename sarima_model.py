import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_model(train, test, plot=True, pars=None):
    if pars:
        p,d,q = pars['p'],pars['d'],pars['q']
        P,D,Q,s = pars['P'],pars['D'],pars['Q'], pars['s']
    else:
        p,d,q = 2,1,1
        P,D,Q,s = 1,1,1,12

    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = SARIMAX(history, order=(p, d, q), seasonal_order=(P, D, Q, s), disp=-1)
        model_fit = model.fit()
        output = model_fit.forecast()
        predictions.append(output)
        true = test[t]
        history.append(true)
        if plot:
            print('predicted=%f, expected=%f' % (output, true))
    if plot:
        plt.plot(test)
        plt.plot(predictions, color='red')
    return model, predictions
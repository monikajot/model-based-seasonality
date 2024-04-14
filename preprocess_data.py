import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import pickle
NUM = 50
FILE_NAME = 'data/trades2020.csv'
SPLIT_VAL_DATE = '2020-10-30 00:00:00'
SPLIT_TEST_DATE = '2020-11-30 00:00:00'


def remove_outliers(df):
    """
    Removes top and bottom percentile. Set to 5% but can be configured.
    Parameters
    ----------
    df
        Input pd.DataFrame with column Reporting_currency_notional
    Returns
    -------
    pd.DataFrame
        A dataframe with the top and bottom outliers removed
    """
    if df.shape[0]>0 and 'Reporting_currency_notional' in df.columns:
        N = 'Reporting_currency_notional'
        q_low = df[N].quantile(0.04)
        q_hi = df[N].quantile(0.95)
        df_filtered = df[(df[N] < q_hi) & (df[N] > q_low)]
        return df_filtered
    else:
        return df

def preprocess(raw_data, training_flag=True, test='KPSS'):
    """
    Function to remove NaN values, outlier values, extract date columns and
    filter out non-stationary clients and return either all data or split to
    train, validation and testing.
    Parameters
    ----------
    raw_data
        The raw data ingested with all client transaction history
    training_flag
        determines if return all data or split to training, valid and test
    test
        defines which stationarity test is used between KPSS and ADF
    Returns
    -------
    Tuple
        if training_flag True, returns 4 of pd.Dataframe for training data,
        validation data, clients names sorted by their Notional value and test data
        if training_flag is False, returns processed transaction data and
        clients names sorted by their Notional value.
    """
    if raw_data.shape[0] < 1:
        print('DataFrame either has no elements.')
        return
    cols = ['Transaction_ID', 'Client_Name', 'Transaction_Timestamp', 'Reporting_currency_notional']
    for col in cols:
        if col not in raw_data.columns:
            print('DataFrame does not have the expected columns.')
            return

    data_nonan = raw_data.dropna(subset=['Client_Name', 'Transaction_Timestamp', 'Reporting_currency_notional'])
    df = remove_outliers(data_nonan)

    new = df['Transaction_Timestamp'].str.split("D", n=1, expand=True)
    df["Date"] = new[0]
    frame = {
        'Transaction_ID': df['Transaction_ID'],
        'Client_Name': df['Client_Name'].astype(int).astype(str),
        'Date': pd.to_datetime(df['Date']),
        'Notional': df['Reporting_currency_notional']
        }
    all_data = pd.DataFrame(frame)

    non_stationary_clients = run_stationarity_tests(all_data)
    stationary_mask = (non_stationary_clients[test] == 0)
    stationary_clients = non_stationary_clients['Client_Name'][stationary_mask]
    if non_stationary_clients.empty:
        print('No clients left after filtering non-seasonal clients')
        return pd.DataFrame()
    data = all_data[all_data['Client_Name'].isin(stationary_clients.values)]

    if training_flag:
        train_data, val_data, clients_sorted, test_data = split_data(data)
        with open('data/test_data.pickle', 'wb') as f:
            pickle.dump(test_data, f)
        with open('data/train_data.pickle', 'wb') as f:
            pickle.dump(train_data, f)
        with open('data/validation_data.pickle', 'wb') as f:
            pickle.dump(val_data, f)
        return train_data, val_data, clients_sorted, test_data
    else:
        clients_sorted = data.groupby(
            'Client_Name', as_index=False
        ).sum(
            'Notional'
        ).sort_values(
            by='Notional', ascending=False
        )
        return data, clients_sorted


def split_data(data):
    """
    Splits data to training, validation and test in 80-10-10 ratio
    Parameters
    ----------
    data
        Transaction data with all clients
    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Transaction data split for training
    """
    train_data = data[data['Date'] <= pd.to_datetime(SPLIT_VAL_DATE)]
    validation_mask = (
            (pd.to_datetime(SPLIT_VAL_DATE) < data['Date']) & (pd.to_datetime(SPLIT_TEST_DATE) > data['Date'])
    )
    val_data = data[validation_mask]
    test_data = data[pd.to_datetime(SPLIT_TEST_DATE) < data['Date']]
    clients_sorted = train_data.groupby(
        'Client_Name', as_index=False
    ).sum(
        'Notional'
    ).sort_values(
        by='Notional', ascending=False
    )
    return train_data, val_data, clients_sorted, test_data


def get_client_data(name, data):
    """
    Obtains a dataframe of client transaction history indexed with dates and Notional values.
    Parameters
    ----------
    name
        Client name
    data
        entire history of transaction data
    Returns
    -------
    pd.DataFrame
        A dataframe indexed with dates and Notional values
    """
    client_data = data[data['Client_Name'] == name]
    client_data_t = client_data[['Date', 'Notional']]
    return client_data_t.groupby('Date').sum()


def get_raw_data(file_name):
    return pd.read_csv(file_name)


def ADF_test(timeseries):
    """
    Implements ADF timeseries stationarity test
    ----------
    timeseries
        1d pd.DataFrame to determine its stationarity
    Returns
    -------
    float
        p-value testing the hypothesis that TS is stationary
    """
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    return dftest[1]


def KPSS_test(timeseries):
    """
        Implements KPSS timeseries stationarity test
        ----------
        timeseries
            1d pd.DataFrame to determine its stationarity
        Returns
        -------
        float
            p-value testing the hypothesis that TS is non-stationary
        """
    kpsstest = kpss(timeseries.dropna(), regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    return kpsstest[1]


def run_stationarity_tests(data):
    """
    Implements stationarity tests for NUM of clients
    ----------
    data
        transaction history after NaN and outliers were removed
    Returns
    -------
    pd.DataFrame
        with columns for client name and 0 or 1 value for both tests,
        where 1 is stationary and 0 is non-stationary
    """
    if data.shape[0]>0 and 'Client_Name' in data.columns:
        clients = data['Client_Name'].unique()
        stationary_frame = {
            'Client_Name': [],
            'ADF': [],
            'KPSS': []
        }
        for client in clients[:5]:
            try:
                d = get_client_data(name=client, data=data)

                if ADF_test(d) < 0.05:
                    a = 1
                else:
                    a = 0

                if KPSS_test(d) < 0.05:
                    k = 0
                else:
                    k = 1
                stationary_frame['Client_Name'].append(client)
                stationary_frame['ADF'].append(a)
                stationary_frame['KPSS'].append(k)
            except ValueError:
                print(f'Client {client} does not have enough data to be included')
        return pd.DataFrame(stationary_frame)
    else:
        return
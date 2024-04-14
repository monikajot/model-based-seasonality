from preprocess_data import get_raw_data, preprocess, get_client_data
from evaluation import evaluate
from sarima_model import sarimax_model
import pandas as pd
import pickle


job_specs = {
    'outliers': 0.5,
    'FILE_NAME': 'data/trades2020.csv',
    'model_method': sarimax_model,
    'stationarity_test': 'KPSS',
}


class StatsModel:
    def __init__(
            self,
            model_fn = job_specs['model_method'],
            FILE_NAME = job_specs['FILE_NAME']
    ):
        self.model_fn = model_fn
        self.best_models = {}
        self.raw_data = get_raw_data(FILE_NAME)

    def fit(self, data) -> tuple:
        """
        Function to remove NaN values, outlier values, extract date columns and
        filter out non-stationary clients and return either all data or split to
        train, validation and testing.
        Parameters
        ----------
        raw_data
            The raw data ingested with all client transaction history
        training_flag
            Determines if return all data or split to training, valid and test
        test
            Defines which stationarity test is used between KPSS and ADF
        Returns
        -------
        Tuple
            If training_flag True, returns 4 of pd.Dataframe for training data,
            validation data, clients names sorted by their Notional value and test data
            if training_flag is False, returns processed transaction data and
            clients names sorted by their Notional value.
        """
        train_data, val_data, clients_sorted, test_data = preprocess(data)

        training_results = self.train_models_all_clients(train_data, val_data, clients_sorted, self.model_fn)
        if training_results:
            best_models, predictions = training_results
            self.best_models = best_models
            return best_models, predictions
        else:
            print('Training not successful due to insufficient data')
            return

    def predict(
            self,
            client_name: int,
            predict_n_days: int
    ) -> list :
        """
        Function inputs client data and predicts next n days
        Parameters
        ----------
        client_name
            Client name (int)
        predict_n_days
            integer predicting next n days
        Returns
        -------
        pd.DataFrame
            Returns predicted daily notional values for the next n days
        """
        predictions = list()
        # walk-forward validation
        for t in range(predict_n_days):
            output = self.best_models[client_name][0].forecast()
            predictions.append(output)
        return predictions

    def train_models_all_clients(
            self,
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            clients_sorted: pd.DataFrame,
            model_fn
    ) -> tuple:
        """
        Function inputs data of all clients and  trains a model for each
        Parameters
        ----------
        train_data
            pd.DataFrame with all transaction data until the validation date
        val_data
            pd.DataFrame with all transaction data until the test dataset date
        clients_sorted
            sorted list of all clients by Notional requests
        model_fn
            model method chose for training. Configurable, default SARIMAX
        Returns
        -------
        Tuple[List, pd.DataFrame]
            Returns list of best models and their evaluation metrics
        """
        results = []
        best_models = {}
        for client in clients_sorted['Client_Name'][:5]:
            train = get_client_data(client, train_data).Notional.reset_index(drop=True)
            val = get_client_data(client, data=val_data).Notional.reset_index(drop=True)
            if len(val) > 10 and len(train) > 100:
                fn, pred = model_fn(train, val)
                metrics = evaluate(pred, val)
                results.append(metrics)
                if metrics['mape'] < 70:
                    best_models[client] = [fn, pred]
        if len(results) < 1:
            print('No client has enough train/val data to train a model.')
        else:
            self.model_save()
            return best_models, pd.DataFrame(results)

    def model_save(self) -> None:
        if self.best_models:
            with open('data/sarimax_model.pickle', 'w') as f:
                pickle.dump(self.best_models, f)
        else:
            print('No models found.')


def run_model_seasonality_job(jobs_specs: dict) -> tuple:
    # pick a client and get predictions
    CLIENT_NAME = '24009226'
    # train clients on all historical data
    with open('data/sarimax_model_client.pickle', 'rb') as f:
        stats_models = pickle.load(f)
    with open('data/test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)
    # predict next few days on the best models
    Model = StatsModel(model_fn=job_specs['model_method'], FILE_NAME=job_specs['FILE_NAME'])
    Model.best_models[CLIENT_NAME] = stats_models[CLIENT_NAME]
    predictions = Model.predict(client_name=CLIENT_NAME, predict_n_days=len(test_data))
    metrics = evaluate(predictions, test_data['Notional'])
    return predictions, metrics


if __name__ == '__main__':
    # train SARIMAX model
    # stats_model = StatsModel()
    # models, predictions = stats_model.fit()
    # print(predictions)
    run_model_seasonality_job(job_specs)








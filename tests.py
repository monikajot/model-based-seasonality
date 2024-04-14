import pandas as pd
import numpy as np
import pickle
from preprocess_data import remove_outliers, run_stationarity_tests, preprocess, get_raw_data
from start_app import StatsModel


class TestRemoveOutliers:
    def test_no_data(self):
        no_data = pd.DataFrame()
        no_data_with_cols = pd.DataFrame(None, columns = ['Notional'] )
        df1 = remove_outliers(no_data)
        assert(len(df1) == 0)
        assert (df1.empty == True)
        df2 = remove_outliers(no_data_with_cols)
        assert(len(df2) == 0)
        assert (df2.empty == True)
        df_wrong_col = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        df_wrong_col_after = remove_outliers(df_wrong_col)
        assert(df_wrong_col_after.equals(df_wrong_col) )

    def test_with_data(self):
        df_simple = remove_outliers(
            pd.DataFrame({'Reporting_currency_notional': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        )
        df_simple_after = pd.DataFrame({'Reporting_currency_notional': [2, 3, 4, 5, 6, 7, 8, 9]})
        assert(df_simple.reset_index(drop=True).equals(df_simple_after))


class TestStationarityTests:
    def test_no_data(self):
        stationary_clients = run_stationarity_tests(pd.DataFrame())
        assert (stationary_clients == None)

    def test_with_data(self):
        all_data = pd.read_csv('data/all_preprocessed_data.csv')
        stationarity_data = run_stationarity_tests(all_data)
        stationarity_results = pd.DataFrame({
            'Client_Name': pd.Series([24009226, 39624943, 36004363, 42867964, 36066751]),
            'ADF': pd.Series([1, 1, 1, 1, 1]),
            'KPSS': pd.Series([0, 0, 1, 1, 1])
        })
        assert (stationarity_data.equals(stationarity_results))


class TestPreprocessing:
    def test_preprocessing_no_data(self):
        df1 = preprocess(pd.DataFrame(columns=['Transaction_ID', 'Client_Name', 'Date', 'Notional']))
        df2 = preprocess(pd.DataFrame(np.ones((3,3)),columns=['Client_Name', 'Date', 'Notional']))
        assert (df1 == None)
        assert (df2 == None)

    def test_preprocessing_data(self):
        raw_data = get_raw_data('data/trades2020.csv')
        train_data, val_data, clients_sorted, test_data = preprocess(raw_data)
        # with open('test_preprocessing_training_data.pickle', 'wb') as f:
        #     pickle.dump((train_data, val_data, clients_sorted, test_data), f)
        # return
        with open('data/test_preprocessing_training_data.pickle', 'rb') as f:
            train_data_test, val_data_test, clients_sorted_test, test_data_test = \
                pickle.load(f)
        assert (train_data.equals(train_data_test))
        assert (val_data.equals(val_data))
        assert (clients_sorted.equals(clients_sorted))
        assert (test_data.equals(test_data))


class StatsModelTest:
    def test_no_data_fit(self):
        pass

    def test_with_data_fit(self):
        Model = StatsModel()
        raw_data = get_raw_data('data/trades2020.csv')
        best_models, _ = Model.fit(raw_data)
        assert(37707594 in best_models)

    def test_no_data_predict(self):
        pass

    def test_with_data_predict(self):
        pass


if __name__ == '__main__':
    Test = TestRemoveOutliers()
    Test.test_no_data()
    Test.test_with_data()

    Test1 = TestStationarityTests()
    Test1.test_no_data()
    Test1.test_with_data()

    Test2 = TestPreprocessing()
    Test2.test_preprocessing_no_data()
    Test2.test_preprocessing_data()

    TestModel = StatsModelTest()
    TestModel.test_with_data_fit()


import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")


def fill_values(df, window, give_info: bool):
    """
    Fill the NA values of the dataframe according to the running mean of the last [window-1] values.
    If the there are any other NA values remaining we fill them with 0.

    :param df: Dataframe on which we fill the NA values
    :param window: The length of the window to use in the rolling() method of the dataframe
    :param give_info: If True prints out information about the percentage of NA values and how they are filled
    :return: New dataframe
    """

    for col in df.columns[:-1]:  # exclude the label variable
        df[col] = df[col].fillna(df.rolling(window = window,
                                            min_periods = 1).mean()[col])
    new_perc = np.round(100 * df.isna().sum().sum() / (len(df) * len(df.columns)), 2)

    df = df.fillna(0)
    new_perc_2 = np.round(100 * df.isna().sum().sum() / (len(df) * len(df.columns)), 2)

    if give_info:
        print(f'Filling NA values with the running mean of {window - 1} previous values (window = {window}) \n')

        print(f'- The amount of NAs was reduced to {new_perc}% \n')
        print('- For the next step we will fill the remaining NAs with zeroes \n')

        print('Filling NA values with 0 \n')

        print(f'- The amount of NAs was reduced to {new_perc_2}% \n')
        print(f'- No more NA values left \n')

    return df


def n_step_shift(df, shift: int):
    """
    When trying to convert a time series forecasting problem to a supervised one you must use information at t and t+1
    to predict the label at t+1. This is what this function does.

    :param df: Dataframe with time series which we will convert to supervised problem
    :param shift: Number of steps we use to predict the next one.
    :return: New Dataframe
    """
    # df = df.drop(['anomaly', 'test'], axis = 1)
    new_df = df
    for i in range(1, np.abs(shift)+1):
        df_tpn = df.shift(-i).iloc[: -i, :]
        for j in range(len(df_tpn.columns)):
            df_tpn.rename(columns = {(df_tpn.columns[j]): (df_tpn.columns[j] + f'_tp{i}')}, inplace = True)

        new_df = pd.concat([new_df.iloc[:-1, :], df_tpn], axis = 1)

        if f'total_cases_tp{i-1}' in new_df.columns:
            new_df.drop(f'total_cases_tp{i-1}', inplace = True, axis = 1)

        if f'anomaly_tp{i}' in new_df.columns:
            new_df.drop(f'anomaly_tp{i}', inplace = True, axis = 1)

    if 'total_cases' in new_df.columns:
        new_df.drop('total_cases', inplace = True, axis = 1)

    if 'anomaly' in new_df.columns:
        new_df.drop('anomaly', inplace = True, axis = 1)

    return new_df


def fill_future_predictions(predictions, shift: int):
    for i in range(np.abs(shift)):
        value = np.round(np.mean(predictions[-4:])).astype('int32')
        predictions = np.append([predictions], [value])

    return predictions


def make_windowed_data(df, window):
    xin = []
    yout = []
    for i in range(window, len(df)):
        xin.append(df.values[i - window: i, : -1])
        yout.append(df.values[i, -1])

    xin = np.array(xin)
    yout = np.array(yout).reshape(len(yout), 1)

    return xin, yout


def drop_cols(df, threshold):
    """
    Drop columns according to their correlation with the target variable
    :param df: The dataset we use
    :param threshold: Threshold of the correlation
    :return: The columns we drop
    """
    correlations = df.corr()
    columns_2_drop = np.abs(correlations.total_cases.drop('total_cases')) < threshold
    columns_2_drop = np.array(correlations.total_cases.drop('total_cases').index[columns_2_drop])

    return columns_2_drop


def generate_anomalies(df, contamination):
    iforest = IsolationForest(n_estimators = 200, contamination = contamination, random_state = 100)
    yhat = df.iloc[:, -1].values.reshape(-1, 1)
    anomalies = iforest.fit_predict(yhat)
    anomalies[anomalies == 1] = 0
    anomalies[anomalies == - 1] = 1
    df['anomaly'] = anomalies
    mask = anomalies == 1

    return df, mask


def predictions(reg, anomaly_reg, df, anomaly_weight: float, index: int):
    predictions = []
    for i in range(len(df)):
        row = df.iloc[i, :].values.reshape(1, -1)
        label = row[0, row.shape[1] - 1]
        x = row[0, :-1].reshape(1, -1)
        x_sub = row[0, -14-index:-1].reshape(1, -1)
        if label == 1:
            prediction = (anomaly_weight * anomaly_reg.predict(x_sub) + (2-anomaly_weight) * reg.predict(x)) / 2
        else:
            prediction = reg.predict(x)

        prediction = np.round(prediction).astype('int32')
        if prediction < 0: prediction = 0
        predictions.append(prediction)

    return np.array(predictions)


def anomaly_detector_training(x_train, y_train, x_test, y_test, n_estimators, max_depth,
                              learning_rate, verbose = True):

    anomaly_detector = XGBClassifier(n_estimators = n_estimators, booster = 'gbtree',
                                     objective = 'binary:hinge',
                                     max_depth = max_depth, learning_rate = learning_rate,
                                     random_state = 50)

    anomaly_detector.fit(x_train, y_train['anomaly'])

    x_train['predicted_anomaly'] = anomaly_detector.predict(x_train)
    x_test['predicted_anomaly'] = anomaly_detector.predict(x_test)

    y_train['predicted_anomaly'] = 0
    y_test['predicted_anomaly'] = 0

    train_dates = y_train[y_train['anomaly'] == 1].index
    train_dates_to_test = x_train[x_train['predicted_anomaly'] == 1].index

    test_dates = y_test[y_test['anomaly'] == 1].index
    test_dates_to_test = x_test[x_test['predicted_anomaly'] == 1].index

    y_train.loc[y_train.index.isin(train_dates_to_test), 'predicted_anomaly'] = 1
    y_test.loc[y_test.index.isin(test_dates_to_test), 'predicted_anomaly'] = 1

    if verbose:
        print('Train anomalies:', len(train_dates), 'Train predicted anomalies:', len(train_dates_to_test))
        print('Train Anomalies in common:',
              len(y_train[(y_train['predicted_anomaly'] == 1) & (y_train['anomaly'] == 1)]))
        train_f1 = f1_score(y_train['anomaly'], y_train['predicted_anomaly'])
        print(f'Train F1 Score: {np.round(train_f1, 3)} \n')

        print('Test anomalies:', len(test_dates), 'Test predicted anomalies:', len(test_dates_to_test))
        print('Test Anomalies in common:',
              len(y_test[(y_test['predicted_anomaly'] == 1) & (y_test['anomaly'] == 1)]))
        test_f1 = f1_score(y_test['anomaly'], y_test['predicted_anomaly'])
        print(f'Test F1 Score: {np.round(test_f1, 3)}')

    mask1 = (y_train['predicted_anomaly'] == 1).values
    mask2 = (y_test['predicted_anomaly'] == 1).values

    return x_train, y_train, x_test, y_test, mask1, mask2, anomaly_detector


def anomaly_regressor_training(x_train, y_train, x_test, y_test, n_estimators, max_depth,
                               learning_rate, verbose = True):
    anomaly_reg = XGBRegressor(objective = 'reg:absoluteerror',
                               n_estimators = n_estimators, booster = 'gbtree',
                               max_depth = max_depth, learning_rate = learning_rate,
                               random_state = 50)

    anomaly_reg.fit(x_train.values, y_train.values)

    # Get MAE on Train and Test sets
    train_preds = np.round(anomaly_reg.predict(x_train)).astype('int32')
    train_preds[train_preds < 0] = 0

    test_preds = np.round(anomaly_reg.predict(x_test)).astype('int32')
    test_preds[test_preds < 0] = 0

    train_mae = np.round(np.mean(np.abs(train_preds - y_train)), 3)
    test_mae = np.round(np.mean(np.abs(test_preds - y_test)), 3)

    if verbose:
        print(f'Train MAE: {train_mae}')
        print(f'Test MAE: {test_mae}')

    return anomaly_reg

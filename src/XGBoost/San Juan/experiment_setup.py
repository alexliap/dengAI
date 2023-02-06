from src.utils.helper_functions import *

import warnings
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")


def run_experiment(window: int, shift: int, drop_threshold: float, train_amount: float,
                   booster: str, n_estimators: int, max_depth: int, learning_rate: float,
                   contamination: float):

    df = pd.read_csv('../../../San_Juan_train.csv', index_col = 0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace = True)

    df = fill_values(df, window, False)
    df.drop(drop_cols(df, drop_threshold), axis = 1, inplace = True)
    # Generate anomaly column for spike detection
    df, mask = generate_anomalies(df, contamination)

    new_df = n_step_shift(df, shift)

    train_amount = train_amount
    cutoff = int(train_amount * len(df))

    # Unshifted Datasets
    un_x_train = df.iloc[:cutoff, 2: -2]
    un_x_test = df.iloc[cutoff:, 2: -2]

    un_y_train = df.iloc[:cutoff, -2:]
    un_y_test = df.iloc[cutoff:, -2:]

    # Shifted Datasets
    df_x_train = new_df.iloc[:cutoff, : -1]
    df_x_test = new_df.iloc[cutoff:, : -1]

    df_y_train = new_df.iloc[:cutoff, -1:]
    df_y_test = new_df.iloc[cutoff:, -1:]

    un_x_train, un_y_train, un_x_test, un_y_test, mask1, mask2, anomaly_detector = anomaly_detector_training(un_x_train,
                                                                                                             un_y_train,
                                                                                                             un_x_test, un_y_test,
                                                                                                             n_estimators = 10,
                                                                                                             max_depth = 3,
                                                                                                             learning_rate = 0.1,
                                                                                                             verbose = False)

    new_x_train = un_x_train[un_x_train['predicted_anomaly'] == 1].drop('predicted_anomaly', axis = 1)
    new_y_train = un_y_train[un_y_train['predicted_anomaly'] == 1].iloc[:, 0]

    new_x_test = un_x_test[un_x_test['predicted_anomaly'] == 1].drop('predicted_anomaly', axis = 1)
    new_y_test = un_y_test[un_y_test['predicted_anomaly'] == 1].iloc[:, 0]

    anomaly_reg = anomaly_regressor_training(new_x_train, new_y_train, new_x_test, new_y_test, n_estimators = 50,
                                             max_depth = 4, learning_rate = 1, verbose = False)

    # Shift the anomalies columns
    shift = np.abs(shift).astype('int32')
    train_shifted_anomalies = pd.concat([un_y_train['predicted_anomaly'][shift:],
                                         un_y_test['predicted_anomaly'][:shift]],
                                        axis = 0)
    train_shifted_anomalies.index = df_x_train.index

    test_shifted_anomalies = un_y_test['predicted_anomaly'][shift:]
    test_shifted_anomalies.index = df_x_test.index

    assert len(train_shifted_anomalies) == len(df_x_train)
    assert len(test_shifted_anomalies) == len(df_x_test)

    # Add anomaly column to X datasets
    df_x_train = pd.concat([df_x_train, train_shifted_anomalies], axis = 1)
    df_x_test = pd.concat([df_x_test, test_shifted_anomalies], axis = 1)

    with mlflow.start_run():
        model_1 = XGBRegressor(n_estimators = n_estimators, booster = booster, subsample = 1, tree_method = 'exact',
                               objective = 'reg:absoluteerror', max_depth = max_depth, learning_rate = learning_rate,
                               nthread = 4, gamma = 0, random_state = 50)
        # Fit model
        model_1.fit(df_x_train.iloc[:, :-1], df_y_train)

        # Get MAE on Train and Test sets
        train_preds = predictions(model_1, anomaly_reg, df_x_train)
        test_preds = predictions(model_1, anomaly_reg, df_x_test)

        train_mae = np.round(np.mean(np.abs(train_preds - df_y_train.values)), 3)
        test_mae = np.round(np.mean(np.abs(test_preds - df_y_test.values)), 3)
        # --------------- Logging --------------- #
        mlflow.log_param('Window', window)
        mlflow.log_param('Shift', shift)
        mlflow.log_param('Train amount', train_amount)
        mlflow.log_param('Drop Threshold', drop_threshold)
        mlflow.log_param('Booster', model_1.booster)
        mlflow.log_param('Number of Estimators', model_1.n_estimators)
        mlflow.log_param('Max Depth', model_1.max_depth)
        mlflow.log_param('Gamma', model_1.gamma)
        mlflow.log_param('Learning Rate', model_1.learning_rate)

        mlflow.log_metric('Train MAE', train_mae)
        mlflow.log_metric('Test MAE', test_mae)

        mlflow.xgboost.log_model(model_1, 'XGBoostRegressor', model_format = 'xgb')
        mlflow.xgboost.log_model(anomaly_reg, 'Anomaly Regressor', model_format = 'xgb')
        mlflow.xgboost.log_model(anomaly_detector, 'Anomaly Detector', model_format = 'xgb')
        # --------------- Logging --------------- #

        fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (14, 5))

        train_title = (f'Train MAE: {train_mae} \n',
                       f'Train Target Mean: {np.round(df_y_train.values.mean(), 3)}',
                       f'Predictions Mean: {np.round(np.mean(train_preds), 3)}')
        axs[0].set_title('\n'.join(train_title))
        axs[0].plot(df_y_train)
        axs[0].plot(df_y_train.index, train_preds)

        test_title = (f'Test MAE: {test_mae} \n',
                      f'Test Target Mean: {np.round(df_y_test.values.mean(), 3)}',
                      f'Predictions Mean: {np.round(np.mean(test_preds), 3)}')
        axs[1].set_title('\n'.join(test_title))
        axs[1].plot(df_y_test)
        axs[1].plot(df_y_test.index, test_preds)

        plt.savefig('predictions_viz.png')

        mlflow.log_artifact('predictions_viz.png')

    print('Experiment completed')

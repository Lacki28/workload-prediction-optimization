import time
from datetime import timezone, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from matplotlib import ticker
from sklearn.ensemble import RandomForestRegressor


def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")


def naive_ratio(t, prediction, real_value):
    # Compute the absolute difference between corresponding elements of a and b
    prediction_nr = prediction[t:]
    real_value_nr = real_value[:-t]
    abs_diff_et1 = torch.abs(prediction_nr - real_value_nr)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / len(prediction_nr)) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / (et1)

def calculate_prediction_results(t, pred_cpu_test, act_cpu_test, file_path, start_time,
                                 training_time):
    for i in range(t):
        append_to_file(file_path, str(i + 1) + " timestamp ahead prediction")
        current_act_cpu_test = act_cpu_test[:, i]
        current_pred_cpu_test = pred_cpu_test[:, i]
        calc_MSE_Accuracy(t, current_act_cpu_test, current_pred_cpu_test, file_path, start_time, training_time)
def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, torch.from_numpy(y_test_pred), torch.tensor(y_test))
    append_to_file(file_path, "mae & mse & r2 & nr & training & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(r2) + " & " + str(
                       np.round(nr.numpy(), decimals=5)) + " & " + str(training_time) + " & " + str(
                       round((time.time() - start_time), 2)))



def plot_results(t, sequence_length, df, actual_values_cpu, predictions_cpu, target, max_depth):
    indices = pd.DatetimeIndex(df["start_time"])
    indices = indices.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = indices[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    indices = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
               [first_timestamp + i * increment for i in range(len(indices))]]
    indices = indices[int(len(df) * 0.7) + t - 1:]
    indices = [str(period) for period in indices]

    for i in range(t):
        current_predictions_cpu = predictions_cpu[:, i]
        current_actual_values_cpu = actual_values_cpu[:, i]
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
        axs.plot(indices, current_actual_values_cpu, label='actual ' + target, linewidth=1, color='orange')
        axs.plot(indices, current_predictions_cpu, label='predicted ' + target, linewidth=1, color='blue', linestyle='dashed')
        axs.set_xlabel('Time', fontsize=18)
        plt.xticks(rotation=45)  # 'vertical')
        plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
        axs.set_ylabel(target, fontsize=18)
        axs.set_title('Random forest ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(i+1), fontsize=20)
        axs.legend(fontsize=16)
        plt.savefig(str(max_depth)+'_rf_' + 'h' + str(sequence_length) + '_t' + str(i+1) + '' + '.png')




def create_sliding_window(t, sequence_length, x_data, y_data):
    X = []
    y = []

    for i in range(0, len(x_data) - sequence_length):
        if i < sequence_length:
            padding = np.tile(np.array([x_data.iloc[0]]), sequence_length - i - 1).flatten()
            x_window = x_data.iloc[0:i + 1].values.flatten()
            x_window = np.concatenate((padding, x_window), axis=0)
        else:
            x_window = x_data.iloc[i - sequence_length + 1:i + 1].values.flatten()

        y_window = y_data.iloc[i + 1:i + t + 1].values.flatten()
        X.append(x_window)
        y.append(y_window)

    X = np.array(X)
    y = np.array(y)
    return X, y


def main(t=2, sequence_length=12, target="mean_CPU_usage", features="mean_CPU_usage", trees=200, max_depth=3):
    file_path = 'rf.txt'
    start_time = time.time()
    df = pd.read_csv("../../sortedGroupedJobFiles/3418324.csv", sep=",")
    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length))
    append_to_file(file_path, 'trees=' + str(trees) + ', max depth=' + str(max_depth))
    # create correct index
    df.index = pd.DatetimeIndex(df["start_time"])
    df.index = df.index.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = df.index[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    df.index = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                [first_timestamp + i * increment for i in range(len(df))]]
    # split into training and test set - check until what index the training data is
    test_head = int(len(df) * 0.7)
    df_train = df.iloc[:test_head, :]
    df_test = df.iloc[test_head - sequence_length:, :]
    target = [target]
    features = [features]
    X_train, y_train = create_sliding_window(t, sequence_length, df_train[features], df_train[target])
    X_test, y_test = create_sliding_window(t, sequence_length, df_test[features], df_test[target])
    samples, sequences, features = X_train.shape
    X_train = np.reshape(X_train, (samples, sequences * features))
    samples, sequences, features = X_test.shape
    X_test = np.reshape(X_test, (samples, sequences * features))
    # X has the size observations[sequences[features]] - it needs to be reshaped to observations[sequences*features]
    regressor = RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=0)
    joblib.dump(regressor, 'random_forest_' + str(sequence_length) + '_sequence_length_' + str(t) + '_max_depth_' + str(
        max_depth) + "_trees_" + str(trees) + '.pkl')
    # y_train = np.ravel(y_train)  # transform y_train into one dimensional array
    X_train = X_train.squeeze()
    y_train = y_train.squeeze()
    if sequence_length == 1:
        X_train = [[item] for item in X_train]
    regressor.fit(X_train, y_train)
    training_time = round((time.time() - start_time), 2)
    # Predict on new data
    y_prediction = regressor.predict(X_test)
    y_test = y_test.squeeze()
    calculate_prediction_results(t, y_prediction, y_test, file_path, start_time,
                                training_time)
    plot_results(t, sequence_length, df, y_test, y_prediction, target[0], max_depth)


if __name__ == "__main__":
    for history in (1, 6, 12):
        for max_depth in (3, 4):
            main(6, history, 'mean_CPU_usage', 'mean_CPU_usage', 150, max_depth)

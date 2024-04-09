import os
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
    for job_index in range(len(act_cpu_test)):
        for i in range(t):
            append_to_file(file_path, str(i + 1) + " timestamp ahead prediction")
            current_act_cpu_test = act_cpu_test[job_index][:, i]
            current_pred_cpu_test = pred_cpu_test[job_index][:, i]
            calc_MSE_Accuracy(t, current_act_cpu_test.squeeze(), current_pred_cpu_test, file_path,
                              start_time, training_time)


def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, torch.from_numpy(y_test_pred), torch.tensor(y_test))
    # append_to_file(file_path, "mae & mse & r2 & nr & training & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(r2) + " & " + str(
                       np.round(nr.numpy(), decimals=5)) + " & " + str(training_time) + " & " + str(
                       round((time.time() - start_time), 2)))


def plot_results(t, sequence_length, df, actual_values_cpu, predictions_cpu, target):
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
        axs.plot(indices, current_actual_values_cpu, label='actual ' + target, linewidth=1, color='orange')
        axs.plot(indices, current_predictions_cpu, label='predicted ' + target, linewidth=1, color='blue',
                 linestyle='dashed')
        axs.set_xlabel('Time')
        plt.xticks(rotation=45)  # 'vertical')
        plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
        axs.set_ylabel(target)
        axs.set_title('Random forest ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(i + 1))
        axs.legend()
        plt.savefig('rf' + 'h' + str(sequence_length) + '_t' + str(i + 1) + '' + '.png')


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

def read_file_names(file_path, path, index_start, index_end):

    dir = "~/Documents/pythonScripts/" + path + "/"
    expanded_path = os.path.expanduser(dir)
    g0 = os.listdir(expanded_path)
    g0 = g0[index_start: index_end]
    g0_files = [expanded_path + filename for filename in g0]
    append_to_file(file_path, "jobs group " + path)
    append_to_file(file_path, str(g0))
    return g0_files


def read_file_names_test(path, nr_0):
    dir = "~/Documents/pythonScripts/" + path
    expanded_path = os.path.expanduser(dir)
    g0 = os.listdir(expanded_path)
    g0 = g0[:nr_0]
    g0_files = [expanded_path + filename for filename in g0]

    return g0_files


def read_files(training_files, t, sequence_length, features, target):
    training_files_csv_x = list()
    training_files_csv_y = list()
    for file in training_files:
        df = pd.read_csv(file, sep=",")
        df.index = pd.DatetimeIndex(df["start_time"])
        df.index = df.index.tz_localize(timezone.utc).tz_convert('US/Eastern')
        first_timestamp = df.index[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
        increment = timedelta(minutes=5)
        df.index = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                    [first_timestamp + i * increment for i in range(len(df))]]
        X_train, y_train = create_sliding_window(t, sequence_length, df[features], df[target])
        training_files_csv_x.append(X_train)
        training_files_csv_y.append(y_train)

    return training_files_csv_x, training_files_csv_y


def predict(test_files, t, sequence_length, features, target, start_time,
            training_time, dir, regressor):
    test_files_csv_x, test_files_csv_y = read_files(test_files, t, sequence_length, features, target)

    test_files_csv_x_reshaped = [np.reshape(arr, (-1, 1)).tolist() for arr in test_files_csv_x]
    y_predictions = list()
    for test_files_csv_x in test_files_csv_x_reshaped:
        y_prediction = regressor.predict(test_files_csv_x)
        y_predictions.append(y_prediction)

    calculate_prediction_results(t, y_predictions, test_files_csv_y, "rf_" + dir + "_.txt", start_time,
                                 training_time)


def main(t=2, sequence_length=12, target="mean_CPU_usage", features="mean_CPU_usage", trees=200, max_depth=3):
    file_path = 'rf.txt'
    start_time = time.time()
    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length))
    append_to_file(file_path, 'trees=' + str(trees) + ', max depth=' + str(max_depth))
    training_files = read_file_names(file_path, "0", 0, 50)
    validation_files = read_file_names(file_path, "0", 50, 100)
    test_files = read_file_names(file_path, "1", 0, 50)

    training_files_csv_x, training_files_csv_y = read_files(training_files, t, sequence_length, features, target)

    regressor = RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=0)

    reshaped_training_files = [np.reshape(arr, (-1, 1)) for arr in training_files_csv_x]
    stacked_inputs = np.vstack(reshaped_training_files)
    training_files_csv_y = np.concatenate(training_files_csv_y, axis=0)  # Flatten the nested arrays
    training_files_csv_y = np.reshape(training_files_csv_y, (-1, 6))  # Reshape to have 6 elements per inner array
    regressor.fit(stacked_inputs, training_files_csv_y)

    # [[x1],[x2],[x3],..] [[y11,y12,y13,y14,y15,y16],[y21,y22,y23,y24,y25,y26]...]

    print("DONE")
    joblib.dump(regressor, 'random_forest_' + str(sequence_length) + '_sequence_length_' + str(t) + '_max_depth_' + str(
        max_depth) + "_trees_" + str(trees) + '.pkl')
    training_time = round((time.time() - start_time), 2)

    predict(validation_files, t, sequence_length, features, target, start_time,
            training_time, "validation", regressor)

    predict(test_files, t, sequence_length, features, target, start_time,
            training_time, "test", regressor)


if __name__ == "__main__":
    main(6, 1, ['mean_CPU_usage'], ['mean_CPU_usage'], 200, 4)
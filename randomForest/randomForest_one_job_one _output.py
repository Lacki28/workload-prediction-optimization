import time
from datetime import timezone, timedelta

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


def plot_results(t, sequence_length, df, y_test, y_prediction, target):
    indices = df.index
    indices = indices[int(len(df) * 0.7) + t - 1:]
    indices = [str(period) for period in indices]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    axs.plot(indices, y_test[:, 0], label='actual ' + target, linewidth=1, color='orange')
    axs.plot(indices, y_prediction, label='predicted ' + target, linewidth=1, color='blue', linestyle='dashed')
    axs.set_xlabel('Time')
    plt.xticks(rotation=45)  # 'vertical')
    plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
    axs.set_ylabel(target)
    axs.set_title('Random forest ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(t))
    axs.legend()
    plt.savefig('rf' + 'h' + str(sequence_length) + '_t' + str(t) + '' + '.png')
    plt.show()


def create_sliding_window(t, sequence_length, x_data, y_data):
    X = []
    y = []
    for i in range(sequence_length, len(x_data) - t + 1):
        X.append(x_data.values[i - sequence_length:i])
        y.append(y_data.values[i + t - 1])
    X = np.array(X)
    y = np.array(y)
    return X, y


def main(t=2, sequence_length=12, target="mean_CPU_usage", features="mean_CPU_usage", trees=200, max_depth=3):
    file_path = 'rf.txt'
    start_time = time.time()
    df = pd.read_csv("../sortedGroupedJobFiles/3418324.csv", sep=",")
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
    y_train = np.ravel(y_train)  # transform y_train into one dimensional array
    regressor.fit(X_train, y_train)
    training_time = round((time.time() - start_time), 2)
    # Predict on new data
    y_prediction = regressor.predict(X_test)
    calc_MSE_Accuracy(t, y_test, y_prediction, file_path, start_time, training_time)
    plot_results(t, sequence_length, df, y_test, y_prediction, target[0])


if __name__ == "__main__":
    for t in (1, 2, 3, 12):
        for history in (1, 12, 288):
            for trees in (150, 200):
                for max_depth in (2, 3, 4):
                    if t == 12 and history == 1:
                        main(t, 24, 'mean_CPU_usage', 'mean_CPU_usage', trees, max_depth)
                    else:
                        main(t, history, 'mean_CPU_usage', 'mean_CPU_usage', trees, max_depth)

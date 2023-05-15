from datetime import timezone, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from matplotlib import ticker
from sklearn.ensemble import RandomForestRegressor


def naive_ratio(n, prediction, real_value):
    # Compute the absolute difference between corresponding elements of a and b
    prediction_nr = prediction[n:]
    real_value_nr = real_value[:-n]
    abs_diff_et1 = torch.abs(prediction_nr - real_value_nr)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / len(prediction_nr)) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / (et1)


def calc_MSE_Accuracy(n, y_test, y_test_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 5))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 5))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 5))
    nr = naive_ratio(n, torch.from_numpy(y_test_pred), torch.tensor(y_test))
    print("Naive ratio =", nr)


def plot_results(n, sequence_length, df, y_test, y_prediction, target):
    indices = df.index
    indices = indices[int(len(df) * 0.7) + n - 1:]
    indices = [str(period) for period in indices]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    axs.plot(indices, y_test[:, 0], label='actual ' + target, linewidth=1, color='orange')
    axs.plot(indices, y_prediction, label='predicted ' + target, linewidth=1, color='blue', linestyle='dashed')
    axs.set_xlabel('Time')
    plt.xticks(rotation=45)#'vertical')
    plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
    axs.set_ylabel(target)
    axs.set_title('Random forest ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(n))
    axs.legend()
    plt.savefig('RF_' + 'h' + str(sequence_length) + '_t' + str(n) + '' + '.png')
    plt.show()


def create_sliding_window(n, sequence_length, x_data, y_data):
    X = []
    y = []
    for i in range(sequence_length, len(x_data) - n + 1):
        X.append(x_data.values[i - sequence_length:i])
        y.append(y_data.values[i + n - 1])
    X = np.array(X)
    y = np.array(y)
    return X, y


def main(n=1, sequence_length=12, target="mean_CPU_usage", features="mean_CPU_usage", trees=200, max_depth=3):
    df = pd.read_csv("../sortedGroupedJobFiles/3418324.csv", sep=",")
    print("n=" + str(n) + ", sequence length=" + str(sequence_length))
    print('trees=' + str(trees) + ', max depth=' + str(max_depth))
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
    X_train, y_train = create_sliding_window(n, sequence_length, df_train[features], df_train[target])
    X_test, y_test = create_sliding_window(n, sequence_length, df_test[features], df_test[target])
    samples, sequences, features = X_train.shape
    X_train = np.reshape(X_train, (samples, sequences * features))
    samples, sequences, features = X_test.shape
    X_test = np.reshape(X_test, (samples, sequences * features))
    # X has the size observations[sequences[features]] - it needs to be reshaped to observations[sequences*features]
    regressor = RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=0)
    y_train = np.ravel(y_train)  # transform y_train into one dimensional array
    regressor.fit(X_train, y_train)
    # Predict on new data
    y_prediction = regressor.predict(X_test)
    calc_MSE_Accuracy(n, y_test, y_prediction)
    plot_results(n, sequence_length, df, y_test, y_prediction, target[0])


if __name__ == "__main__":
    main()

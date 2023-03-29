import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import sklearn.metrics as sm

# hyperparameters
sequence_length = 2
n = 1
trees = 100
max_depth = 30


def naive_ratio(prediction, real_value, size):
    # Compute the absolute difference between corresponding elements of a and b
    abs_diff_et1 = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / size) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / size) * sum_abs_diff
    return et / (et1 + 0.000000000001)


def calc_MSE_Accuracy(y_test, y_test_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 5))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 5))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 5))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 5))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 5))
    nr = naive_ratio(torch.from_numpy(y_test_pred), torch.tensor(y_test), len(y_test))
    print("Naive ratio =", nr)


def plot_results(df, y_test, y_prediction, train_size):
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index_in_hours = ((df['start_time'] - 600000000) / in_hours)
    index_test = index_in_hours.iloc[train_size + sequence_length:len(df) - n + 1]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].plot(index_test, y_test[:, 0], label='actual CPU usage', linewidth=1,
                markerfacecolor='blue')
    axs[0].plot(index_test, y_prediction[:, 0], label='predicted CPU', linewidth=1, markerfacecolor='red')
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('CPU prediction')
    axs[0].set_title('Mean CPU prediction')
    axs[0].legend()

    axs[1].plot(index_test, y_test[:, 1], label='actual memory usage', linewidth=1,
                markerfacecolor='blue')
    axs[1].plot(index_test, y_prediction[:, 1], label='predicted disk IO time', linewidth=1, markerfacecolor='red')
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Mean disk IO time')
    axs[1].set_title('Disk IO time prediction')
    axs[1].legend()

    plt.show()


def create_sliding_window(x_data, y_data):
    X = []
    y = []
    for i in range(sequence_length, len(x_data) - n + 1):
        X.append(x_data.values[i - sequence_length:i])
        y.append(y_data.values[i + n - 1])
    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    df = pd.read_csv("job_smaller.csv", sep=",")
    # split into training and test set - check until what index the training data is

    test_head = df.index[int(0.8 * len(df))]
    df_train = df.loc[:test_head - 1, :]
    df_test = df.loc[test_head:len(df), :]
    features = ['start_time', 'mean_CPU_usage', 'mean_disk_IO_time', 'unmapped_page_cache_mem_usage']
    target = ["mean_CPU_usage", 'mean_disk_IO_time']

    X_train, y_train = create_sliding_window(df_train[features], df_train[target])
    X_test, y_test = create_sliding_window(df_test[features], df_test[target])
    samples, sequences, features = X_train.shape
    X_train = np.reshape(X_train, (samples, sequences * features))
    samples, sequences, features = X_test.shape
    X_test = np.reshape(X_test, (samples, sequences * features))

    # X has the size observations[sequences[features]] - it needs to be reshaped to observations[sequences*features]
    # MultiOutputRegressor fits one random forest for each target. Each tree inside then is predicting one of your outputs. Without the wrapper, RandomForestRegressor fits trees targeting all the outputs at once.

    regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=0))

    regressor.fit(X_train, y_train)
    # Predict on new data
    y_prediction = regressor.predict(X_test)
    calc_MSE_Accuracy(y_test, y_prediction)

    plot_results(df, y_test, y_prediction, len(df_train))


if __name__ == "__main__":
    main()

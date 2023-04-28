import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import sklearn.metrics as sm

# hyperparameters
sequence_length = 10
n = 1
trees = 200
max_depth = 2

def naive_ratio(prediction, real_value):
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


def calc_MSE_Accuracy(y_test, y_test_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 5))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 5))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 5))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 5))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 5))
    nr = naive_ratio(torch.from_numpy(y_test_pred), torch.tensor(y_test))
    print("Naive ratio =", nr)
def plot_results(data, indices):
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index_list = []
    for index in indices:
        index_list.append(((index - 600000000) / in_hours))
    fig, axs = plt.subplots(nrows=len(data), ncols=2, figsize=(10, 5))
    for i in range(len(index_list)):
        if(i ==0):
            cpu_pred = np.array(data[i][0])[:, 0]
            memory_pred = np.array(data[i][0])[:, 1]
            cpu_act = np.array(data[i][1])[:, 0]
            memory_act = np.array(data[i][1])[:, 1]
            axs[0].plot(index_list[i], cpu_act, label='actual CPU usage', linewidth=1,
                           markerfacecolor='blue')
            axs[0].plot(index_list[i],cpu_pred, label='predicted CPU', linewidth=1, markerfacecolor='red')
            axs[0].set_xlabel('Time (hours)')
            axs[0].set_ylabel('CPU prediction')
            axs[0].set_title('Mean CPU prediction')
            axs[0].legend()

            axs[1].plot(index_list[i], memory_act, label='actual memory usage', linewidth=1,
                           markerfacecolor='blue')
            axs[1].plot(index_list[i], memory_pred, label='predicted memory', linewidth=1, markerfacecolor='red')
            axs[1].set_xlabel('Time (hours)')
            axs[1].set_ylabel('Memory prediction')
            axs[1].set_title('Mean memory prediction')
            axs[1].legend()
    plt.savefig('8_training_sets.png')
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

def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        df_list.append(pd.read_csv(file, sep=","))
    return df_list

def main():
    training_data_files = read_data("../training")
    test_data_files = read_data("../test")
    xy_train =[]
    xy_test=[]
    features = ['start_time', 'mean_CPU_usage', 'canonical_mem_usage', 'assigned_mem_usage', 'max_mem_usage',
                'mean_local_disk_space_used', 'max_CPU_usage', 'nr_of_tasks', 'scheduling_class']
    target = ["mean_CPU_usage", 'canonical_mem_usage']
    for file in training_data_files:
        X_train, y_train = create_sliding_window(file[features], file[target])
        samples, sequences, features_size = X_train.shape
        X_train = np.reshape(X_train, (samples, sequences * features_size))
        xy_train.append((X_train, y_train))
    for file in test_data_files:
        X_test, y_test = create_sliding_window(file[features], file[target])
        samples, sequences, features_size = X_test.shape
        X_test = np.reshape(X_test, (samples, sequences * features_size))
        xy_test.append((X_test, y_test))

    # X has the size observations[sequences[features]] - it needs to be reshaped to observations[sequences*features]
    # MultiOutputRegressor fits one random forest for each target. Each tree inside then is predicting one of your outputs. Without the wrapper, RandomForestRegressor fits trees targeting all the outputs at once.

    regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=0))
    for (X_train, y_train) in xy_train:
        regressor.fit(X_train, y_train)
    # Predict on new data
    data=[]
    indices=[]
    for (X_test, y_test) in xy_test:
        y_prediction =regressor.predict(X_test)
        data.append((y_prediction, y_test))
        indices.append(X_test[:,0])
        calc_MSE_Accuracy(y_test, y_prediction)
    plot_results(data, indices)


if __name__ == "__main__":
    main()

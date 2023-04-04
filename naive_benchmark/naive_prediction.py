import math
import os
from functools import partial

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from scipy.signal import savgol_filter
from torch.utils.data import Dataset, DataLoader

# hyperparameters
sequence_length = 8  # I want to make a prediction based on how many values before
n = 2 # how many timestamps after I want to predict - example: n=1, sequ =3: x=[1,2,3],y=[4]
epochs = 500
features = ['start_time', 'mean_CPU_usage', 'canonical_mem_usage', 'assigned_mem_usage', 'max_mem_usage',
            'mean_local_disk_space_used', 'max_CPU_usage', 'nr_of_tasks', 'scheduling_class']
target = ["mean_CPU_usage", 'canonical_mem_usage']


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        # shape length, -sequence length, because for the first I do not have any predecessors
        # - n + 1, because I might predict a few timestamps ahead - therefore I may not predict some at the beginning
        return self.X.shape[0] - self.sequence_length - n + 1

    # returns the input sequence and the target value
    def __getitem__(self, i):
        # start at element i and go to element i+sequence length, the result is "sequence length many" rows
        x = self.X[i:(i + self.sequence_length), :]
        # start at the last element of x (sequence length +i) and predict n timestamps ahead and subtract -1
        return x, self.y[i + self.sequence_length + n - 1]


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


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


def my_loss_fn(output, target):
    loss = 0
    loss += mse(output, target)
    # loss += naive_ratio(output, target, size)
    return loss


def my_accuracy_fn(output, target):
    r2 = sm.r2_score(target, output)
    if math.isnan(r2):
        return - math.inf
    return round(r2)


def predict(data_loader, model, device):
    output1 = torch.tensor([])
    output2 = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_prediction_multiple = model(X)
            y_prediction_1 = y_prediction_multiple[0]
            y_prediction_2 = y_prediction_multiple[1]
            output1 = torch.cat((output1, y_prediction_1), 0)
            output2 = torch.cat((output2, y_prediction_2), 0)
    return output1, output2


def calc_MSE_Accuracy(y_test, y_test_pred, index):
    mae = []
    mse = []
    r2 = []
    nr = []
    for i in range(len(y_test)):
        y_test_pred[i] = (y_test_pred[i][0].squeeze(), y_test_pred[i][1].squeeze())
        mae.append(round(sm.mean_absolute_error(y_test[i][index], y_test_pred[i][index])))
        mse.append(round(sm.mean_squared_error(y_test[i][index], y_test_pred[i][index])))
        r2.append(round(sm.r2_score(y_test[i][index], y_test_pred[i][index])))
        nr.append(naive_ratio(y_test_pred[i][index], y_test[i][index]))
    print("Mean absolute error =", (sum(mae) / len(mae)))
    print("Mean squared error =", (sum(mse) / len(mse)))
    print("R2 score =", (sum(r2) / len(r2)))
    print("Naive ratio =", (sum(nr) / len(nr)))


def calculate_prediction_results(prediction_test, actual_test_values, prediction_training, actual_train_values):
    print("TRAIN ERRORS CPU:")
    calc_MSE_Accuracy(actual_train_values, prediction_training, 0)
    print("TRAIN ERRORS MEM:")
    calc_MSE_Accuracy(actual_train_values, prediction_training, 1)
    print("TEST ERRORS CPU:")
    calc_MSE_Accuracy(actual_test_values, prediction_test, 0)
    print("TEST ERRORS MEM:")
    calc_MSE_Accuracy(actual_test_values, prediction_test, 1)


def plot_results(predictions, original_test_files):
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    actual_test_values = []
    index_list = []
    for test_file in original_test_files:
        # actual results needs to have the same size as the prediction
        start_train_index = sequence_length + n - 1
        test_file.X = test_file.X[start_train_index:]
        test_file.y = test_file.y[start_train_index:]
        actual_test_values.append((test_file.y[:, 0], test_file.y[:, 1]))
        indices = ((test_file.X[:, 0] - 600000000) / in_hours)  # first index is timestamp
        index_list.append(indices)
    fig, axs = plt.subplots(nrows=2, ncols=len(index_list), figsize=(10, 5))
    for i in range(len(index_list)):
        axs[i][0].plot(index_list[i], actual_test_values[i][0], label='actual CPU usage', linewidth=1,
                       markerfacecolor='blue')
        axs[i][0].plot(index_list[i], predictions[i][0], label='predicted CPU', linewidth=1, markerfacecolor='red')
        axs[i][0].set_xlabel('Time (hours)')
        axs[i][0].set_ylabel('CPU prediction')
        axs[i][0].set_title('Mean CPU prediction')
        axs[i][0].legend()

        axs[i][1].plot(index_list[i], actual_test_values[i][1], label='actual memory usage', linewidth=1,
                       markerfacecolor='blue')
        axs[i][1].plot(index_list[i], predictions[i][1], label='predicted memory', linewidth=1, markerfacecolor='red')
        axs[i][1].set_xlabel('Time (hours)')
        axs[i][1].set_ylabel('Memory prediction')
        axs[i][1].set_title('Mean memory prediction')
        axs[i][1].legend()
    plt.savefig('8_training_sets.png')
    plt.show()


def get_prediction_results(test_data_files_sequence):
    prediction_test = []
    for test_dataset in test_data_files_sequence:
        # in a niave model - the prediction = the last actual value of the sequence
        start_train_index = sequence_length + n - 1
        prediction_test_cpu = test_dataset.y[:, 0][start_train_index - n:-n]
        prediction_test_mem = test_dataset.y[:, 1][start_train_index - n:-n]
        prediction_test.append((prediction_test_cpu, prediction_test_mem))

    actual_test_values = []
    for sequence_file in test_data_files_sequence:
        # actual results needs to have the same size as the prediction
        start_train_index = sequence_length + n - 1
        actual_test_cpu = sequence_file.y[:, 0][start_train_index:]
        actual__test_mem = sequence_file.y[:, 1][start_train_index:]
        actual_test_values.append((actual_test_cpu, actual__test_mem))

    return prediction_test, actual_test_values


def get_test_training_data(test_data_files=None, training_data_files=None):
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable
    train_sequence_dataset = []
    test_sequence_dataset = []
    for df_train in training_data_files:
        train_sequence_dataset.append(SequenceDataset(
            df_train,
            target=target,
            features=features,
            sequence_length=sequence_length
        ))
    for df_test in test_data_files:
        test_sequence_dataset.append(SequenceDataset(
            df_test,
            target=target,
            features=features,
            sequence_length=sequence_length
        ))

    return test_sequence_dataset, train_sequence_dataset


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        df_list.append(pd.read_csv(file, sep=","))
    return df_list


def main():
    training_data_files = read_data("../training")
    test_data_files = read_data("../test")

    test_data_files_sequence, training_data_files_sequence = get_test_training_data(test_data_files,
                                                                                    training_data_files)
    print("Get test results")
    prediction_test, actual_test_values = get_prediction_results(test_data_files_sequence)
    print("Get training results")
    prediction_training, actual_train_values = get_prediction_results(training_data_files_sequence)
    print("calculate results")
    calculate_prediction_results(prediction_test, actual_test_values, prediction_training, actual_train_values)
    plot_results(prediction_test, test_data_files_sequence)


if __name__ == "__main__":
    main()

import math
import os
import time
from datetime import timezone, timedelta

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from matplotlib import pyplot as plt, ticker
from torch.utils.data import Dataset

min_max_dict = {}


# use a sequence of observations for the prediction
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, t, sequence_length=5):
        self.features = features
        self.target = target
        self.t = t
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.t

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i+1: i + self.t+1]  # return target n time stamps ahead


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


def naive_ratio(t, prediction, real_value):
    prediction = torch.from_numpy(prediction)
    real_value = torch.from_numpy(real_value)
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
    return et / et1


def my_loss_fn(output, target):
    loss = 0
    loss += mse(output, target)
    return loss


def my_r2_fn(output, target):
    r2 = sm.r2_score(target, output)
    if math.isnan(r2):
        return - math.inf
    return r2


def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")


def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, y_test_pred, y_test)
    append_to_file(file_path, "mae & mse & r2 & nr & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(r2) + " & " + str(
                       np.round(nr.numpy(), decimals=5)) + " & " + str(
                       round((time.time() - start_time), 2)))


def normalize_data_minMax(features, df):
    pd.options.mode.chained_assignment = None
    # find min and max
    for c in df.columns:
        if c in features:
            min = min_max_dict[c]['min']
            max = min_max_dict[c]['max']
            value_range = max - min
            df.loc[:, c] = (df.loc[:, c] - min) / value_range
    return df


def calculate_prediction_results(t, pred_cpu, act_cpu, start_time, file_path):
    for job_index in range(len(act_cpu)):
        append_to_file(file_path, str(t) + " timestamp ahead prediction")

        current_act_cpu_train = act_cpu[job_index].values
        current_pred_cpu_train = pred_cpu[job_index].values

        calc_MSE_Accuracy(t, current_act_cpu_train, current_pred_cpu_train, file_path, start_time)


def get_prediction_results(sequence_length, t, test_datasets, target):
    prediction_test_cpus = list()
    actual_test_cpus = list()
    for test_dataset in test_datasets:
        # in a naive model - the prediction = the last actual value of the sequence
        start_train_index = sequence_length + t - 1
        prediction_test_cpu = test_dataset[target][start_train_index - t:-t]
        prediction_test_cpus.append(prediction_test_cpu)
        # actual results needs to have the same size as the prediction
        actual_test_cpu = test_dataset[target][start_train_index:]
        actual_test_cpus.append(actual_test_cpu)

    return prediction_test_cpus, actual_test_cpus


def read_file_names(file_path, path, index_start, index_end):
    dir = "~/Documents/pythonScripts/new/" + path + "/"
    expanded_path = os.path.expanduser(dir)
    g0 = os.listdir(expanded_path)
    g0 = g0[index_start: index_end]
    g0_files = [expanded_path + filename for filename in g0]
    append_to_file(file_path, "jobs group " + path)
    append_to_file(file_path, str(g0))
    return g0_files


def read_files(training_files):
    training_files_csv = list()
    for file in training_files:
        df_train = pd.read_csv(file, sep=",")
        training_files_csv.append(df_train)
    return training_files_csv


def main(t, sequence_length, target, features):
    file_path = 'nb.txt'
    start_time = time.time()
    training_files = read_file_names(file_path, "0", 0, 50)
    training_files_csv = read_files(training_files)
    validation_files = read_file_names(file_path, "0", 50, 100)
    validation_files_csv = read_files(validation_files)
    test_files = read_file_names(file_path, "1", 0, 50)
    test_files_csv = read_files(test_files)

    pred_cpu_train, act_cpu_train = get_prediction_results(sequence_length, t, training_files_csv, target)
    pred_cpu_validation, act_cpu_validation = get_prediction_results(sequence_length, t, validation_files_csv, target)
    pred_cpu_test, act_cpu_test = get_prediction_results(sequence_length, t, test_files_csv, target)

    calculate_prediction_results(t, pred_cpu_train, act_cpu_train, start_time, "train.txt")
    calculate_prediction_results(t, pred_cpu_test, act_cpu_test, start_time, "test.txt")
    calculate_prediction_results(t, pred_cpu_validation, act_cpu_validation, start_time, "validation.txt")


if __name__ == "__main__":
    for t in (1, 2, 3, 4, 5, 6):
        main(t, 1, 'mean_CPU_usage', 'mean_CPU_usage')

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
        return x, self.y[i: i + self.t]  # return target n time stamps ahead


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


def append_results_to_file(act_cpu, pred_cpu, file_path, start_time, t):
    for job_index in range(len(act_cpu)):
        append_to_file(file_path, str(t) + " timestamp ahead prediction")

        current_act_cpu_train = act_cpu[job_index].values
        current_pred_cpu_train = pred_cpu[job_index].values

        calc_MSE_Accuracy(t, current_act_cpu_train, current_pred_cpu_train, file_path, start_time)

def calculate_prediction_results(t, pred_cpu_test, act_cpu_test, pred_cpu_train, act_cpu_train, file_path, start_time):
    append_results_to_file(act_cpu_test, pred_cpu_test, "test.txt", start_time, t)
    append_results_to_file(act_cpu_train, pred_cpu_train, "training.txt", start_time, t)


def plot_results(t, predictions_cpu, actual_values_cpu, sequence_length, target,
                 df):
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
        axs.plot(indices, current_actual_values_cpu, label='actual ' + target[0], linewidth=1,
                 color='orange')
        axs.plot(indices, current_predictions_cpu, label='predicted ' + target[0], linewidth=1,
                 color='blue', linestyle='dashed')
        axs.set_xlabel('Time')
        plt.xticks(rotation=45)  # 'vertical')
        plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
        axs.set_ylabel(target[0])
        axs.set_title('LSTM ' + target[0] + ' prediction h=' + str(sequence_length) + ', t=' + str(i + 1))
        axs.legend()
        plt.savefig('LSTM_bi_directional_' + 'h' + str(sequence_length) + '_t' + str(i + 1) + '' + '.png')


def get_prediction_results(sequence_length, t, test_datasets,target):
    prediction_test_cpus = list()
    actual_test_cpus = list()
    for test_dataset in test_datasets:
        # in a naive model - the prediction = the last actual value of the sequence
        start_train_index = sequence_length + t - 1
        print(test_dataset[target])
        print(test_dataset[target][start_train_index - t:-t])
        print(test_dataset[target][start_train_index:])
        prediction_test_cpu = test_dataset[target][start_train_index - t:-t]
        prediction_test_cpus.append(prediction_test_cpu)
        # actual results needs to have the same size as the prediction
        actual_test_cpu = test_dataset[target][start_train_index:]
        actual_test_cpus.append(actual_test_cpu)

    return prediction_test_cpus, actual_test_cpus


def read_file_names(file_path, path, nr_0, nr_1):
    dir = "~/Documents/pythonScripts/" + path + "/0/"
    expanded_path = os.path.expanduser(dir)
    g0 = os.listdir(expanded_path)
    g0 = g0[:nr_0]
    g0_files = [expanded_path + filename for filename in g0]
    append_to_file(file_path, "jobs group 0")
    append_to_file(file_path, str(g0))
    dir = "~/Documents/pythonScripts/" + path + "/1/"
    expanded_path = os.path.expanduser(dir)

    g1 = os.listdir(expanded_path)
    g1 = g1[:nr_1]
    g1_files = [expanded_path + filename for filename in g1]
    append_to_file(file_path, "jobs group -1")
    append_to_file(file_path, str(g1))
    return g0_files + g1_files


def read_files(training_files):
    training_files_csv = list()
    for file in training_files:
        df_train = pd.read_csv(file, sep=",")
        training_files_csv.append(df_train)
    return training_files_csv


def main(t,sequence_length, target, features):
    file_path = 'nb.txt'
    start_time = time.time()
    training_files = read_file_names(file_path, "training", 80, 20)  # 80, 20)
    training_files_csv = read_files(training_files)
    test_files = read_file_names(file_path, "test", 40, 10)  # 40, 10)
    test_files_csv = read_files(test_files)
    pred_cpu_train, act_cpu_train = get_prediction_results(sequence_length, t, training_files_csv, target)
    pred_cpu_test, act_cpu_test = get_prediction_results(sequence_length, t, test_files_csv,target)

    calculate_prediction_results(t, pred_cpu_test, act_cpu_test, pred_cpu_train,
                                 act_cpu_train, file_path, start_time)


if __name__ == "__main__":
    for t in (1, 2, 3, 6):
        main(t, 1, 'mean_CPU_usage', 'mean_CPU_usage')
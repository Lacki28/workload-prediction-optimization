import time
from datetime import timezone, timedelta

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
from matplotlib import pyplot as plt, ticker
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5, t=1):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.t = t
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        # shape length, -sequence length, because for the first I do not have any predecessors
        # - n + 1, because I might predict a few timestamps ahead - therefore I may not predict some at the beginning
        return self.X.shape[0] - self.sequence_length - self.t + 1

    # returns the input sequence and the target value
    def __getitem__(self, i):
        # start at element i and go to element i+sequence length, the result is "sequence length many" rows
        x = self.X[i:(i + self.sequence_length), :]
        # start at the last element of x (sequence length +i) and predict n timestamps ahead and subtract -1
        return x, self.y[i + self.sequence_length + self.t - 1]


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


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


def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")


def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, y_test_pred, y_test)
    append_to_file(file_path, "mae & mse & r2 & nr & training & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(r2) + " & " + str(
                       np.round(nr.numpy(), decimals=5)) + " & " + str(training_time) + " & " + str(
                       round((time.time() - start_time), 2)))

def calculate_prediction_results(t, pred_cpu_test, pred_mem_test, act_cpu_test, act_mem_test, pred_cpu_train,
                                 pred_mem_train, act_cpu_train, act_mem_train, file_path, start_time, training_time):
    append_to_file(file_path, "TRAIN ERRORS:")
    append_to_file(file_path, "CPU:")
    calc_MSE_Accuracy(t, act_cpu_train, pred_cpu_train, file_path, start_time, training_time)
    append_to_file(file_path, "MEM:")
    calc_MSE_Accuracy(t, act_mem_train, pred_mem_train, file_path, start_time, training_time)
    append_to_file(file_path, "TEST ERRORS CPU:")
    append_to_file(file_path, "CPU:")
    calc_MSE_Accuracy(t, act_cpu_test, pred_cpu_test, file_path, start_time, training_time)
    append_to_file(file_path, "MEM:")
    calc_MSE_Accuracy(t, act_mem_test, pred_mem_test, file_path, start_time, training_time)



def plot_results(t, prediction_test_cpu, prediction_test_mem, actual_test_cpu, actual_test_mem, sequence_length,
                 target, df):
    indices = df.index
    indices = indices[int(len(df) * 0.7) + t - 1:]
    indices = [str(period) for period in indices]
    start_train_index = sequence_length + t - 1
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
    axs[0].plot(indices, prediction_test_cpu, label='actual ' + target[0], linewidth=1,
                color='orange')
    axs[0].plot(indices, actual_test_cpu, label='predicted ' + target[0], linewidth=1, color='blue', linestyle='dashed')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel(target[0])
    axs[0].set_title('NB ' + target[0] + ' prediction h=' + str(sequence_length) + ', t=' + str(t))
    axs[0].legend()
    axs[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    axs[0].xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # Set x-axis tick frequency

    axs[1].plot(indices, prediction_test_mem, label='actual ' + target[1], linewidth=1,
                color='orange')
    axs[1].plot(indices, actual_test_mem, label='predicted ' + target[1], linewidth=1, color='blue', linestyle='dashed')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel(target[1])
    axs[1].set_title('NB ' + target[1] + ' prediction h=' + str(sequence_length) + ', t=' + str(t))
    axs[1].legend()
    axs[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    axs[1].xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # Set x-axis tick frequency
    plt.savefig('NB_multivariate_' + 'h' + str(sequence_length) + '_t' + str(t) + '' + '.png')


def get_prediction_results(sequence_length, t, test_dataset):
    # in a naive model - the prediction = the last actual value of the sequence
    start_train_index = sequence_length + t - 1
    prediction_test = test_dataset.y[start_train_index - t:-t]
    prediction_test_cpu = prediction_test[:, 0]
    prediction_test_mem = prediction_test[:, 1]
    # actual results needs to have the same size as the prediction
    start_train_index = sequence_length + t - 1
    actual_test = test_dataset.y[start_train_index:]
    actual_test_cpu = actual_test[:, 0]
    actual_test_mem = actual_test[:, 1]

    return prediction_test_cpu, prediction_test_mem, actual_test_cpu, actual_test_mem


def get_test_training_data(sequence_length, features, target, df_test=None, df_train=None):
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable
    train_sequence_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_sequence_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    return test_sequence_dataset, train_sequence_dataset


# sequence_length - I want to make a prediction based on how many values before
# n - how many timestamps after I want to predict - example: n=1, sequ =3: x=[1,2,3],y=[4]
def main(t=1, sequence_length=12, target="mean_CPU_usage", features='mean_CPU_usage'):
    file_path = 'NB_multivariate.txt'
    start_time = time.time()

    df = pd.read_csv("../sortedGroupedJobFiles/3418324.csv", sep=",")
    # split into training and test set - check until what index the training data is
    append_to_file(file_path, "t=" + str(t) + ", sequence length=1/12/72")
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

    test_data_sequence, training_data_sequence = get_test_training_data(sequence_length, features, target,
                                                                        df_test, df_train)
    print("Get test results")
    prediction_test_cpu, prediction_test_mem, actual_test_cpu, actual_test_mem = get_prediction_results(sequence_length,
                                                                                                        t,
                                                                                                        test_data_sequence)
    print("Get training results")
    prediction_train_cpu, prediction_train_mem, actual_train_cpu, actual_train_mem = get_prediction_results(
        sequence_length, t, training_data_sequence)
    print("calculate results")
    training_time = 0
    calculate_prediction_results(t, prediction_test_cpu, prediction_test_mem, actual_test_cpu, actual_test_mem,
                                 prediction_train_cpu, prediction_train_mem, actual_train_cpu, actual_train_mem,
                                 file_path, start_time, training_time)

    plot_results(t, prediction_test_cpu, prediction_test_mem, actual_test_cpu, actual_test_mem, sequence_length,
                 target, df)



if __name__ == "__main__":
    for t in (1, 2, 3, 6):
        main(t, 1, ['mean_CPU_usage', 'canonical_mem_usage'], ['mean_CPU_usage', 'canonical_mem_usage'])

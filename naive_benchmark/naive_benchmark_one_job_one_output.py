import time
from datetime import timezone, timedelta

import pandas as pd
import sklearn.metrics as sm
import torch
from matplotlib import pyplot as plt, ticker
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5, n=1):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.n = n
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        # shape length, -sequence length, because for the first I do not have any predecessors
        # - n + 1, because I might predict a few timestamps ahead - therefore I may not predict some at the beginning
        return self.X.shape[0] - self.sequence_length - self.n + 1

    # returns the input sequence and the target value
    def __getitem__(self, i):
        # start at element i and go to element i+sequence length, the result is "sequence length many" rows
        x = self.X[i:(i + self.sequence_length), :]
        # start at the last element of x (sequence length +i) and predict n timestamps ahead and subtract -1
        return x, self.y[i + self.sequence_length + self.n - 1]


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


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

def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")

def calc_MSE_Accuracy(y_test, y_test_pred, n, file_path):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = round(sm.mean_squared_error(y_test, y_test_pred), 5)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(n, y_test_pred, y_test)
    append_to_file(file_path, "Mean absolute error =" + str(mae))
    append_to_file(file_path, "Mean squared error =" + str(mse))
    append_to_file(file_path, "R2 score =" + str(r2))
    append_to_file(file_path, "Naive ratio =" + str(nr))


def calculate_prediction_results(n, prediction_test, actual_test_values, prediction_training, actual_train_values,file_path):
    append_to_file(file_path, "TRAIN ERRORS CPU:")
    calc_MSE_Accuracy(actual_train_values, prediction_training, n,file_path)
    append_to_file(file_path, "TEST ERRORS CPU:")
    calc_MSE_Accuracy(actual_test_values, prediction_test, n, file_path)

def plot_results(n, sequence_length, df, y_test, y_prediction, target):
    indices = df.index
    indices = indices[int(len(df) * 0.7) + n - 1:]
    indices = [str(period) for period in indices]
    start_train_index = sequence_length + n - 1
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    axs.plot(indices, y_test.y[start_train_index:], label='actual ' + target, linewidth=1, color='orange')
    axs.plot(indices, y_prediction, label='predicted ' + target, linewidth=1, color='blue', linestyle='dashed')
    axs.set_xlabel('Time')
    plt.xticks(rotation=45)  # 'vertical')
    plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
    axs.set_ylabel(target)
    axs.set_title('Naive benchmark ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(n))
    axs.legend()
    plt.savefig('NB_' + 'h' + str(sequence_length) + '_t' + str(n) + '' + '.png')
    plt.show()


def get_prediction_results(sequence_length, n, test_dataset):
    # in a naive model - the prediction = the last actual value of the sequence
    start_train_index = sequence_length + n - 1
    prediction_test_cpu = test_dataset.y[start_train_index - n:-n]

    # actual results needs to have the same size as the prediction
    start_train_index = sequence_length + n - 1
    actual_test_cpu = test_dataset.y[start_train_index:]

    return prediction_test_cpu, actual_test_cpu


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
def main(n=1, sequence_length=12, target="mean_CPU_usage", features='mean_CPU_usage'):
    file_path = 'NB.txt'
    start_time = time.time()

    df = pd.read_csv("../sortedGroupedJobFiles/3418324.csv", sep=",")
    # split into training and test set - check until what index the training data is
    append_to_file(file_path, "n=" + str(n) + ", sequence length=" + str(sequence_length))
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
    prediction_test, actual_test_values = get_prediction_results(sequence_length, n, test_data_sequence)
    print("Get training results")
    prediction_training, actual_train_values = get_prediction_results(sequence_length, n, training_data_sequence)
    print("calculate results")
    calculate_prediction_results(n, prediction_test, actual_test_values, prediction_training, actual_train_values, file_path)
    plot_results(n, sequence_length, df, test_data_sequence, prediction_test, target)
    append_to_file(file_path, "--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    for t in (1, 2, 3, 12):
        for history in (1, 12, 288):
            if t == 12 and history == 1:
                main(t, 24, 'mean_CPU_usage', 'mean_CPU_usage')
            else:
                main(t, history, 'mean_CPU_usage', 'mean_CPU_usage')

import math
import os
import time
from datetime import timezone, timedelta
from functools import partial

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, ticker
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

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
        return x, self.y[i + self.t]  # return target n time stamps ahead


class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, num_hidden_units, num_layers, t, dropout):
        super().__init__()
        self.input_size = num_sensors  # this is the number of features
        self.hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.t = t

        # self.bidirectional = True
        self.lstm_cpu = nn.LSTM(
            input_size=num_sensors,  # the number of expected features in the input x
            hidden_size=num_hidden_units,  # The number of features in the hidden state h
            batch_first=True,
            dropout=dropout,
            # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            # bidirectional=True,
            num_layers=self.num_layers  # number of layers that have some hidden units
        )
        self.linear_cpu = nn.Linear(num_hidden_units, 1)

    def forward(self, x):
        batch_size = x.shape[
            0]  # x is the tensor I get from the training, batchsize[sequence size[nr of input elements]]
        # requires_grad: tracks all operations performed on the tensor, and creates a computation graph that connects all
        # the operations from the input tensor to the output tensor. This computation graph is used to compute the gradients
        # of the output tensor with respect to the input tensor.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()  # a tensor containing the initial cell state for each element in the batch, of shape (batch, hidden_size).
        out_cpu, (hn, cn) = self.lstm_cpu(x, (h0, c0))  # pass the input sequence and initial states to the lstm
        out_cpu = out_cpu[:, -self.t, :]  # Predicting the last n hidden states
        out_lin_cpu = self.linear_cpu(out_cpu)
        return out_lin_cpu


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
    return et / et1


def my_loss_fn(output, target):
    loss = 0
    loss += mse(output, target)
    # loss += naive_ratio(output, target, size)
    return loss


def my_accuracy_fn(output, target):
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


def test_model(data_loader, model, optimizer, ix_epoch, device):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            cpu = model(X)
            total_loss += my_loss_fn(cpu, y[:, 0])
            accuracy += my_accuracy_fn(cpu, y[:, 0])
    with tune.checkpoint_dir(
            ix_epoch) as checkpoint_dir:  # context manager creates a new directory for each epoch in the tuning process and returns the path to that directory as checkpoint_dir.
        path = os.path.join(checkpoint_dir,
                            "checkpoint")  # /home/anna/ray_results/train_and_test_model_2023-03-30_10-46-09/train_and_test_model_50fa3_00001_1_batch_size=16,layers=8,lr=0.0014,units=4_2023-03-30_10-46-10/checkpoint_000004/checkpoint
        torch.save((model.state_dict(), optimizer.state_dict()),
                   path)  # The torch.save() function saves the state of the PyTorch model and optimizer as a dictionary containing the state of each object.
    tune.report(loss=(total_loss / len(data_loader)), accuracy=(
            accuracy / len(
        data_loader)))  # The tune.report() function is used to report the loss and accuracy of the model to the Ray Tune framework. This function takes a dictionary of metrics as input, where the keys are the names of the metrics and the values are the metric values.
    return (total_loss / len(data_loader))


def train_model(data_loader, model, optimizer, device):
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        cpu = model(X)
        optimizer.zero_grad()  # sets gradients back to zero: when I start the training loop: zero out the gradients so that I can perform this tracking correctly
        loss = my_loss_fn(cpu, y[:, 0])
        loss.backward()  # gradients computed
        optimizer.step()  # to proceed gradient descent


def predict(data_loader, model, device):
    output1 = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_prediction_1 = model(X)
            output1 = torch.cat((output1, y_prediction_1), 0)
    return output1


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


def denormalize_data_minMax(target, prediction_test):
    prediction_test = (prediction_test * (
            min_max_dict[target[0]]["max"] - min_max_dict[target[0]]["min"])) + min_max_dict[
                          target[0]]["min"]
    return prediction_test


def denormalize_start_time_minMax(time):
    return (time * (min_max_dict["start_time"]["max"] - min_max_dict["start_time"]["min"])) + \
        min_max_dict["start_time"]["min"]


def calculate_prediction_results(t, prediction_test, actual_test_values, prediction_training, actual_train_values,
                                 file_path, start_time, training_time):
    append_to_file(file_path, "TRAIN ERRORS CPU:")
    calc_MSE_Accuracy(t, actual_train_values, prediction_training, file_path, start_time, training_time)
    append_to_file(file_path, "TEST ERRORS CPU:")
    calc_MSE_Accuracy(t, actual_test_values, prediction_test, file_path, start_time, training_time)


def plot_results(t, predictions, actual_values, sequence_length, target, df):
    indices = pd.DatetimeIndex(df["start_time"])
    indices = indices.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = indices[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    indices = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
               [first_timestamp + i * increment for i in range(len(indices))]]
    indices = indices[int(len(df) * 0.7) + t - 1:]
    indices = [str(period) for period in indices]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    axs.plot(indices, actual_values, label='actual ' + target, linewidth=1, color='orange')
    axs.plot(indices, predictions, label='predicted ' + target, linewidth=1, color='blue', linestyle='dashed')
    axs.set_xlabel('Time')
    plt.xticks(rotation=45)  # 'vertical')
    plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
    axs.set_ylabel(target)
    axs.set_title('LSTM ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(t))
    axs.legend()
    plt.savefig('LSTM_filter_big_test_' + 'h' + str(sequence_length) + '_t' + str(t) + '' + '.png')


def get_prediction_results(t, target, test_dataset, best_trained_model, device, config):
    test_eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    prediction_test_cpu = predict(test_eval_loader, best_trained_model, device)
    prediction_test = denormalize_data_minMax(target, prediction_test_cpu)
    start_train_index = config["sequence_length"] + t - 1
    prediction_test = prediction_test[config["sequence_length"] - 1:]
    # actual results needs to have the same size as the prediction
    actual_test_cpu = test_dataset.y[:, 0][start_train_index:]
    actual_test_values = denormalize_data_minMax(target, actual_test_cpu)
    return prediction_test, actual_test_values


def get_min_max_values_of_training_data(df):
    for col in df.columns:
        if col not in min_max_dict:
            min_max_dict[col] = {"min": df[col].min(), "max": df[col].max()}
        else:
            min_max_dict[col]["min"] = min(min_max_dict[col]["min"], df[col].min())
            min_max_dict[col]["max"] = max(min_max_dict[col]["max"], df[col].max())


def get_training_data(t, target, features, df_train=None, config=None):
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable
    # first use the filter, then normalize the data
    df_train = df_train.apply(lambda x: savgol_filter(x, 51, 4))
    # z_scores = (df_train - np.mean(df_train)) / np.std(df_train)
    #
    # # Define a threshold for outlier detection
    # threshold = 3
    #
    # # Identify the outliers using the Z-score method
    # outliers = np.abs(z_scores) > threshold
    # # Remove the outliers from the data set
    # clean_data = df_train[~outliers]
    df_train = normalize_data_minMax(features, df_train)
    train_sequence = SequenceDataset(
        df_train,
        target=target,
        features=features, t=t,
        sequence_length=config["sequence_length"])

    return train_sequence


def get_test_data(t, target, features, df_test=None, config=None):
    df_test = normalize_data_minMax(features, df_test)
    test_sequence = SequenceDataset(
        df_test,
        target=target,
        features=features,
        t=t,
        sequence_length=config["sequence_length"])

    return test_sequence


def train_and_test_model(config, checkpoint_dir="checkpoint", training_data_file=None, t=None, epochs=None,
                         features=None, target=None, file_path=None):
    model = RegressionLSTM(num_sensors=len(features), num_hidden_units=config["units"], num_layers=config["layers"],
                           t=t, dropout=0)
    # Wrap the model in nn.DataParallel to support data parallel training on multiple GPUs:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    batch_size = config["batch_size"]
    cv = KFold(n_splits=5, shuffle=False)
    training_sequence = get_training_data(t, target, features, training_data_file, config)
    losses = list()
    for ix_epoch in range(epochs):  # in each epoch, train with the file that performs worse
        for train_index, validation_index in cv.split(training_sequence):
            train_subset = torch.utils.data.Subset(training_sequence, train_index)
            val_subset = torch.utils.data.Subset(training_sequence, validation_index)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            train_model(train_loader, model, optimizer=optimizer, device=device)
        loss = test_model(validation_loader, model, optimizer, ix_epoch, device=device)
        losses.append(loss)
        print(ix_epoch)
        if ix_epoch == epochs / 4:
            plt.plot(losses)
            plt.savefig('LSTM_progress.png')


def main(t=1, sequence_length=12, epochs=2000, features=['mean_CPU_usage'], target=["mean_CPU_usage"],
         num_samples=100):
    file_path = '1lstm_filter_big_test.txt'
    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length))
    start_time = time.time()

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=epochs / 2,
        reduction_factor=2)  # if it is set to 2, then half of the configurations survive each round.
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    # first choose lin layers, units, then choose layers and sequence length
    config = {
        "sequence_length": sequence_length,
        "units": tune.choice([8, 16, 32]),
        "layers": tune.choice([4]),
        "lr": tune.loguniform(0.000001, 0.009),  # takes lower and upper bound
        "batch_size": tune.choice([64]),
    }
    df = pd.read_csv("../sortedGroupedJobFiles/3418324.csv", sep=",")
    # split into training and test set - check until what index the training data is
    test_head = int(len(df) * 0.7)
    df_train = df.iloc[:test_head, :]
    get_min_max_values_of_training_data(df_train)
    df_test = df.iloc[test_head - sequence_length:, :]
    result = tune.run(
        partial(train_and_test_model, training_data_file=df_train, t=t, epochs=epochs, features=features,
                target=target, file_path=file_path),
        resources_per_trial={"cpu": 2},
        # By default, Tune automatically runs N concurrent trials, where N is the number of CPUs (cores) on your machine.
        config=config,
        num_samples=num_samples,  # how often I sample from hyperparameters
        scheduler=scheduler,
        progress_reporter=reporter)

    # retrieve the best trial from a Ray Tune experiment using the get_best_trial() method of the tune.ExperimentAnalysis object.
    # three arguments: the name of the metric to optimize, the direction of optimization ("min" for minimizing the metric or "max" for maximizing it), and the mode for selecting the best trial ("last" for selecting the last trial that achieved the best metric value, or "all" for selecting all trials that achieved the best metric value).
    best_trial = result.get_best_trial("accuracy", "max", "last")
    training_time = round((time.time() - start_time), 2)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
    append_to_file(file_path,
                   "u=" + str(best_trial.config["units"]) + ", l=" + str(best_trial.config["layers"]) + ", lr=" + str(
                       round(best_trial.config["lr"], 5)) + ", bs=" +
                   str(best_trial.config["batch_size"]))

    best_trained_model = RegressionLSTM(num_sensors=len(features), num_hidden_units=best_trial.config["units"],
                                        num_layers=best_trial.config["layers"], t=t, dropout=0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    append_to_file(file_path, str(best_checkpoint_dir))

    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    # get normalized sequence data
    test_data_files_sequence = get_test_data(t, target, features, df_test, best_trial.config)
    training_data_files_sequence = get_training_data(t, target, features, df_train, best_trial.config)
    print("Get test results")
    prediction_test, actual_test_values = get_prediction_results(t, target, test_data_files_sequence,
                                                                 best_trained_model,
                                                                 device,
                                                                 best_trial.config)
    print("Get training results")
    prediction_training, actual_train_values = get_prediction_results(t, target, training_data_files_sequence,
                                                                      best_trained_model,
                                                                      device, best_trial.config)
    print("calculate results")
    calculate_prediction_results(t, prediction_test, actual_test_values, prediction_training, actual_train_values,
                                 file_path, start_time, training_time)
    plot_results(t, prediction_test, actual_test_values, best_trial.config["sequence_length"],
                 target[0], df)


if __name__ == "__main__":
    main(t=2, sequence_length=12, epochs=2000, features=['mean_CPU_usage'],
         target=["mean_CPU_usage"],
         num_samples=12)
    # for t in (1, 2, 3, 12):
    #     for history in (1, 12, 72):
    #         if t == 12 and history == 1:
    #             main(t=t, sequence_length=24, epochs=1000, features=['mean_CPU_usage'],
    #                  target=["mean_CPU_usage"],
    #                  num_samples=12)
    #         else:
    #             main(t=t, sequence_length=history, epochs=1000, features=['mean_CPU_usage'],
    #                  target=["mean_CPU_usage"],
    #                  num_samples=12)

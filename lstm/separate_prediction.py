import hashlib
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
n = 1  # how many timestamps after I want to predict - example: n=1, sequ =3: x=[1,2,3],y=[4]
epochs = 500
features = ['start_time', 'mean_CPU_usage', 'canonical_mem_usage']
# , 'assigned_mem_usage', 'max_mem_usage', 'mean_local_disk_space_used', 'max_CPU_usage', 'nr_of_tasks', 'scheduling_class']
target = ["mean_CPU_usage", 'canonical_mem_usage']


# use a sequence of observations for the prediction
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


class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, num_hidden_units, num_layers, lin_layers):
        super().__init__()
        self.input_size = num_sensors  # this is the number of features
        self.hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.lin_layers = lin_layers
        # self.bidirectional = True
        self.lstm_cpu = nn.LSTM(
            input_size=num_sensors,  # the number of expected features in the input x
            hidden_size=num_hidden_units,  # The number of features in the hidden state h
            batch_first=True,
            # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            # bidirectional=True,
            num_layers=self.num_layers  # number of layers that have some hidden units
        )
        self.lstm_mem = nn.LSTM(
            input_size=num_sensors,
            hidden_size=num_hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.linear_cpu = nn.Linear(num_hidden_units, lin_layers)
        self.linear_mem = nn.Linear(num_hidden_units, lin_layers)
        self.output_cpu = nn.Linear(lin_layers, 1)
        self.output_mem = nn.Linear(lin_layers, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
        out_cpu = out_cpu[:, -1, :]
        out_lin_cpu = self.relu(self.linear_cpu(out_cpu))
        final_out_cpu = (self.output_cpu(out_lin_cpu))

        out_mem, (hn, cn) = self.lstm_mem(x, (h0, c0))  # pass the input sequence and initial states to the lstm
        out_mem = out_mem[:, -1, :]
        out_lin_mem = self.relu(self.linear_mem(out_mem))
        final_out_mem = (self.output_mem(out_lin_mem))
        return final_out_cpu, final_out_mem


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


def test_model(data_loader, model, optimizer, ix_epoch, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            cpu, mem = model(X)
            loss1 = my_loss_fn(cpu, y[:, 0])
            loss2 = my_loss_fn(mem, y[:, 1])
            total_loss = loss1 + loss2
            acc1 = my_accuracy_fn(cpu, y[:, 0])
            acc2 = my_accuracy_fn(mem, y[:, 1])
            accuracy = acc1 + acc2
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
        cpu, mem = model(X)
        loss1 = my_loss_fn(cpu, y[:, 0])
        loss2 = my_loss_fn(mem, y[:, 1])
        loss = loss1 + loss2
        optimizer.zero_grad()  # sets gradients back to zero: when I start the training loop: zero out the gradients so that I can perform this tracking correctly
        loss.backward()  # gradients computed
        optimizer.step()  # to proceed gradient descent


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
        mae.append(round(sm.mean_absolute_error(y_test[i][index], y_test_pred[i][index]), 5))
        mse.append(round(sm.mean_squared_error(y_test[i][index], y_test_pred[i][index]), 5))
        r2.append(round(sm.r2_score(y_test[i][index], y_test_pred[i][index]), 5))
        nr.append(naive_ratio(y_test_pred[i][index], y_test[i][index]))
    print("Mean absolute error =", (sum(mae) / len(mae)))
    print("Mean squared error =", (sum(mse) / len(mse)))
    print("R2 score =", (sum(r2) / len(r2)))
    print("Naive ratio =", (sum(nr) / len(nr)))


# normalize data
mins = {}
maxs = {}


# mins["start_time"] = 600000000
# maxs["start_time"] = 2505900000000


def normalize_data_minMax(df):
    pd.options.mode.chained_assignment = None
    # find min and max
    for c in df.columns:
        if mins.get(c) is None:
            min = np.min(df[c])
            max = np.max(df[c])
            mins[c] = min
            maxs[c] = max
        elif np.min(df[c]) < mins.get(c):
            mins[c] = np.min(df[c])
        elif maxs[c] > maxs.get(c):
            maxs[c] = maxs.get(c)
    for c in df.columns:
        min = mins[c]
        max = maxs[c]
        value_range = max - min
        df.loc[:, c] = (df.loc[:, c] - min) / value_range
    return df


def denormalize_data_minMax(prediction_test_cpu, prediction_test_mem):
    prediction_test_cpu = (prediction_test_cpu * (maxs["mean_CPU_usage"] - mins["mean_CPU_usage"])) + mins[
        "mean_CPU_usage"]
    prediction_test_mem = (prediction_test_mem * (
            maxs["canonical_mem_usage"] - mins["canonical_mem_usage"])) + mins[
                              "canonical_mem_usage"]
    return prediction_test_cpu, prediction_test_mem


def denormalize_start_time_minMax(time):
    return (time * (maxs["start_time"] - mins["start_time"])) + mins["start_time"]


#
# #Standardize data
# stds = {}
# means = {}
# def normalize_data_std(df):
#     pd.options.mode.chained_assignment = None
#     for c in df.columns:
#         if means.get(c) is None:
#             mean = np.mean(df[c])
#             std = np.std(df[c])
#             means[c] = mean
#             stds[c] = std
#         else:
#             mean = means[c]
#             std = stds[c]
#         df.loc[:, c] = (df.loc[:, c] - mean) / std
#     return df
#
#
# def denormalize_data_std(prediction_test_cpu, prediction_test_mem):
#     prediction_test_cpu = (prediction_test_cpu * stds["mean_CPU_usage"]) + means["mean_CPU_usage"]
#     prediction_test_mem = (prediction_test_mem * stds["canonical_mem_usage"]) + means[
#         "canonical_mem_usage"]
#     return prediction_test_cpu, prediction_test_mem
#
#
# def denormalize_start_std(time):
#     return (time * stds["start_time"]) + means["start_time"]


def calculate_prediction_results(prediction_test, actual_test_values, prediction_training, actual_train_values):
    print("TRAIN ERRORS CPU:")
    calc_MSE_Accuracy(actual_train_values, prediction_training, 0)
    print("TRAIN ERRORS MEM:")
    calc_MSE_Accuracy(actual_train_values, prediction_training, 1)
    print("TEST ERRORS CPU:")
    calc_MSE_Accuracy(actual_test_values, prediction_test, 0)
    print("TEST ERRORS MEM:")
    calc_MSE_Accuracy(actual_test_values, prediction_test, 1)


def plot_results(predictions, actual_values, original_test_files, config):
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index_list = []
    actual_values=actual_values[0]
    actual_test_values=[]
    for test_file in original_test_files:
        # actual results needs to have the same size as the prediction
        start_train_index = config["sequence_length"] + n - 1
        actual_values_cpu=actual_values[0]
        actual_values_mem=actual_values[1]
        actual_test_values.append((actual_values_cpu,actual_values_mem))
        test_file.X = test_file.X[start_train_index:]
        indices = ((denormalize_start_time_minMax(
            test_file.X[:, 0]) - 600000000) / in_hours)  # first index is timestamp
        index_list.append(indices)
    fig, axs = plt.subplots(nrows=len(index_list), ncols=2, figsize=(10, 5))
    for i in range(len(index_list)):
        if len(index_list) == 1:
            axs[i].plot(index_list[i], actual_test_values[i][0], label='actual CPU usage', linewidth=1,
                        markerfacecolor='blue')
            axs[i].plot(index_list[i], predictions[i][0], label='predicted CPU', linewidth=1, markerfacecolor='red')
            axs[i].set_xlabel('Time (hours)')
            axs[i].set_ylabel('CPU prediction')
            axs[i].set_title('Mean CPU prediction')
            axs[i].legend()

            axs[1].plot(index_list[i], actual_test_values[i][1], label='actual memory usage', linewidth=1,
                        markerfacecolor='blue')
            axs[1].plot(index_list[i], predictions[i][1], label='predicted memory', linewidth=1,
                        markerfacecolor='red')
            axs[1].set_xlabel('Time (hours)')
            axs[1].set_ylabel('Memory prediction')
            axs[1].set_title('Mean memory prediction')
            axs[1].legend()
        else:
            axs[i][0].plot(index_list[i], actual_test_values[i][0], label='actual CPU usage', linewidth=1,
                           markerfacecolor='blue')
            axs[i][0].plot(index_list[i], predictions[i][0], label='predicted CPU', linewidth=1, markerfacecolor='red')
            axs[i][0].set_xlabel('Time (hours)')
            axs[i][0].set_ylabel('CPU prediction')
            axs[i][0].set_title('Mean CPU prediction')
            axs[i][0].legend()

            axs[i][1].plot(index_list[i], actual_test_values[i][1], label='actual memory usage', linewidth=1,
                           markerfacecolor='blue')
            axs[i][1].plot(index_list[i], predictions[i][1], label='predicted memory', linewidth=1,
                           markerfacecolor='red')
            axs[i][1].set_xlabel('Time (hours)')
            axs[i][1].set_ylabel('Memory prediction')
            axs[i][1].set_title('Mean memory prediction')
            axs[i][1].legend()
    plt.savefig('8_training_sets.png')
    plt.show()


def get_prediction_results(test_data_files_sequence, best_trained_model, device, config):
    prediction_test = []
    for test_dataset in test_data_files_sequence:
        test_eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        prediction_test_cpu, prediction_test_mem = predict(test_eval_loader, best_trained_model, device)
        prediction_test.append(denormalize_data_minMax(prediction_test_cpu, prediction_test_mem))

    actual_test_values = []
    for sequence_file in test_data_files_sequence:
        # actual results needs to have the same size as the prediction
        start_train_index = config["sequence_length"] + n - 1
        actual_test_cpu = sequence_file.y[:, 0][start_train_index:]
        actual__test_mem = sequence_file.y[:, 1][start_train_index:]
        actual_test_values.append(denormalize_data_minMax(actual_test_cpu, actual__test_mem))
    return prediction_test, actual_test_values


def get_test_training_data(test_data_files=None, training_data_files=None, config=None):
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable
    train_sequence_dataset = []
    test_sequence_dataset = []
    for df_train in training_data_files:
        # first use the filter, then normalize the data
        df_train = df_train.apply(lambda x: savgol_filter(x, 51, 4))
        df_train = normalize_data_minMax(df_train)
        train_sequence_dataset.append(SequenceDataset(
            df_train,
            target=target,
            features=features,
            sequence_length=config["sequence_length"]
        ))
    for df_test in test_data_files:
        df_test = normalize_data_minMax(df_test)
        test_sequence_dataset.append(SequenceDataset(
            df_test,
            target=target,
            features=features,
            sequence_length=config["sequence_length"]
        ))

    return test_sequence_dataset, train_sequence_dataset


def train_and_test_model(config, checkpoint_dir="checkpoint", test_data_files=None, training_data_files=None):
    test_sequence_dataset, train_sequence_dataset = get_test_training_data(test_data_files, training_data_files, config)
    model = RegressionLSTM(num_sensors=len(features), num_hidden_units=config["units"], num_layers=config["layers"],
                           lin_layers=config["lin_layers"])
    # Wrap the model in nn.DataParallel to support data parallel training on multiple GPUs:
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    batch_size = config["batch_size"]
    train_loader = []
    test_loader = []
    for train_dataset in train_sequence_dataset:
        train_loader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1))
    for test_dataset in test_sequence_dataset:
        test_loader.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1))
    losses = {}
    for ix_epoch in range(epochs):  # in each epoch, train with the file that performs worse
        for train_df in train_loader:  # random sample
            hash_value = hash(train_df.dataset.y)
            if hash_value in losses:
                max_key = max(losses, key=lambda k: losses[k])
                if hash_value == max_key:
                    print(hash_value)
                    print(losses)
                    train_model(train_df, model, optimizer=optimizer, device=device)
                    loss = test_model(train_df, model, optimizer, ix_epoch, device=device)
                    losses[hash_value] = loss
            else:
                train_model(train_df, model, optimizer=optimizer, device=device)
                loss = test_model(train_df, model, optimizer, ix_epoch, device=device)
                losses[hash_value] = loss

    print("Finished Training")


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        read_file = pd.read_csv(file, sep=",")
        read_file.index = read_file["start_time"]
        df_list.append(read_file)
    return df_list


def main():
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=300,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["lin_layers", "units", "layers", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    # first choose lin layers, units, then choose layers and sequence length
    config = {
        "lin_layers": tune.choice([5]),
        "units": tune.choice([4, 6, 12]),
        "layers": tune.choice([4]),
        "lr": tune.loguniform(0.00025, 0.0005),  # takes lower and upper bound
        "batch_size": tune.choice([8]),
        "sequence_length": tune.choice([3])  # , 6, 12, 24]),
    }
    training_data_files = read_data("../training")
    test_data_files = read_data("../test")
    result = tune.run(
        partial(train_and_test_model, test_data_files=test_data_files, training_data_files=training_data_files),
        resources_per_trial={"cpu": 2, "gpu": 0},
        # By default, Tune automatically runs N concurrent trials, where N is the number of CPUs (cores) on your machine.
        config=config,
        num_samples=1,  # how often I sample from hyperparameters
        scheduler=scheduler,
        progress_reporter=reporter)

    # retrieve the best trial from a Ray Tune experiment using the get_best_trial() method of the tune.ExperimentAnalysis object.
    # three arguments: the name of the metric to optimize, the direction of optimization ("min" for minimizing the metric or "max" for maximizing it), and the mode for selecting the best trial ("last" for selecting the last trial that achieved the best metric value, or "all" for selecting all trials that achieved the best metric value).
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    lin_layers = best_trial.config["lin_layers"]
    best_trained_model = RegressionLSTM(num_sensors=len(features), num_hidden_units=best_trial.config["units"],
                                        num_layers=best_trial.config["layers"], lin_layers=lin_layers)
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    # get normalized sequence data
    test_data_files_sequence, training_data_files_sequence = get_test_training_data(test_data_files,
                                                                                    training_data_files,
                                                                                    best_trial.config)
    print("Get test results")
    prediction_test, actual_test_values = get_prediction_results(test_data_files_sequence, best_trained_model, device,
                                                                 best_trial.config)
    print("Get training results")
    prediction_training, actual_train_values = get_prediction_results(training_data_files_sequence, best_trained_model,
                                                                      device, best_trial.config)
    print("calculate results")
    calculate_prediction_results(prediction_test, actual_test_values, prediction_training, actual_train_values)
    plot_results(prediction_test, actual_test_values, test_data_files_sequence, best_trial.config)


if __name__ == "__main__":
    main()

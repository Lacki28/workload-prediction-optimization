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
n = 1  # how many timestamps after I want to predict - example: n=1, sequ =3: x=[1,2,3],y=[4]
epochs = 200
features = ['start_time', 'mean_CPU_usage', 'unmapped_page_cache_mem_usage', 'canonical_mem_usage']
target = ["mean_CPU_usage", 'unmapped_page_cache_mem_usage']


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
        # start at elemet i and go to element i+sequence length, the result is "sequence length many" rows
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
        self.lstm = nn.LSTM(
            input_size=num_sensors,  # the number of expected features in the input x
            hidden_size=num_hidden_units,  # The number of features in the hidden state h
            batch_first=True,
            # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            # bidirectional=True,
            num_layers=self.num_layers  # number of layers that have some hidden units
        )
        print(lin_layers)
        self.linear1 = nn.Linear(num_hidden_units, lin_layers)
        self.linear2 = nn.Linear(num_hidden_units, lin_layers)
        self.output1 = nn.Linear(lin_layers, 1)
        self.output2 = nn.Linear(lin_layers, 1)
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
        out, (hn, cn) = self.lstm(x, (h0, c0))  # pass the input sequence and initial states to the lstm
        out = out[:, -1, :]
        out11 = self.relu(self.linear1(out))
        out21 = self.relu(self.linear2(out))
        out1 = self.sigmoid(self.output1(out11))
        out2 = self.sigmoid(self.output2(out21))
        return out1, out2


def mse(prediction, real_value):
    MSE = torch.square(torch.subtract(real_value, prediction)).mean()
    return MSE


def naive_ratio(prediction, real_value):
    # Compute the absolute difference between corresponding elements of a and b
    abs_diff_et1 = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / len(prediction)) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / (et1 + 0.000000000001)


def my_loss_fn(output, target):
    loss = 0
    loss += mse(output, target)
    # loss += naive_ratio(output, target, size)
    return loss


def my_accuracy_fn(output, target):
    accuracy = round(sm.r2_score(target, output))
    return accuracy


def test_model(data_loader, model, optimizer, ix_epoch, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            output1, output2 = model(X)
            loss1 = my_loss_fn(output1, y[:, 0])
            loss2 = my_loss_fn(output2, y[:, 1])
            total_loss = loss1 + loss2
            acc1 = my_accuracy_fn(output1, y[:, 0])
            acc2 = my_accuracy_fn(output2, y[:, 1])
            accuracy = acc1 + acc2
    print(f"Total test loss: {total_loss}")
    print(f"Total test accuracy: {accuracy}")
    with tune.checkpoint_dir(
            ix_epoch) as checkpoint_dir:  # context manager creates a new directory for each epoch in the tuning process and returns the path to that directory as checkpoint_dir.
        path = os.path.join(checkpoint_dir,
                            "checkpoint")  # /home/anna/ray_results/train_and_test_model_2023-03-30_10-46-09/train_and_test_model_50fa3_00001_1_batch_size=16,layers=8,lr=0.0014,units=4_2023-03-30_10-46-10/checkpoint_000004/checkpoint
        torch.save((model.state_dict(), optimizer.state_dict()),
                   path)  # The torch.save() function saves the state of the PyTorch model and optimizer as a dictionary containing the state of each object.
    tune.report(loss=(total_loss), accuracy=(
        accuracy))  # The tune.report() function is used to report the loss and accuracy of the model to the Ray Tune framework. This function takes a dictionary of metrics as input, where the keys are the names of the metrics and the values are the metric values.


def train_model(data_loader, model, optimizer, device):
    total_loss = 0
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()  # sets gradients back to zero: when I start the training loop: zero out the gradients so that I can perform this tracking correctly
        output1, output2 = model(X)
        loss1 = my_loss_fn(output1, y[:, 0])
        loss2 = my_loss_fn(output2, y[:, 1])
        loss = loss1 + loss2
        loss.backward()  # gradients computed
        optimizer.step()  # to proceed gradient descent

        total_loss += loss.item()
    print(f"Train loss: {total_loss}")


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


def calc_MSE_Accuracy(y_test, y_test_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 5))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 5))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 5))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 5))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 5))
    nr = naive_ratio(y_test_pred.clone().detach(), torch.tensor(y_test.values))
    print("Naive ratio =", nr)


mins = {}
maxs = {}


def normalize_data_minMax(df_train, df, df_test):
    pd.options.mode.chained_assignment = None
    for c in df_train.columns:
        min = np.min(df[c])
        max = np.max(df[c])
        mins[c] = min
        maxs[c] = max
        value_range = max - min
        df_train.loc[:, c] = (df_train.loc[:, c] - min) / value_range
        df_test.loc[:, c] = (df_test.loc[:, c] - min) / value_range
    return df_train, df_test


def denormalize_data_minMax(df_out, prediction_test_cpu, prediction_test_mem, prediction_train_cpu,
                            prediction_train_mem):
    for c in df_out.columns:
        max = maxs[c]
        min = mins[c]
        df_out[c] = (df_out[c] * (max - min)) + min
    prediction_test_cpu = (prediction_test_cpu * (maxs["mean_CPU_usage"] - mins["mean_CPU_usage"])) + mins[
        "mean_CPU_usage"]
    prediction_test_mem = (prediction_test_mem * (
            maxs["unmapped_page_cache_mem_usage"] - mins["unmapped_page_cache_mem_usage"])) + mins[
                              "unmapped_page_cache_mem_usage"]
    prediction_train_cpu = (prediction_train_cpu * (maxs["mean_CPU_usage"] - mins["mean_CPU_usage"])) + mins[
        "mean_CPU_usage"]
    prediction_train_mem = (prediction_train_mem * (
            maxs["unmapped_page_cache_mem_usage"] - mins["unmapped_page_cache_mem_usage"])) + mins[
                               "unmapped_page_cache_mem_usage"]
    return df_out, prediction_test_cpu, prediction_test_mem, prediction_train_cpu, prediction_train_mem


stds = {}
means = {}


def normalize_data_std(df_train, df, df_test):
    for c in df_train.columns:
        mean = np.mean(df[c])
        std = np.std(df[c])
        means[c] = mean
        stds[c] = std
        df_train.loc[:, c] = (df_train.loc[:, c] - mean) / std
        df_test.loc[:, c] = (df_test.loc[:, c] - mean) / std
    return df_train, df_test


def denormalize_data_std(df_out, prediction_test_cpu, prediction_test_mem, prediction_train_cpu,
                         prediction_train_mem):
    for c in df_out.columns:
        mean = means[c]
        std = stds[c]
        df_out[c] = (df_out[c] * std) + mean
    prediction_test_cpu = (prediction_test_cpu * stds["mean_CPU_usage"]) + means["mean_CPU_usage"]
    prediction_test_mem = (prediction_test_mem * stds["unmapped_page_cache_mem_usage"]) + means[
        "unmapped_page_cache_mem_usage"]
    prediction_train_cpu = (prediction_train_cpu * stds["mean_CPU_usage"]) + means["mean_CPU_usage"]
    prediction_train_mem = (prediction_train_mem * stds["unmapped_page_cache_mem_usage"]) + means[
        "unmapped_page_cache_mem_usage"]
    return df_out, prediction_test_cpu, prediction_test_mem, prediction_train_cpu, prediction_train_mem


def calculate_prediction_results(df_train, df_out, prediction_train_cpu, prediction_train_mem, prediction_test_cpu,
                                 prediction_test_mem):
    print("TRAIN ERRORS CPU:")
    start_train_index = sequence_length + n - 1
    end_train_index = len(df_train)
    calc_MSE_Accuracy(df_out["mean_CPU_usage"].iloc[start_train_index:end_train_index], prediction_train_cpu)
    print("TRAIN ERRORS MEM:")
    calc_MSE_Accuracy(df_out["unmapped_page_cache_mem_usage"].iloc[start_train_index:end_train_index],
                      prediction_train_mem)
    print("TEST ERRORS CPU:")
    start_test_index = end_train_index - 1 + sequence_length + n
    end_test_index = len(df_out)
    calc_MSE_Accuracy(
        df_out["mean_CPU_usage"].iloc[start_test_index:end_test_index],
        prediction_test_cpu)
    print("TEST ERRORS MEM:")
    calc_MSE_Accuracy(
        df_out["unmapped_page_cache_mem_usage"].iloc[start_test_index:end_test_index],
        prediction_test_mem)


def plot_results(index_in_hours, df_train, df_test, prediction_test_cpu, prediction_train_cpu, prediction_test_mem,
                 prediction_train_mem, df_out):
    index_test = index_in_hours.iloc[len(df_train) - 1 + sequence_length + n:len(df_train) + len(df_test)]
    index_train = index_in_hours.iloc[sequence_length + n - 1:len(df_train)]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].plot(index_in_hours, df_out["mean_CPU_usage"].values, label='actual CPU usage', linewidth=1,
                markerfacecolor='blue')
    axs[0].plot(index_test, prediction_test_cpu, label='predicted CPU', linewidth=1, markerfacecolor='red')
    axs[0].plot(index_train, prediction_train_cpu, label='predicted CPU', linewidth=1, markerfacecolor='red')
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('CPU prediction')
    axs[0].set_title('Mean CPU prediction')
    axs[0].legend()

    axs[1].plot(index_in_hours, df_out["unmapped_page_cache_mem_usage"].values, label='actual memory usage',
                linewidth=1,
                markerfacecolor='blue')
    axs[1].plot(index_test, prediction_test_mem, label='predicted disk IO time', linewidth=1, markerfacecolor='red')
    axs[1].plot(index_train, prediction_train_mem, label='predicted disk IO time', linewidth=1, markerfacecolor='red')
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Mean disk IO time')
    axs[1].set_title('Disk IO time prediction')
    axs[1].legend()

    plt.show()


def get_test_training_data(df):
    # split into training and test set - check until what index the training data is
    test_head = df.index[int(0.8 * len(df))]
    df_train = df.loc[:test_head - 1, :]
    df_test = df.loc[test_head:len(df), :]
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable
    df_train = df_train.apply(lambda x: savgol_filter(x, 8, 4))
    df_train, df_test = normalize_data_minMax(df_train, df, df_test)
    # Create datasets that PyTorch DataLoader can work with
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    return df_test, df_train, test_dataset, train_dataset


def train_and_test_model(config, checkpoint_dir="checkpoint", df=None):
    _, _, test_dataset, train_dataset = get_test_training_data(df)
    model = RegressionLSTM(num_sensors=len(features), num_hidden_units=config["units"], num_layers=config["layers"],  lin_layers=config["lin_layers"])
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Untrained test\n--------")
    test_model(test_loader, model, optimizer, 0, device)
    for ix_epoch in range(epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, optimizer=optimizer, device=device)
        test_model(test_loader, model, optimizer, ix_epoch, device=device)
        print()

    print("Finished Training")


def main():
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=50,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["lin_layers", "units", "layers", "lr","batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    config = {
        "lin_layers":tune.choice([100, 125, 150, 175]),
        "units": tune.choice([4, 8, 10]),
        "layers": tune.choice([2, 3, 4, 5]),
        "lr": tune.loguniform(0.0006, 0.0004),  # takes lower and upper bound
        "batch_size": tune.choice([16])
    }
    df = pd.read_csv("job_smaller.csv", sep=",")

    result = tune.run(
        partial(train_and_test_model, df=df),
        resources_per_trial={"cpu": 6, "gpu": 0},
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
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    df_test, df_train, test_dataset, train_dataset = get_test_training_data(df)
    train_eval_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    prediction_train = predict(train_eval_loader, best_trained_model, device)
    test_eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    prediction_test = predict(test_eval_loader, best_trained_model, device)
    df_out = pd.concat([df_train, df_test])
    prediction_test_cpu, prediction_test_mem = prediction_test
    prediction_train_cpu, prediction_train_mem = prediction_train
    df_out, prediction_test_cpu, prediction_test_mem, prediction_train_cpu, prediction_train_mem = denormalize_data_minMax(
        df_out, prediction_test_cpu, prediction_test_mem, prediction_train_cpu, prediction_train_mem)
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index_in_hours = ((df_out['start_time'] - 600000000) / in_hours)

    calculate_prediction_results(df_train, df_out, prediction_train_cpu, prediction_train_mem, prediction_test_cpu,
                                 prediction_test_mem)
    plot_results(index_in_hours, df_train, df_test, prediction_test_cpu, prediction_train_cpu, prediction_test_mem,
                 prediction_train_mem, df_out)


if __name__ == "__main__":
    main()

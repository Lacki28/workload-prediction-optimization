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
        return x, self.y[i+1: i + self.t+1]  # return target n time stamps ahead

class TimeSeriesTransformer(nn.Module):

    def __init__(self, input_dim, output_dim, d_model, nhead, dim_feedforward, num_layers, sequence_length):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )

        self.decoder = nn.Linear(d_model * sequence_length, output_dim)

    def forward(self, input):
        batch_size, seq_len, input_dim = input.size()
        input = input.transpose(0, 1)  # Shape: (seq_len, batch_size, input_dim)
        input = self.embedding(input)  # Shape: (seq_len, batch_size, d_model)

        output = self.transformer_encoder(input)  # Shape: (seq_len, batch_size, d_model)
        output = output.view(batch_size, -1)  # Shape: (batch_size, seq_len * d_model)
        output = self.decoder(output)  # Shape: (batch_size, output_dim)
        return output

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


def my_r2_fn(output, target):
    output_has_nan = torch.isnan(output).any().item()
    if output_has_nan:
        return - math.inf
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


def test_model(data_loader, model, optimizer, ix_epoch, device, t):
    model.eval()
    loss_cpu = 0
    r2 = 0
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            cpu = model(X)
            desired_shape = (len(cpu), t)
            actual_cpu = y[..., 0]
            actual_cpu = actual_cpu.view(desired_shape)

            loss_cpu += my_loss_fn(cpu, actual_cpu)
            r2_cpu = my_r2_fn(cpu, actual_cpu)
            r2 = r2_cpu
    with tune.checkpoint_dir(
            ix_epoch) as checkpoint_dir:  # context manager creates a new directory for each epoch in the tuning process and returns the path to that directory as checkpoint_dir.
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()),
                   path)  # The torch.save() function saves the state of the PyTorch model and optimizer as a dictionary containing the state of each object.
    loss_cpu = (loss_cpu / len(data_loader)).item()
    r2 = (r2 / len(data_loader))
    tune.report(r2=r2,
                loss=loss_cpu)  # The tune.report() function is used to report the loss and r2 of the model to the Ray Tune framework. This function takes a dictionary of metrics as input, where the keys are the names of the metrics and the values are the metric values.


def train_model(data_loader, model, optimizer, device, t):
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()  # sets gradients back to zero: when I start the training loop: zero out the gradients so that I can perform this tracking correctly
        cpu = model(X)
        desired_shape = (len(cpu), t)  # should be same as batch size, but in case data%batch size isn't 0, we need this
        actual_cpu = y[..., 0]
        actual_cpu = actual_cpu.view(desired_shape)
        loss = my_loss_fn(cpu, actual_cpu)
        loss.backward()  # gradients computed
        optimizer.step()  # to proceed gradient descent


def predict(data_loader, model, device):
    cpu = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_prediction_cpu = model(X)
            cpu = torch.cat((cpu, y_prediction_cpu), 0)
    return cpu


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
            min_max_dict[target]["max"] - min_max_dict[target]["min"])) + min_max_dict[
                          target]["min"]
    return prediction_test


def calculate_prediction_results(t, pred_cpu_test, act_cpu_test, pred_cpu_train, act_cpu_train, file_path, start_time,
                                 training_time):
    for i in range(t):
        append_to_file(file_path, str(i + 1) + " timestamp ahead prediction")
        current_act_cpu_train = act_cpu_train[:, i]
        current_pred_cpu_train = pred_cpu_train[:, i]
        current_act_cpu_test = act_cpu_test[:, i]
        current_pred_cpu_test = pred_cpu_test[:, i]
        append_to_file(file_path, "TRAIN ERRORS CPU:")
        calc_MSE_Accuracy(t, current_act_cpu_train, current_pred_cpu_train, file_path, start_time, training_time)
        append_to_file(file_path, "TEST ERRORS CPU:")
        calc_MSE_Accuracy(t, current_act_cpu_test, current_pred_cpu_test, file_path, start_time, training_time)
        append_to_file(file_path, "")


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
        axs.set_xlabel('Time', fontsize=18)
        plt.xticks(rotation=45)  # 'vertical')
        plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
        axs.set_ylabel(target[0], fontsize=18)
        axs.set_title('Transformer ' + target[0] + ' prediction h=' + str(sequence_length) + ', t=' + str(i + 1),
                      fontsize=20)
        axs.legend(fontsize=16)
        plt.savefig('Transformer_' + 'h' + str(sequence_length) + '_t' + str(i + 1) + '' + '.png')


def get_prediction_results(t, target, test_dataset, best_trained_model, device, config):
    test_eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    prediction_test_cpu = predict(test_eval_loader, best_trained_model, device)
    denorm_cpu = denormalize_data_minMax(target[0], prediction_test_cpu)
    start_train_index = config["sequence_length"]
    pred_denorm_cpu = denorm_cpu[config["sequence_length"] - 1:]

    # actual results needs to have the same size as the prediction
    actual_test_cpu = test_dataset.y[:, 0][start_train_index:]
    actual_test_cpu = actual_test_cpu.unfold(0, t, 1)
    act_denorm_cpu = denormalize_data_minMax(target[0], actual_test_cpu)
    return pred_denorm_cpu, act_denorm_cpu


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
    # df_train = df_train.apply(lambda x: savgol_filter(x, 51, 4))
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
                         features=None, target=None):
    model = TimeSeriesTransformer(len(features), t, config["d_model"], config["nhead"],
                                  config["dim_feedforward"], config["num_layers"], config["sequence_length"])

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
    for ix_epoch in range(epochs):  # in each epoch, train with the file that performs worse
        for train_index, validation_index in cv.split(training_sequence):
            train_subset = torch.utils.data.Subset(training_sequence, train_index)
            val_subset = torch.utils.data.Subset(training_sequence, validation_index)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            train_model(train_loader, model, optimizer=optimizer, device=device, t=t)
        test_model(validation_loader, model, optimizer, ix_epoch, device=device, t=t)


def main(t=1, sequence_length=12, epochs=2000, features=['mean_CPU_usage'], target=["mean_CPU_usage"],
         num_samples=100):
    seed = 28
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    file_path = 'transformer_univariate_tests.txt'
    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length) + ", epochs=" + str(epochs))
    start_time = time.time()
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=epochs,
        grace_period=epochs / 4,
        reduction_factor=2)  # if it is set to 2, then half of the configurations survive each round.
    reporter = CLIReporter(
        metric_columns=["loss", "r2", "training_iteration"])
    config = {
        "sequence_length": sequence_length,
        "d_model": tune.grid_search([16, 32]),
        "nhead": tune.grid_search([1]),
        "dim_feedforward": tune.grid_search([16, 32]),
        "num_layers": tune.grid_search([1]),
        "lr": tune.loguniform(0.00001, 0.0009),  # takes lower and upper bound
        "batch_size": tune.grid_search([4, 8, 16]),
    }
    df = pd.read_csv("../../sortedGroupedJobFiles/3418324.csv", sep=",")
    # split into training and test set - check until what index the training data is
    test_head = int(len(df) * 0.7)
    df_train = df.iloc[:test_head, :]
    get_min_max_values_of_training_data(df_train)
    df_test = df.iloc[test_head - sequence_length:, :]
    result = tune.run(
        partial(train_and_test_model, training_data_file=df_train, t=t, epochs=epochs, features=features,
                target=target),
        resources_per_trial={"cpu": 4},
        # By default, Tune automatically runs N concurrent trials, where N is the number of CPUs (cores) on your machine.
        config=config,
        num_samples=num_samples,  # how often I sample from hyperparameters
        scheduler=scheduler,
        progress_reporter=reporter
    )
    for trial in result.trials:
        print(trial.metric_analysis)
    # retrieve the best trial from a Ray Tune experiment using the get_best_trial() method of the tune.ExperimentAnalysis object.
    # three arguments: the name of the metric to optimize, the direction of optimization ("min" for minimizing the metric or "max" for maximizing it), and the mode for selecting the best trial ("last" for selecting the last trial that achieved the best metric value, or "all" for selecting all trials that achieved the best metric value).
    best_trial = result.get_best_trial("loss", "min", "last")
    training_time = round((time.time() - start_time), 2)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation r2: {}".format(best_trial.last_result["r2"]))
    append_to_file(file_path,
                   "dm=" + str(best_trial.config["d_model"]) + ", nh=" + str(
                       best_trial.config["nhead"]) + ", lr=" + str(
                       round(best_trial.config["lr"], 5)) + ", bs=" +
                   str(best_trial.config["batch_size"]) + ", df=" +
                   str(best_trial.config["dim_feedforward"]) + ", nl=" +
                   str(best_trial.config["num_layers"]))

    best_trained_model = TimeSeriesTransformer(len(features), t, best_trial.config["d_model"],
                                               best_trial.config["nhead"], best_trial.config["dim_feedforward"],
                                               best_trial.config["num_layers"], sequence_length)

    device = "cpu"
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
    pred_cpu_test, act_cpu_test = get_prediction_results(t, target, test_data_files_sequence,
                                                         best_trained_model,
                                                         device,
                                                         best_trial.config)
    print("Get training results")
    pred_cpu_train, act_cpu_train = get_prediction_results(t, target, training_data_files_sequence, best_trained_model,
                                                           device, best_trial.config)
    print("calculate results")
    calculate_prediction_results(t, pred_cpu_test, act_cpu_test, pred_cpu_train,
                                 act_cpu_train, file_path, start_time, training_time)
    plot_results(t, pred_cpu_test, act_cpu_test, best_trial.config["sequence_length"],
                 target, df)


if __name__ == "__main__":
    for history in (1, 6, 12):
        main(t=6, sequence_length=history, epochs=150, features=['mean_CPU_usage'],
             target=['mean_CPU_usage'], num_samples=2)

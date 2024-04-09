import math
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import torch
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from scipy.signal import savgol_filter
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
    return loss


def my_r2_fn(output, target):
    output_has_nan = torch.isnan(output.cpu()).any().item()
    if output_has_nan:
        return - math.inf
    r2 = sm.r2_score(target.cpu(), output.cpu())
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
            r2 = my_r2_fn(cpu, actual_cpu)
    with tune.checkpoint_dir(
            ix_epoch) as checkpoint_dir:  # context manager creates a new directory for each epoch in the tuning process and returns the path to that directory as checkpoint_dir.
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()),
                   path)  # The torch.save() function saves the state of the PyTorch model and optimizer as a dictionary containing the state of each object.
    loss = (loss_cpu / len(data_loader)).item()
    r2 = (r2 / len(data_loader))
    return r2, loss


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
    cpu = torch.empty(0, device=device)  # Initialize an empty tensor on the desired device
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
    # append_to_file(file_path, "mae & mse & r2 & nr & training & total")
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


def calculate_prediction_results(t, prediction, actual_values, start_time, training_time, path):
    for job_index in range(len(prediction)):
        for i in range(t):
            current_act_cpu_validation = actual_values[job_index][:, i]
            current_pred_cpu_validation = prediction[job_index][:, i]
            calc_MSE_Accuracy(t, current_act_cpu_validation.cpu(), current_pred_cpu_validation.cpu(),
                              path + str(i + 1) + ".txt", start_time, training_time)


def get_prediction_results(t, target, test_data, best_trained_model, device, config):
    pred_denorm_cpus = list()
    act_denorm_cpus = list()
    for data in test_data:
        print("H")
        test_eval_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)
        prediction_test_cpu = predict(test_eval_loader, best_trained_model, device)
        denorm_cpu = denormalize_data_minMax(target[0], prediction_test_cpu)
        start_train_index = config["sequence_length"]
        pred_denorm_cpu = denorm_cpu[config["sequence_length"] - 1:]

        # actual results needs to have the same size as the prediction
        actual_test_cpu = data.y[:, 0][start_train_index:]
        actual_test_cpu = actual_test_cpu.unfold(0, t, 1)
        act_denorm_cpu = denormalize_data_minMax(target[0], actual_test_cpu)
        pred_denorm_cpus.append(pred_denorm_cpu)
        act_denorm_cpus.append(act_denorm_cpu)
    return pred_denorm_cpus, act_denorm_cpus


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


def train_and_test_model(config, checkpoint_dir="checkpoint", training_files=None, validation_files=None, t=None,
                         epochs=None,
                         features=None, target=None, device=None):
    training_files = read_files(training_files, True)
    validation_files = read_files(validation_files, False)
    training_loaders = list()
    for df_train in training_files:
        training_sequence = get_training_data(t, target, features, df_train, config)
        train_loader = DataLoader(training_sequence, batch_size=config["batch_size"], shuffle=False)
        training_loaders.append(train_loader)
    validation_loaders = list()
    for df_validation in validation_files:
        validation_sequence = get_test_data(t, target, features, df_validation, config)
        train_loader = DataLoader(validation_sequence, batch_size=config["batch_size"], shuffle=False)
        validation_loaders.append(train_loader)

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

    for ix_epoch in range(epochs):  # in each epoch, train with the file that performs worse
        r2s = list()
        losses = list()
        for training_loader in training_loaders:
            train_model(training_loader, model, optimizer=optimizer, device=device, t=t)
        for validation_loader in validation_loaders:
            r2, loss = test_model(validation_loader, model, optimizer, ix_epoch, device=device, t=t)
            r2s.append(r2)
            losses.append(loss)
        tune.report(r2=sum(r2s),
                    loss=sum(
                        losses))  # The tune.report() function is used to report the loss and r2 of the model to the Ray Tune framework. This function takes a dictionary of metrics as input, where the keys are the names of the metrics and the values are the metric values.


def read_file_names(file_path, path, index_start, index_end):
    dir = "~/Documents/pythonScripts/new/" + path + "/"
    expanded_path = os.path.expanduser(dir)
    g0 = os.listdir(expanded_path)
    g0 = g0[index_start: index_end]
    g0_files = [expanded_path + filename for filename in g0]
    append_to_file(file_path, "jobs group " + path)
    append_to_file(file_path, str(g0))
    return g0_files


def read_files(training_files, training):
    training_files_csv = list()
    for file in training_files:
        df_train = pd.read_csv(file, sep=",")
        if (training):
            get_min_max_values_of_training_data(df_train)
        training_files_csv.append(df_train)
    return training_files_csv


def main(t=1, sequence_length=12, epochs=2000, features=['mean_CPU_usage'], target=["mean_CPU_usage"],
         num_samples=100):
    seed = 28
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    file_path = 'transformer.txt'
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
    # Best trial config: {'sequence_length': 1, 'd_model': 32, 'nhead': 1, 'dim_feedforward': 64, 'num_layers': 1, 'lr': 2.0728938773827123e-05, 'batch_size': 16}
    # Best trial config: {'sequence_length': 1, 'd_model': 16, 'nhead': 1, 'dim_feedforward': 64, 'num_layers': 1, 'lr': 1.8196927794602183e-05, 'batch_size': 32}
    # Best trial config: {'sequence_length': 1, 'd_model': 16, 'nhead': 1, 'dim_feedforward': 64, 'num_layers': 1, 'lr': 1.3967205462958928e-05, 'batch_size': 32}

    config = {
        "sequence_length": sequence_length,
        "d_model": tune.grid_search([32]),
        "nhead": tune.grid_search([1]),
        "dim_feedforward": tune.grid_search([16]),
        "num_layers": tune.grid_search([1]),
        "lr": tune.loguniform(0.000001, 0.000005),  # takes lower and upper bound
        "batch_size": tune.grid_search([64, 128]),
    }
    training_files = read_file_names(file_path, "0", 0, 50)
    training_files_csv = read_files(training_files, True)
    validation_files = read_file_names(file_path, "0", 50, 100)
    validation_files_csv = read_files(validation_files, False)
    test_files = read_file_names(file_path, "1", 0, 50)
    test_files_csv = read_files(test_files, False)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    result = tune.run(
        partial(train_and_test_model, training_files=training_files, validation_files=validation_files, t=t,
                epochs=epochs, features=features,
                target=target, device=device),
        resources_per_trial={"cpu": 4, "gpu": 0.25},
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

    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    append_to_file(file_path, str(best_checkpoint_dir))

    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    training_data = list()
    test_data = list()
    validation_data = list()
    for df_train in training_files_csv:
        training_sequence = get_training_data(t, target, features, df_train, best_trial.config)
        training_data.append(training_sequence)
    for df_test in test_files_csv:
        training_sequence = get_test_data(t, target, features, df_test, best_trial.config)
        test_data.append(training_sequence)
    for df_validation in validation_files_csv:
        validation_sequence = get_test_data(t, target, features, df_validation, config)
        validation_data.append(validation_sequence)

    print("Get test results")
    pred_cpu_test, act_cpu_test = get_prediction_results(t, target, test_data,
                                                         best_trained_model,
                                                         device,
                                                         best_trial.config)
    print("Get training results")
    pred_cpu_train, act_cpu_train = get_prediction_results(t, target, training_data, best_trained_model,
                                                           device, best_trial.config)

    print("Get validation results")
    pred_cpu_validation, act_cpu_validation = get_prediction_results(t, target, validation_data, best_trained_model,
                                                                     device, best_trial.config)

    print("calculate results")
    calculate_prediction_results(t, pred_cpu_train, act_cpu_train, start_time, training_time, "new_data_train")
    calculate_prediction_results(t, pred_cpu_test, act_cpu_test, start_time, training_time, "new_data_test")
    calculate_prediction_results(t, pred_cpu_validation, act_cpu_validation, start_time, training_time,
                                 "new_data_validation")


if __name__ == "__main__":
    main(t=6, sequence_length=1, epochs=300, features=['mean_CPU_usage'],
         target=['mean_CPU_usage'], num_samples=4)

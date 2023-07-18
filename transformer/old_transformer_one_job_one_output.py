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

from transformer_model import TimeSeriesTransformer

min_max_dict = {}


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
            enc = self.X[i_start:(i + 1), :]
            dec = self.y[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            enc = self.X[0:(i + 1), :]
            enc = torch.cat((padding, enc), 0)
            dec = self.y[0:(i + 1), :]
            dec = torch.cat((padding, dec), 0)

        # encoder, decoder, target
        return enc, dec, self.y[i: i + self.t]


def mse(prediction, real_value):
    loss_function = torch.nn.MSELoss()
    MSE = loss_function(prediction, real_value)
    # mse=torch.square(torch.subtract(real_value, prediction)).mean() has the same result
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


def push_to_tensor(decoder_input, output):
    # remove first element of the decoder input
    decoder_input = decoder_input[:, 1:]
    # get the first element of the output - of the prediction of one timestamp ahead
    first_elements = output[:, 0]
    first_elements = first_elements.unsqueeze(1)
    # replace the last element with the last element of the output
    return torch.cat((decoder_input, first_elements.unsqueeze(1)), dim=1)


def test_model(data_loader, model, optimizer, ix_epoch, device, t):
    model.train(mode=False)
    loss_cpu = 0
    r2 = 0
    with torch.no_grad():  # do not calculate the gradient
        for x_enc, x_dec, target in data_loader:
            x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
            out = model.forward(x_enc.float(), x_dec.float(), training=False)
            target = torch.squeeze(target)
            loss_cpu += my_loss_fn(out, target)
            r2_cpu = my_r2_fn(out, target)
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
    model.train(mode=True)
    for x_enc, x_dec, target in data_loader:
        x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
        optimizer.zero_grad()  # sets gradients back to zero: when I start the training loop: zero out the gradients so that I can perform this tracking correctly
        out = model.forward(x_enc.float(), x_dec.float(), training=True)
        target = torch.squeeze(target)
        loss = my_loss_fn(out.double(), target.double())
        loss.backward()
        optimizer.step()


def predict(data_loader, model, device):
    cpu = torch.tensor([])
    model.eval()
    first_it = 0
    decoder_input = torch.ones([1, 24, 1])
    out = 0
    with torch.no_grad():  # do not calculate the gradient
        for x_enc, x_dec, target in data_loader:
            x_enc, x_dec, target = x_enc.to(device), x_dec.to(device), target.to(device)
            if first_it == 0:
                decoder_input = x_dec
                first_it = first_it + 1
            else:
                decoder_input = push_to_tensor(decoder_input, out)
            # Forecast
            out = model.forward(x_enc.float(), decoder_input.float(), training=False)
            cpu = torch.cat((cpu, out), 0)
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
        axs.set_xlabel('Time')
        plt.xticks(rotation=45)  # 'vertical')
        plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
        axs.set_ylabel(target[0])
        axs.set_title('Transformer ' + target[0] + ' prediction h=' + str(sequence_length) + ', t=' + str(i + 1))
        axs.legend()
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
                         features=None, target=None, file_path=None):
    model = TimeSeriesTransformer(len(features), config["dim_attn"], config["input_feat_enc"], config["input_feat_dec"],
                                  config["sequence_length"], config["n_decoder_layers"], config["n_encoder_layers"],
                                  config["n_heads"], t, device="cpu")
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
    for ix_epoch in range(epochs):
        for train_index, validation_index in cv.split(training_sequence):
            train_subset = torch.utils.data.Subset(training_sequence, train_index)
            val_subset = torch.utils.data.Subset(training_sequence, validation_index)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            train_model(train_loader, model, optimizer=optimizer, device=device, t=t)
        test_model(validation_loader, model, optimizer, ix_epoch, device=device, t=t)

    # training_sequence = get_training_data(t, target, features, training_data_file, config)
    # for ix_epoch in range(epochs):  # in each epoch, train with the file that performs worse
    #
    #     train_loader = DataLoader(training_sequence, batch_size=batch_size, shuffle=False)
    #
    #     train_model(train_loader, model, optimizer=optimizer, device=device, t=t)
    #     test_model(train_loader, model, optimizer, ix_epoch, device=device, t=t)


def main(t=1, sequence_length=12, epochs=2000, features=['mean_CPU_usage'], target=["mean_CPU_usage"],
         num_samples=100):
    file_path = 'new_transformer_univariate_all_at_once.txt'
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
    # grid_search
    config = {
        "dim_val": tune.grid_search([12]),  # 16
        "dim_attn": tune.grid_search([16]),  # 128, 256
        "input_feat_enc": tune.grid_search([1]),
        "input_feat_dec": tune.grid_search([1]),
        "sequence_length": sequence_length,
        "n_decoder_layers": tune.grid_search([1]),
        "n_encoder_layers": tune.grid_search([1]),
        "n_heads": tune.grid_search([16]),  # 16
        "lr": tune.loguniform(0.001, 0.01),  # takes lower and upper bound
        "batch_size": tune.grid_search([32, 64, 128]),
    }

    df = pd.read_csv("../../sortedGroupedJobFiles/3418324.csv", sep=",")
    # split into training and test set - check until what index the training data is
    test_head = int(len(df) * 0.7)
    df_train = df.iloc[:test_head, :]
    get_min_max_values_of_training_data(df_train)
    df_test = df.iloc[test_head - sequence_length:, :]
    result = tune.run(
        partial(train_and_test_model, training_data_file=df_train, t=t, epochs=epochs, features=features,
                target=target, file_path=file_path),
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
                   "dv=" + str(best_trial.config["dim_val"]) + ", da=" + str(best_trial.config["dim_attn"]) +
                   ", ife=" + str(best_trial.config["input_feat_enc"]) + ", ifd=" + str(
                       best_trial.config["input_feat_dec"]) + ", nel=" + str(
                       best_trial.config["n_encoder_layers"]) + ", ndl=" + str(
                       best_trial.config["n_decoder_layers"]) + ", nh=" + str(
                       best_trial.config["n_heads"]) + ", lr=" + str(round(best_trial.config["lr"], 5)) + ", bs=" +
                   str(best_trial.config["batch_size"]))

    best_trained_model = TimeSeriesTransformer(len(features), best_trial.config["dim_attn"],
                                               best_trial.config["input_feat_enc"], best_trial.config["input_feat_dec"],
                                               best_trial.config["sequence_length"],
                                               best_trial.config["n_decoder_layers"],
                                               best_trial.config["n_encoder_layers"], best_trial.config["n_heads"], t,
                                               device="cpu")
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
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
    # for history in (1, 6, 12):
    main(t=6, sequence_length=6, epochs=100, features=['mean_CPU_usage'],
         target=['mean_CPU_usage'], num_samples=1)

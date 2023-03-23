import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# hyperparameters
sequence_length = 3  # I want to make a prediction based on how many values before
n = 1  # how many timestamps after I want to predict - example: n=1, sequ =3: x=[1,2,3],y=[4]
batch_size = 1  # batch_size = nr of sequences that are processed in parallel during training - gradients of the loss function are calculated based on the average loss over the entire batch.
learning_rate = 0.005
num_hidden_units = 5  # The size of the LSTM layers, or the number of hidden units in each layer, will affect the capacity of the model. A larger number of hidden units will allow the model to capture more complex patterns in the data, but may also make the model more prone to overfitting.
num_layers = 5
epochs = 10


# source: https://www.crosstab.io/articles/time-series-pytorch-lstm/
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length + 1

    # returns the input sequence and the target value
    def __getitem__(self, i):
        x = self.X[i:(i + self.sequence_length - 1), :]
        return x, self.y[i + n + (self.sequence_length - 2)]


class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        # self.bidirectional = True
        self.lstm = nn.LSTM(
            input_size=num_sensors,  # the number of expected features in the input x
            hidden_size=hidden_units,  # The number of features in the hidden state h
            batch_first=True,
            # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            # bidirectional=True,
            num_layers=self.num_layers  # number of layers that have some hidden units
        )
        self.linear1 = nn.Linear(hidden_units, 32)
        self.linear2 = nn.Linear(hidden_units, 32)
        self.output1 = nn.Linear(32, 1)
        self.output2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[
            0]  # x is the tensor I get from the training, batchsize[sequence size[nr of input elements]]
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


def mse(prediction, real_value, size):
    mse = nn.MSELoss()
    MSE = mse(prediction, real_value)
    return MSE


def naive_ratio(prediction, real_value, size):
    # Compute the absolute difference between corresponding elements of a and b
    abs_diff_et1 = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / size) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / size) * sum_abs_diff
    return et / (et1 + 0.000000000001)


def my_loss_fn(output, target, size):
    loss = 0
    loss += mse(output, target, size)
    #loss += naive_ratio(output, target, size)
    return loss


def test_model(data_loader, model):
    num_batches = len(data_loader)
    model.eval()
    with torch.no_grad():  # do not calculate the gradient
        for i, (X, y) in enumerate(data_loader):

            if i <= ((num_batches - n) // batch_size):
                output1, output2 = model(X)
                loss1 = my_loss_fn(output1, y[:, 0], num_batches)
                loss2 = my_loss_fn(output2, y[:, 1], num_batches)
                loss = loss1 + loss2
            else:
                break
    print(f"Total test loss: {(loss) / num_batches}")


def train_model(data_loader, model, optimizer):
    num_batches = len(data_loader)  # trainingdata/batches
    total_loss = 0
    model.train()
    for i, (X, y) in enumerate(data_loader):
        if i <= (num_batches - n) // batch_size:
            optimizer.zero_grad()  # sets gradients back to zero
            output1, output2 = model(X)
            loss1 = my_loss_fn(output1, y[:, 0], num_batches)
            loss2 = my_loss_fn(output2, y[:, 1], num_batches)
            loss = loss1 + loss2
            loss.backward()  # gradients computed
            optimizer.step()  # to proceed gradient descent

            total_loss += loss.item()
        else:
            break
    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def predict(data_loader, model):
    output1 = torch.tensor([])
    output2 = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_prediction_multiple = model(X)
            y_prediction_1 = y_prediction_multiple[0]
            y_prediction_2 = y_prediction_multiple[1]
            output1 = torch.cat((output1, y_prediction_1), 0)
            output2 = torch.cat((output2, y_prediction_2), 0)
    return output1, output2


def calc_MSE_Accuracy(y_test, y_test_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
    nr = naive_ratio(y_test_pred.clone().detach(), torch.tensor(y_test.values), len(y_test.values))
    print("Naive ratio =", nr)


def main():
    df = pd.read_csv("job_smaller.csv", sep=",")
    # split into training and test set - check until what index the training data is
    test_head = df.index[int(0.8 * len(df))]
    df_train = df.loc[:test_head - 1, :]
    df_test = df.loc[test_head:len(df), :]
    features = ['start_time', 'mean_CPU_usage', 'mean_disk_IO_time', 'unmapped_page_cache_mem_usage']
    target = ["mean_CPU_usage", 'mean_disk_IO_time']
    mins = {}
    maxs = {}
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable
    for c in df_train.columns:
        min = np.min(df[c])
        max = np.max(df[c])
        mins[c] = min
        maxs[c] = max
        value_range = max - min
        df_train.loc[:, c] = (df_train.loc[:, c] - min) / value_range
        df_test.loc[:, c] = (df_test.loc[:, c] - min) / value_range
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
    # this will have the size batch_size x sequence_length x number_of_features
    # For training, we’ll shuffle the data (the rows within each data sequence are not shuffled, only the order in which we draw the blocks).
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(test_loader, model)
    for ix_epoch in range(epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, optimizer=optimizer)
        test_model(test_loader, model)
        print()

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    prediction_train = predict(train_eval_loader, model)
    prediction_test = predict(test_loader, model)

    df_out = pd.concat([df_train, df_test])
    prediction_test_cpu, prediction_test_mem = prediction_test
    print(prediction_test_cpu)
    print(prediction_test_mem)
    prediction_train_cpu, prediction_train_mem = prediction_train
    for c in df_out.columns:
        max = maxs[c]
        min = mins[c]
        df_out[c] = (df_out[c] * (max - min)) + min
    prediction_test_cpu = (prediction_test_cpu * (maxs["mean_CPU_usage"] - mins["mean_CPU_usage"])) + mins[
        "mean_CPU_usage"]
    prediction_test_mem = (prediction_test_mem * (maxs["mean_disk_IO_time"] - mins["mean_disk_IO_time"])) + mins[
        "mean_disk_IO_time"]
    prediction_train_cpu = (prediction_train_cpu * (maxs["mean_CPU_usage"] - mins["mean_CPU_usage"])) + mins[
        "mean_CPU_usage"]
    prediction_train_mem = (prediction_train_mem * (maxs["mean_disk_IO_time"] - mins["mean_disk_IO_time"])) + mins[
        "mean_disk_IO_time"]
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index_in_hours = ((df_out['start_time'] - 600000000) / in_hours)
    print("TRAIN ERRORS CPU:")
    calc_MSE_Accuracy(df_out["mean_CPU_usage"].iloc[sequence_length - 1:len(df_train)], prediction_train_cpu)
    print("TRAIN ERRORS MEM:")
    calc_MSE_Accuracy(df_out["mean_disk_IO_time"].iloc[sequence_length - 1:len(df_train)], prediction_train_mem)
    print("TEST ERRORS CPU:")
    calc_MSE_Accuracy(df_out["mean_CPU_usage"].iloc[len(df_train) + sequence_length - 1:len(df_train) + len(df_test)],
                      prediction_test_cpu)
    print("TEST ERRORS MEM:")
    calc_MSE_Accuracy(
        df_out["mean_disk_IO_time"].iloc[len(df_train) + sequence_length - 1:len(df_train) + len(df_test)],
        prediction_test_mem)

    index_test = index_in_hours.iloc[len(df_train) + sequence_length - 1:len(df_train) + len(df_test)]
    index_train = index_in_hours.iloc[sequence_length - 1:len(df_train)]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].plot(index_in_hours, df_out["mean_CPU_usage"].values, label='actual CPU usage', linewidth=1,
               markerfacecolor='blue')
    axs[0].plot(index_test, prediction_test_cpu, label='predicted CPU', linewidth=1, markerfacecolor='red')
    axs[0].plot(index_train, prediction_train_cpu, label='predicted CPU', linewidth=1, markerfacecolor='red')
    axs[0].set_xlabel('Time (hours)')
    axs[0].set_ylabel('CPU prediction')
    axs[0].set_title('Mean CPU prediction')
    axs[0].legend()

    axs[1].plot(index_in_hours, df_out["mean_disk_IO_time"].values, label='actual memory usage', linewidth=1,
               markerfacecolor='blue')
    axs[1].plot(index_test, prediction_test_mem, label='predicted disk IO time', linewidth=1, markerfacecolor='red')
    axs[1].plot(index_train, prediction_train_mem, label='predicted disk IO time', linewidth=1, markerfacecolor='red')
    axs[1].set_xlabel('Time (hours)')
    axs[1].set_ylabel('Mean disk IO time')
    axs[1].set_title('Disk IO time prediction')
    axs[1].legend()

    plt.show()


if __name__ == "__main__":
    main()

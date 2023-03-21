import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# hyperparameters
# for each training instance, we’re going to give the model a sequence of observations.
# dataset = [1,2,3,4], in sequence set 2 I would have training X:[1,2],[3,4] Y:[2],[4]
sequence_length = 2
batch_size = 5  # batch_size - number of samples we want to pass into the training loop at each iteration
learning_rate = 5e-4
num_hidden_units = 10  # The size of your LSTM layers, or the number of hidden units in each layer, will affect the capacity of your model. A larger number of hidden units will allow the model to capture more complex patterns in the data, but may also make the model more prone to overfitting.
num_layers = 10  # each layer consists of some hidden units
epochs = 5


# source: https://www.crosstab.io/articles/time-series-pytorch-lstm/
# This custom Dataset specifies what happens when somebody requests the i’th element of the dataset. In a tabular dataset, this would be the i’th row of the table, but here we need to retrieve a sequence of rows
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]


class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        #self.bidirectional = True
        self.lstm = nn.LSTM(
            input_size=num_sensors,  # the number of expected features in the input x
            hidden_size=hidden_units,  # The number of features in the hidden state h
            batch_first=True,
            # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
            # bidirectional=True,
            num_layers=self.num_layers
            # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=2)

    def forward(self, x):
        batch_size = x.shape[
            0]  # x is the tensor I get from the training, batchsize[sequence size[nr of input elements]]
        # a tensor containing the initial hidden state for each element in the batch, of shape (batch, hidden_size).
        # tensor with batchsize[sequence size[nr of hidden units]]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        # a tensor containing the initial cell state for each element in the batch, of shape (batch, hidden_size).
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


def mse(prediction, real_value, size):
    mse = nn.MSELoss()
    MSE = mse(prediction, real_value)
    return MSE


def naive_ratio(prediction, real_value, size):
    prediction_et1 = prediction[1:]  # - shift to the right by one index - remove first element
    real_value_et1 = real_value[:-1]  # remove last element
    # Compute the absolute difference between corresponding elements of a and b
    abs_diff_et1 = torch.abs(prediction_et1 - real_value_et1)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = torch.sum(abs_diff_et1)
    et1 = (1 / size) * sum_abs_diff_et1
    abs_diff = torch.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = torch.sum(abs_diff)
    et = (1 / size) * sum_abs_diff
    return et / et1


def my_loss_fn(output, target1, target2, size):
    loss = 0
    loss += mse(output[:, 0], target1, size)
    loss += mse(output[:, 1], target2, size)
    loss += naive_ratio(output[:, 0], target1, size)
    loss += naive_ratio(output[:, 1], target2, size)
    return loss


def train_model(data_loader, model, optimizer):
    num_batches = len(data_loader)  # trainingdata/batches
    total_loss = 0
    model.train()

    for X, y in data_loader:
        # X has the shape of an array the size number of batches that has sequences inside of sequence length
        # Y has the batch size, but only one value inside - sequence length =1
        optimizer.zero_grad()  # sets gradients back to zero
        output = model(X)
        loss = my_loss_fn(output, y[:, 0], y[:, 1], num_batches)
        loss.backward()  # gradients computed
        optimizer.step()  # to proceed gradient descent

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model):
    num_batches = len(data_loader)
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            loss = my_loss_fn(output, y[:, 0], y[:, 1], num_batches)
    print(f"Total test loss: {(loss) / num_batches}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    # Use the last n inputs from the training data as initial input sequence
    # input_seq = x_train[:, -n:, :]
    # Generate predictions for the next m timestamps
    # for i in range(m):
    # output = model(input_seq)
        # input_seq = torch.cat([input_seq[:, 1:, :], output.unsqueeze(1)], dim=1)
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output


def calc_MSE_Accuracy(y_test, y_test_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

    nr = naive_ratio(torch.tensor(y_test_pred.values), torch.tensor(y_test.values), len(y_test.values))
    print("Naive ratio =", nr)


def main():
    df = pd.read_csv("job_3418339.csv", sep=",")
    # split into training and test set - check until what index the training data is
    test_head = df.index[int(0.8 * len(df))]
    df_train = df.loc[:test_head - 1, :]
    df_test = df.loc[test_head:len(df), :]
    features = ['start_time', 'mean_cpu_usage', 'mean_disk_io_time', 'canonical_mem_usage']
    target = ["mean_cpu_usage", 'canonical_mem_usage']
    means = {}
    stdevs = {}
    # normalize data: this improves model accuracy as it gives equal weights/importance to each variable so that no single
    # variable steers model performance in one direction just because they are bigger numbers
    for c in df_train.columns:
        mean_train = df_train[c].mean()
        stdev_train = df_train[c].std()
        means[c] = mean_train
        stdevs[c] = stdev_train
        df_train.loc[:, c] = (df_train.loc[:, c] - mean_train) / stdev_train
        df_test.loc[:, c] = (df_test.loc[:, c] - mean_train) / stdev_train

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
    prediction_train = predict(train_eval_loader, model).numpy()
    prediction_test = predict(test_loader, model).numpy()

    df_train.insert(2, "forecast_cpu", prediction_train[:, 0], True)
    df_train.insert(2, "forecast_mem", prediction_train[:, 1], True)
    df_test.insert(2, "forecast_cpu", prediction_test[:, 0], True)
    df_test.insert(2, "forecast_mem", prediction_test[:, 1], True)

    df_out = pd.concat([df_train, df_test])
    for c in df_out.columns:
        if c == 'forecast_cpu':
            target_stdev = stdevs["mean_cpu_usage"]
            target_mean = means["mean_cpu_usage"]
        elif c == 'forecast_mem':
            target_stdev = stdevs["canonical_mem_usage"]
            target_mean = means["canonical_mem_usage"]
        else:
            target_stdev = stdevs[c]
            target_mean = means[c]
        df_out[c] = df_out[c] * target_stdev + target_mean

    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 12  # each interval has 5 minutes
    in_days = in_hours * 24
    index_in_hours = ((df['start_time'] - 600000000) / in_hours)
    calc_MSE_Accuracy(df_out["mean_cpu_usage"], df_out["forecast_cpu"])
    calc_MSE_Accuracy(df_out["canonical_mem_usage"], df_out["forecast_mem"])
    plt.plot(index_in_hours, df_out["mean_cpu_usage"].values, label='actual CPU usage', linewidth=0.5)
    plt.plot(index_in_hours, df_out["forecast_cpu"].values, label='prediction CPU', linewidth=0.5)
    plt.plot(index_in_hours, df_out["canonical_mem_usage"].values, label='actual memory usage', linewidth=0.5)
    plt.plot(index_in_hours, df_out["forecast_mem"].values, label='prediction memory', linewidth=0.5)

    # Calculate mean of values

    plt.legend()
    # plt.savefig('CPU_mean_days_all.png')
    plt.show()


if __name__ == "__main__":
    main()

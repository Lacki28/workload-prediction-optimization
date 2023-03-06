import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# source: https://www.crosstab.io/articles/time-series-pytorch-lstm/
#This custom Dataset specifies what happens when somebody requests the i’th element of the dataset. In a tabular dataset, this would be the i’th row of the table, but here we need to retrieve a sequence of rows
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


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 8

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
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
    length=len(y_test.values)
    nr_sum_t1 =0
    for i in range(length-1):
        nr_sum_t1 += y_test_pred[i+1]-y_test[i]
    nr_sum_t1= nr_sum_t1/(length-1)
    nr_sum_t =0
    for i in range(length-2):
        nr_sum_t += y_test_pred[i+1]-y_test[i]
    nr_sum_t = nr_sum_t/(length-2)
    print(nr_sum_t1)
    print(nr_sum_t)
    print("Naive ratio")
    print(nr_sum_t/nr_sum_t1)

def main():
    df = pd.read_csv("job_3418339.csv", sep=",")
    data = df

    # split into training and test set - check until what index the training data is
    test_head = data.index[int(0.8 * len(data))]
    df_train = df.loc[:test_head - 1, :]
    df_test = df.loc[test_head:, :]

    features = ['start_time', 'mean_disk_io_time', 'canonical_mem_usage']

    target = "mean_cpu_usage"

    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()
    #The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. For machine learning, every dataset does not require normalization. It is required only when features have different ranges.
    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    batch_size = 2
    sequence_length = 4

    #Create datasets that PyTorch DataLoader can work with
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    learning_rate = 5e-4
    num_hidden_units = 8

    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(1):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        test_model(test_loader, model, loss_function)
        print()

    print(train_dataset)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    print(predict(train_eval_loader, model).numpy())
    ystar_col = "forecast"
    df_train.insert(2, "forecast", predict(train_eval_loader, model).numpy(), True)
    df_test.insert(2, "forecast", predict(test_loader, model).numpy(), True)
    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]
    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean
    print(df_out["forecast"])
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 12  # each interval has 5 minutes
    in_days = in_hours * 24
    index_in_hours = ((df['start_time'] - 600000000) / in_hours)
    calc_MSE_Accuracy(df_out["mean_cpu_usage"], df_out["forecast"])
    plt.plot(index_in_hours, df_out["mean_cpu_usage"].values, label='actual CPU usage', linewidth=0.5)
    plt.plot(index_in_hours, df_out["forecast"].values, label='prediction', linewidth=0.5)

    # Calculate mean of values

    plt.legend()
    plt.savefig('CPU_mean_days_all.png')
    plt.show()


if __name__ == "__main__":
    main()

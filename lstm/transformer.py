# import numpy as np
# import pandas as pd
# dataFrame = pd.read_csv("job_3418339.csv", sep="\t")
# dataFrame.fillna(0)
# dataFrame['mean_disk_io_time'] = dataFrame['mean_disk_io_time'].replace(np.nan, 0)
# print(dataFrame.columns)
# print(dataFrame.values)
# dataFrame.sort_values(["start_time"], axis=0, ascending=True, inplace=True, na_position='first')
# dataFrame.fillna(0)
# print(dataFrame.index)
# dataFrame.to_csv('out.csv', index =False)

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#source: https://www.crosstab.io/articles/time-series-pytorch-lstm/
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


class Transformer(nn.Module):
    def __init__(
            self,
            num_tokens,
            dim_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p,
    ):
        super().__init__()

        # Layers
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )

    def forward(self):
        pass


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
    print("output")
    print(output)
    return output


def main():
    df = pd.read_csv("job_3418339.csv", sep=",")
    time_shift = 2
    data = df.iloc[:-time_shift]
    # until what index is the training data
    test_head = data.index[int(0.8 * len(data))]
    df_train = df.loc[:test_head - 1, :]
    df_test = df.loc[test_head:, :]

    features = ['start_time', 'mean_disk_io_time', 'canonical_mem_usage']
    target = "mean_cpu_usage"
    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()

    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()

        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev

    #target = 'mean_cpu_usage'

    batch_size = 4
    sequence_length = 4

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

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = 5e-4
    num_hidden_units = 8

    model = Transformer()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(50):
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

    plt.plot(index_in_hours.head(100), df_out["mean_cpu_usage"].head(100).values, label='actual CPU usage', linewidth=0.5)
    plt.plot(index_in_hours.head(100), df_out["forecast"].head(100).values, label='prediction', linewidth=0.5)

    # Calculate mean of values

    plt.legend()
    plt.savefig('CPU_mean_days.png')
    plt.show()


if __name__ == "__main__":
    main()

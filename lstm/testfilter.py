import os

import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def plot_results(data):
    df_train8 = data.apply(lambda x: savgol_filter(x, 51, 4))
    df_train9 = data.apply(lambda x: savgol_filter(x, 51, 4))

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
    axs[0].plot(data["start_time"],data["mean_CPU_usage"])
    axs[1].plot(data["start_time"],df_train8["mean_CPU_usage"])
    axs[2].plot(data["start_time"],data["canonical_mem_usage"])
    axs[3].plot(data["start_time"],df_train9["canonical_mem_usage"])


    plt.show()


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        df_list.append(pd.read_csv(file, sep=","))
    return df_list


def main():
    training_data_files = read_data("../training")
    for data in training_data_files:
        plot_results(data)


if __name__ == "__main__":
    main()

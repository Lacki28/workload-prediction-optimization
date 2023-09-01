import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


stds = {}
means = {}
def normalize_data_std(df):
    pd.options.mode.chained_assignment = None
    for c in df.columns:
        if means.get(c) is None:
            mean = np.mean(df[c])
            std = np.std(df[c])
            means[c] = mean
            stds[c] = std
        else:
            mean = means[c]
            std = stds[c]
        df.loc[:, c] = (df.loc[:, c] - mean) / std
    return df

mins = {}
maxs = {}



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

def plot_scaling_methods(data):
    df_trainstd= data.copy()
    df_trainnorm=data.copy()
    df_trainstd = normalize_data_std(df_trainstd)
    df_trainnorm = normalize_data_minMax(df_trainnorm)

    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index = (data["start_time"] - 600000000) / in_days

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))
    axs[0].plot(index,data["mean_CPU_usage"])
    axs[0].set_title('Raw data', fontsize=20)
    axs[0].set_xlabel('Time (days)', fontsize=18)
    axs[0].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    axs[1].plot(index,df_trainstd["mean_CPU_usage"])
    axs[1].set_title('Standardisation', fontsize=20)
    axs[1].set_xlabel('Time (days)', fontsize=18)
    axs[1].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    axs[2].plot(index,df_trainnorm["mean_CPU_usage"])
    axs[2].set_title('Normalisation', fontsize=20)
    axs[2].set_xlabel('Time (days)', fontsize=18)
    axs[2].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    # plt.title("Changes before normalization")
    plt.savefig('norm_vs_std.png')

    plt.show()


def plot_results_after(data):
    df_trainnorm=data.copy()
    df_trainnorm = normalize_data_minMax(df_trainnorm)

    df_trainsavg=df_trainnorm.copy().apply(lambda x: savgol_filter(x, 51, 4))
    z_scores = (df_trainnorm.copy() - np.mean(df_trainnorm.copy())) / np.std(df_trainnorm.copy())
    threshold = 3

    noise = np.abs(z_scores) > threshold
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index = (data["start_time"] - 600000000) / in_days
    clean_data = df_trainnorm.copy()[~noise]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))
    axs[0].plot(index,df_trainnorm["mean_CPU_usage"])
    axs[0].set_title('Normalised data', fontsize=20)
    axs[0].set_xlabel('Time (days)', fontsize=18)
    axs[0].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    axs[1].plot(index,df_trainsavg["mean_CPU_usage"])
    axs[1].set_title('Savitzky-Golay filter', fontsize=20)
    axs[1].set_xlabel('Time (days)', fontsize=18)
    axs[1].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    axs[2].plot(index,clean_data["mean_CPU_usage"])
    axs[2].set_title('Noise removed',fontsize=20)
    axs[2].set_xlabel('Time (days)', fontsize=18)
    axs[2].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    fig.suptitle('Operations after normalisation', fontsize=20)
    plt.savefig('operations_after.png')
    plt.show()
#
# def plot_results_before(data):
#     df_trainnorm=data.copy()
#     df_trainnorm = normalize_data_minMax(df_trainnorm)
#
#     df_trainsavg=data.copy().apply(lambda x: savgol_filter(x, 51, 4))
#     df_trainsavg = normalize_data_minMax(df_trainsavg)
#     z_scores = (data.copy() - np.mean(data.copy())) / np.std(data.copy())
#     threshold = 3
#     # Identify the noise using the Z-score method
#     noise = np.abs(z_scores) > threshold
#     in_seconds = 1000000
#     in_minutes = in_seconds * 60
#     in_hours = in_minutes * 60
#     in_days = in_hours * 24
#     index = (data["start_time"] - 600000000) / in_days
#     # Remove the noise from the data set
#     clean_data = data.copy()[~noise]
#     clean_data = normalize_data_minMax(clean_data)
#
#     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))
#     axs[0].plot(index,df_trainnorm["mean_CPU_usage"])
#     axs[0].set_title('Normalized data', fontsize=20)
#     axs[0].set_xlabel('Time (days)', fontsize=18)
#     axs[0].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
#     axs[1].plot(index,df_trainsavg["mean_CPU_usage"])
#     axs[1].set_title('Savitzky-Golay filter', fontsize=20)
#     axs[1].set_xlabel('Time (days)', fontsize=18)
#     axs[1].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
#     axs[2].plot(index,clean_data["mean_CPU_usage"])
#     axs[2].set_title('Noise removed', fontsize=20)
#     axs[2].set_xlabel('Time (days)', fontsize=18)
#     axs[2].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
#     fig.suptitle('Operations before normalization', fontsize=20)
#     plt.savefig('operations_before.png')
#     plt.show()


def plot_results(data):
    df_trainnorm=data.copy()
    df_trainnorm = normalize_data_minMax(df_trainnorm)

    df_trainsavg=data.copy().apply(lambda x: savgol_filter(x, 51, 4))
    df_trainsavg = normalize_data_minMax(df_trainsavg)
    z_scores = (data.copy() - np.mean(data.copy())) / np.std(data.copy())
    threshold = 3
    # Identify the noise using the Z-score method
    noise = np.abs(z_scores) > threshold
    in_seconds = 1000000
    in_minutes = in_seconds * 60
    in_hours = in_minutes * 60
    in_days = in_hours * 24
    index = (data["start_time"] - 600000000) / in_days
    # Remove the noise from the data set
    clean_data = data.copy()[~noise]
    clean_data = normalize_data_minMax(clean_data)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))
    axs[0].plot(index,df_trainnorm["mean_CPU_usage"])
    axs[0].set_title('Normalised data', fontsize=20)
    axs[0].set_xlabel('Time (days)', fontsize=18)
    axs[0].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    axs[1].plot(index,df_trainsavg["mean_CPU_usage"])
    axs[1].set_title('Savitzky-Golay filter', fontsize=20)
    axs[1].set_xlabel('Time (days)', fontsize=18)
    axs[1].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    axs[2].plot(index,clean_data["mean_CPU_usage"])
    axs[2].set_title('Noise removed', fontsize=20)
    axs[2].set_xlabel('Time (days)', fontsize=18)
    axs[2].set_ylabel('CPU usage (of job 3418324)', fontsize=18)
    plt.savefig('operations.png')
    plt.show()

def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        df_list.append(pd.read_csv(file, sep=","))
    return df_list


def main():
    data = pd.read_csv("../../sortedGroupedJobFiles/3418324.csv", ",")
    plot_scaling_methods(data)
    plot_results_after(data)
    plot_results(data)



if __name__ == "__main__":
    main()

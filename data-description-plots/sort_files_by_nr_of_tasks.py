import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        csv = pd.read_csv(file, sep=",")
        if csv["nr_of_tasks"][1] == 2:
            df_list.append(csv)
    return df_list


def main():
    training_data_files = read_data("../training")
    for data in training_data_files:
        print(data[data["nr_of_tasks"]!=2])


if __name__ == "__main__":
    main()

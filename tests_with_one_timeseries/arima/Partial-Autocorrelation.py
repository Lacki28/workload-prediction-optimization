import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

n = 1


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        csv = pd.read_csv(file, sep=",", header=0)
        index = [pd.Timestamp(int(t / 1000000) + 1304233200, tz="US/Pacific", unit="s")
                 for t in csv["start_time"].values]
        csv.index = index
        df2 = csv.loc[:, csv.columns == "mean_CPU_usage"]
        df_list.append(df2)
    return df_list


# On this plot, there is a significant correlation at lag 1 followed by correlations that are not significant. This pattern indicates an autoregressive term of order 1.
# lag value =n
def main():
    test_data_files = read_data("../test")
    df = test_data_files[0]
    print(df)
    acorr = sm.tsa.acf(df)
    print(str(acorr))
    print(acorr)
    plot_pacf(df)
    plt.show()
    plot_acf(df)
    plt.show()


if __name__ == "__main__":
    main()

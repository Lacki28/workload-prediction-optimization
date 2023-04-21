import os

import pandas as pd
from statsmodels.tsa.stattools import adfuller


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


# This is used to choose parameter d, if p value is less than 0.05, d is zero
def main():
    test_data_files = read_data("../training")
    data_non_sta=0
    for df in test_data_files:
        # perform augmented Dickey-Fuller test
        result = adfuller(df)
        if result[1]>= 0.05:
            data_non_sta=data_non_sta+1
            print(df)
            print('ADF Statistic: %f' % result[0])
            # if less than 5%: data is stationary - easier to model and make predictions from, one can assume that the statistical properties of the data will remain the same in the future, and we can use this information to make accurate forecasts.
            print('p-value: %f' % result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))


if __name__ == "__main__":
    main()

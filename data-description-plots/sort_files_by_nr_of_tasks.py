import os

import pandas as pd


def read_data(path=''):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        csv = pd.read_csv(file, sep=",")
        if csv["nr_of_tasks"][1] == 2:
            df_list.append(csv)
    return df_list

def add_files_into_dir(path,filepaths):
    for file in filepaths:
        df= pd.read_csv(file, sep=",")
        df.to_csv(path+file, index=False)

def main():
    training_data_files = read_data("/sortedGroupedJobFiles")
    test_head = int(0.8 * len(training_data_files))
    df_train = training_data_files[:test_head]
    df_test = training_data_files[test_head:len(training_data_files)]
    add_files_into_dir("../training/",df_train)
    add_files_into_dir("../test/", df_test)


if __name__ == "__main__":
    main()

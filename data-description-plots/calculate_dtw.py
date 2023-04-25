import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from dtaidistance.connectors.sktime import dtw_distance
from matplotlib import pyplot as plt


def read_data(path=''):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    files = list()
    for file in filepaths:
        csv = pd.read_csv(file, sep=",", index_col=False)
        if csv["scheduling_class"][1] == 3:
            df_list.append(normalize_data_minMax(csv))  # [24:])
            files.append(file[24:])  # [24:])
    return df_list, files


def normalize_data_minMax(df):
    pd.options.mode.chained_assignment = None
    mins = {}
    maxs = {}
    # find min and max
    for c in df.columns:
        min = np.min(df[c])
        max = np.max(df[c])
        mins[c] = min
        maxs[c] = max
    for c in df.columns:
        min = mins[c]
        max = maxs[c]
        value_range = max - min
        df.loc[:, c] = (df.loc[:, c] - min) / value_range
    return df


def compute_dtw(dtw_scores,df_list, column_name, i, j):
    a = df_list[i][["start_time", column_name[0], column_name[1]]].to_numpy()
    b = df_list[j][["start_time", column_name[0], column_name[1]]].to_numpy()
    w = dtw_distance(a, b)
    dtw_scores[(i, j)] = w
    print("Computation i="+str(i)+", j="+str(j))


def main():
    start_time = time.time()
    df_list, files = read_data("./sortedGroupedJobFiles")
    # choose the column to compare
    column_names = ['mean_CPU_usage', 'canonical_mem_usage']
    # calculate DTW distance between the columns of each dataframe
    manager = multiprocessing.Manager()
    dtw_scores = manager.dict()
    num_processes = multiprocessing.cpu_count()  # use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
    for i in range(len(df_list)):
        for j in range(i + 1, len(df_list)):
            pool.apply_async(compute_dtw, args=(dtw_scores,df_list, column_names, i, j))
    pool.close()
    pool.join()
    print(dtw_scores)
    # sort the dictionary by similarity score
    sorted_dtw_scores = sorted(dtw_scores.items(), key=lambda x: x[1])
    with open('taskallNorm.txt', 'w') as f:
        f.write(str(sorted_dtw_scores) + "\n" + str(files))
    matrix = np.ones((len(files), len(files)))
    for index in range(len(sorted_dtw_scores)):
        (i, j), value = sorted_dtw_scores[index]
        matrix[i, j] = value
        matrix[j, i] = value
    np.savetxt('taskallmatrixNorm.txt', matrix, fmt='%f')
    print("--- %s seconds ---" % (time.time() - start_time))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the heatmap
    sns.heatmap(matrix, annot=True, cmap="coolwarm", ax=ax, xticklabels=files, yticklabels=files)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    # plt.show()
    plt.savefig('heatmapallNorm.png')
    print("--- %s seconds ---" % (time.time() - start_time))
    for (i, j), dtw in sorted_dtw_scores:
        print(f"Dataframes {files[i]} and {files[j]} have a DTW distance of {dtw} in column {column_names}")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()

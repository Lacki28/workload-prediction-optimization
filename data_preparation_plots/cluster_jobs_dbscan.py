import ast
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

import shutil

def plot_number_of_jobs_per_class(class_files):
    colors = ['green', (0.7, 0, 0)]
    num_values = [len(values) for values in class_files.values()]
    plt.bar(range(len(num_values)), num_values,color=colors, alpha=0.75)
    plt.xticks(range(len(num_values)), class_files.keys())
    plt.title('Number of jobs per cluster', fontsize=16)
    plt.xlabel('Clusters', fontsize=14)
    plt.ylabel('Number of Jobs', fontsize=14)
    plt.savefig('clusters.png')


def copy_files(destination, group,files):
    src = "/home/anna/PycharmProjects/workload-prediction-optimization/sortedGroupedJobFiles/"
    dst = "/home/anna/PycharmProjects/workload-prediction-optimization/clustered_jobs/"+destination+"/"
    for file in files:
        shutil.copyfile(src+file, dst+file)

def main():
    file = open(
        "taskallNorm.txt", "r")
    contents = file.read()
    lines = contents.split("\n")
    data = ast.literal_eval(lines[0])
    files = ast.literal_eval(lines[1])
    dtw_scores = {key: value for key, value in data}
    sorted_dtw_scores = [(k, v) for k, v in dtw_scores.items()]
    matrix = np.zeros((len(files), len(files)))
    for index in range(len(sorted_dtw_scores)):
        (i, j), value = sorted_dtw_scores[index]
        matrix[i, j] = value
        matrix[j, i] = value

    print(matrix)
    print(np.min(matrix))
    print(np.average(matrix))
    print(np.max(matrix))
    print(f"number of non-zero: {np.count_nonzero(matrix)}")
    print(f"number of zeros: {matrix.size - np.count_nonzero(matrix)}")

    dbscan = DBSCAN(metric='precomputed', eps=0.00005, min_samples=50)
    labels = dbscan.fit_predict(matrix)
    print(f"Clusters: {np.unique(labels)}")
    class_files = defaultdict(list)
    for i in range(len(labels)):
        class_files[labels[i]].append(files[i])
    group_0 = class_files[0]
    size = int(len(group_0) * 0.8)
    g0_train, g0_test = group_0[0:size], group_0[size:len(group_0)]
    # copy_files("0", "0", g0_train)
    # print(len((g0_train)))
    # copy_files("test", "0", g0_test)
    # print(len((g0_test)))
    group_1 = class_files[-1]
    size = int(len(group_1) * 0.8)
    g1_train, g1_test = group_1[0:size], group_1[size:len(group_1)]
    # copy_files("1", "-1", g1_train)
    # print(len((g1_train)))
    # copy_files("test", "-1", g1_test)
    # print(len((g1_test)))


    plot_number_of_jobs_per_class(class_files)



if __name__ == "__main__":
    main()

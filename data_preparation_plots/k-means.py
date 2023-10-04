import ast
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def plot_number_of_jobs_per_class(class_files):
    num_values = [len(values) for values in class_files.values()]
    plt.bar(range(len(num_values)), num_values)
    plt.xticks(range(len(num_values)), class_files.keys())
    plt.title('Number of jobs per cluster')
    plt.xlabel('Clusters')
    plt.ylabel('Number of Jobs')
    plt.show()


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
    kmeans = KMeans(n_clusters=20, init='k-means++')
    labels = kmeans.fit_predict(matrix)
    print(labels)

    unique_labels = np.unique(labels)
    print("Number of clusters: " + str(len(unique_labels)))
    class_files = defaultdict(list)
    for i in range(len(labels)):
        class_files[labels[i]].append(files[i])
    plot_number_of_jobs_per_class(class_files)


if __name__ == "__main__":
    main()

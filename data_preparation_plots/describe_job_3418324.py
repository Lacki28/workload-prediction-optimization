import csv
import os
from datetime import timezone, timedelta

import pandas as pd
from matplotlib import pyplot as plt, ticker

csv_file = "../sortedGroupedJobFiles/3418324.csv"
txt_file = "job_description.txt"


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        df_list.append(pd.read_csv(file, sep=","))
    return df_list


def plot_results():
    data = pd.read_csv(csv_file, sep=",")


    indices = pd.DatetimeIndex(data["start_time"])
    indices = indices.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = indices[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    indices = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
               [first_timestamp + i * increment for i in range(len(indices))]]
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 15))
    axs.plot(indices,data["mean_CPU_usage"])
    axs.set_title('Job 3418324', fontsize=20)
    axs.set_xlabel('Time in days', fontsize=18)
    axs.set_ylabel('Mean CPU usage', fontsize=18)
    axs.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    axs.xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # Set x-axis tick frequency

    plt.savefig('3418324_description.png')
    plt.show()

def statistics():
    with open(csv_file, "r") as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        header = True
        mean_cpu_list = []
        tasks_list = []
        for row in datareader:
            # print(row)
            if header:
                print(row[1])
                print(row[14])
                header = False
            elif len(row) > 1 and row[1] not in mean_cpu_list:
                mean_cpu_list.append(float(row[1]))
                tasks_list.append(float(row[14]))
        f = open(txt_file, "a+")
        f.write(str(sum(mean_cpu_list) / len(mean_cpu_list)) + "\n")
        f.write(str(min(mean_cpu_list)) + "\n")
        f.write(str(max(mean_cpu_list)) + "\n")
        f.write(str(sum(tasks_list) / len(tasks_list)) + "\n")
        f.write(";")
        f.close()
def main():
    plot_results()
    statistics()


if __name__ == "__main__":
    main()

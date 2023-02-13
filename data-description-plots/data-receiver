from math import trunc

import pandas as pd
import matplotlib.pyplot as plt

in_seconds = 1000000
in_minutes = in_seconds * 60
in_hours = in_minutes * 60
in_days = in_hours * 24
# dir = "../Desktop/data/task_usage/part-"
dir = ""
files = 3


def main():
    # specify the names of the columns
    column_names = ["start_time", "end_time", "job_id", "task_index", "machine_id", "mean_CPU_usage",
                    "canonical_mem_usage", "assigned_mem_usage", "unmapped_page_cache_mem_usage",
                    "total_page_cache_mem_usage", "max_mem_usage", "mean_disk_IO_time",
                    "mean_local_disk_space_used", "max_CPU_usage", "max_disk_IO_time", "CPI", "MAI",
                    "sample_portion", "aggregation_type ", "sampled_CPU_usage"]
    file_indices = list(range(1, files, 1))
    # read the first file and add the column names
    data_frame = pd.read_csv(dir + "part-00000-of-00500.csv.gz", compression='gzip', header=None)
    data_frame.columns = column_names
    # group by the start time and get the mean value of all jobs that started at the same time
    df_mean = data_frame.groupby(["start_time"])["mean_CPU_usage"].sum()
    df_max = data_frame.groupby(["start_time"])["max_CPU_usage"].mean()

    #(["mean_CPU_usage"].mean(),["max_CPU_usage"].mean())
    #df_max = data_frame.groupby(["start_time"])["max_CPU_usage"].mean()
    # add the other files
    for file_index in file_indices:
        data = pd.read_csv(dir +"part-"+ str(file_index).zfill(5) + "-of-00500.csv.gz",
                           compression='gzip', header=None)
        data.columns = column_names
        df2_mean = data.groupby(["start_time"])["mean_CPU_usage"].sum()
        df2_max = data.groupby(["start_time"])["max_CPU_usage"].max()
        df_mean = pd.concat([df_mean, df2_mean])
        df_max = pd.concat([df_max, df2_max])
        index_in_h = ((df_mean.index - 600000000) / in_hours)
        if index_in_h.to_frame().iloc[-1].item() > 24:
            print((df_mean.index - 600000000) / in_hours)
            break

    plt.ylabel("CPU usage")
    plt.xlabel("start time in hours")
    plt.title("mean CPU usage of 24 hours")
    new_index = (df_mean.index - 600000000) / in_hours
    #df_mean = df_mean[new_index < 4]
    index = [trunc(x) for x in (new_index)]
    plt.bar(index, df_mean.values, label='mean')
    #plt.bar(index, df2_max.values, label='max')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()


if __name__ == "__main__":
    main()

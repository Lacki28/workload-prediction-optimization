import pandas as pd
import matplotlib.pyplot as plt

in_seconds = 1000000
in_minutes = in_seconds * 60
in_hours = in_minutes * 12  # each interval has 5 minutes
in_days = in_hours * 24
dir = ""
#dir = "../Desktop/data/task_usage/"

files = 5

machine_ids = ["4155527081", "317808204", "1436333600"]
data_frames = []


def main():
 first_7_days()

def first_7_days():
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
    for machine_id in machine_ids:
        # group by the start time and get the mean value of all jobs that started at the same time
        df_mean = data_frame.query("machine_id == " + machine_id).groupby(["start_time"])["mean_CPU_usage"].sum()
        # add the other files
        for file_index in file_indices:
            data = pd.read_csv(dir + "part-" + str(file_index).zfill(5) + "-of-00500.csv.gz",
                               compression='gzip', header=None)
            data.columns = column_names
            df2_mean = data.query("machine_id == " + machine_id).groupby(["start_time"])["mean_CPU_usage"].sum()
            df_mean = pd.concat([df_mean, df2_mean])
            #if we only want seven days include this and remove line 46 data_frames.append()
            # index_in_d = ((df_mean.index - 600000000) / in_days)
            # if index_in_d.to_frame().iloc[-1].item() > 7:  #
            #     data_frames.append(df_mean)
            #     break
        data_frames.append(df_mean)
    plt.ylabel("mean CPU usage")
    plt.xlabel("start time in days")
    index_intersection = data_frames[0].index.intersection(data_frames[1].index).intersection(data_frames[2].index)
    print(index_intersection)
    df1 = data_frames[0].loc[index_intersection]
    df2 = data_frames[1].loc[index_intersection]
    df3 = data_frames[2].loc[index_intersection]
    index_in_hours = ((index_intersection - 600000000) / in_days)
    plt.plot(index_in_hours, df1, label='machine id = ' + machine_ids[0], linewidth=0.5)
    plt.plot(index_in_hours, df2, label='machine id = ' + machine_ids[1], linewidth=0.5)
    plt.plot(index_in_hours, df3, label='machine id = ' + machine_ids[2], linewidth=0.5)

    # Calculate mean of values
    mean = (df1 + df2 + df3) / 3
    plt.plot(index_in_hours, mean.values, label='mean value')

    plt.legend()
    plt.savefig('CPU_mean_days.png')
    plt.show()

def first_24_hours():
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
    for machine_id in machine_ids:
        # group by the start time and get the mean value of all jobs that started at the same time
        df_mean = data_frame.query("machine_id == " + machine_id).groupby(["start_time"])["mean_CPU_usage"].sum()
        # add the other files
        for file_index in file_indices:
            data = pd.read_csv(dir + "part-" + str(file_index).zfill(5) + "-of-00500.csv.gz",
                               compression='gzip', header=None)
            data.columns = column_names
            df2_mean = data.query("machine_id == " + machine_id).groupby(["start_time"])["mean_CPU_usage"].sum()
            df_mean = pd.concat([df_mean, df2_mean])
            index_in_h = ((df_mean.index - 600000000) / in_hours)
            if index_in_h.to_frame().iloc[-1].item() > 24:  #
                data_frames.append(df_mean)
                break
    plt.ylabel("mean CPU usage")
    plt.xlabel("start time in hours")
    # plt.title("CPU usage of the first 25 hours")#genauer
    index_intersection = data_frames[0].index.intersection(data_frames[1].index).intersection(data_frames[2].index)
    print(index_intersection)
    df1 = data_frames[0].loc[index_intersection]
    df2 = data_frames[1].loc[index_intersection]
    df3 = data_frames[2].loc[index_intersection]
    index_in_hours = ((index_intersection - 600000000) / in_hours)
    plt.plot(index_in_hours, df1, label='machine id = ' + machine_ids[0], linewidth=0.5)
    plt.plot(index_in_hours, df2, label='machine id = ' + machine_ids[1], linewidth=0.5)
    plt.plot(index_in_hours, df3, label='machine id = ' + machine_ids[2], linewidth=0.5)

    # Calculate mean of values
    mean = (df1 + df2 + df3) / 3
    plt.plot(index_in_hours, mean.values, label='mean value')

    plt.legend()
    plt.savefig('CPU_mean_24_hours.png')
    plt.show()



if __name__ == "__main__":
    main()

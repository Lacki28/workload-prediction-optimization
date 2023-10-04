import csv
import os.path
import shutil
import threading
from io import open

import dask.dataframe as dd
import numpy as np
import pandas as pd
import seaborn as sns
from dask import delayed, compute
from matplotlib import pyplot as plt

# path of the task_usage folder
dir = "../../Desktop/data/task_usage/"
files = 500

data_frames = []
jobs_file = "job_ids_with_high_priority_tasks.csv"


@delayed
def delayed_dataframe_calculation(sum_heatmap):
    for filename in os.listdir("sortedGroupedJobFiles/"):
        filepath = os.path.join("sortedGroupedJobFiles/", filename)
        if os.path.isfile(filepath) and "sortedGroupedJobFiles" in filepath:
            data_frame = dd.read_csv(filepath)

            heatmap = data_frame.corr(method='pearson')
            heatmap = heatmap.fillna(0)
            if sum_heatmap is None:
                # If this is the first heatmap, initialize the sum_heatmap array
                sum_heatmap = heatmap
            else:
                # Otherwise, add the heatmap to the sum_heatmap array
                sum_heatmap += heatmap
    return sum_heatmap


# Step 4 - see what columns are relevant and should be kept
def correlation():
    sum_heatmap = None
    sum_heatmap = delayed_dataframe_calculation(sum_heatmap)
    sum_heatmap = sum_heatmap.compute()
    sns.set(font_scale=1)
    num_heatmaps = len(os.listdir("sortedGroupedJobFiles/"))
    avg_heatmap = np.divide(sum_heatmap, num_heatmaps)
    # filtered_heatmap = avg_heatmap.where(np.abs(avg_heatmap) >= 0.0)  # filter by threshold of 0.5
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.subplots_adjust(left=0.25, bottom=0.3)
    sns.heatmap(avg_heatmap, annot=True, cmap="coolwarm", ax=ax, xticklabels=sum_heatmap, yticklabels=sum_heatmap)
    # plt.show()
    plt.savefig('heatmap.png')


# add scheduling class column
def add_scheduling_class():
    data_frame = pd.read_csv(jobs_file, sep=" ")
    for index, row in data_frame.iterrows():
        file_path = os.path.join("sortedGroupedJobFiles/" + str(row[0]) + ".csv")
        if os.path.isfile(file_path):
            df = dd.read_csv("sortedGroupedJobFiles/" + str(row[0]) + ".csv", blocksize=256e6)
            df['scheduling_class'] = row[1]
            df.to_csv("sortedGroupedJobFiles/" + str(row[0]) + ".csv", index=False)


# Step 3.5
@delayed
def process_file(filename):
    filepath = os.path.join("sortedJobFiles/", filename)
    data_frame = dd.read_csv(filepath, blocksize=256e6)
    print(filename)
    data_frame = data_frame.groupby(['start_time']).agg({'mean_CPU_usage': 'sum',
                                                         'canonical_mem_usage': 'sum',
                                                         'assigned_mem_usage': 'sum',
                                                         'unmapped_page_cache_mem_usage': 'sum',
                                                         'total_page_cache_mem_usage': 'sum',
                                                         'max_mem_usage': 'sum',
                                                         'mean_disk_IO_time': 'sum',
                                                         'mean_local_disk_space_used': 'sum',
                                                         'max_CPU_usage': 'sum',
                                                         'max_disk_IO_time': 'sum',
                                                         'CPI': 'sum',
                                                         'MAI': 'sum',
                                                         'sampled_CPU_usage': 'sum',
                                                         'nr_of_tasks': 'count'}).sort_values(ascending=True,
                                                                                              by="start_time")
    computed_data_frame = data_frame.compute()
    computed_data_frame.to_csv('sortedGroupedJobFiles/' + filename)


# Step 3.5
def group_data_frames(files):
    tasks = [process_file(filename) for filename in files]
    # tasks are delayed objects - we compute the results by calling dask.compute on the list of delayed objects.
    compute(*tasks)


# Step 3.5
def group_data_frames_threaded(num_threads):
    # only take files that have not already been sorted
    files = [filename for filename in os.listdir("sortedJobFiles") if
             filename.endswith('.csv') and not os.path.isfile(
                 'sortedGroupedJobFiles/' + filename)]
    # each thread should roughly deal with the same amount of files
    files_per_thread = (len(files) + num_threads - 1) // num_threads

    # create threads
    threads = []
    for i in range(num_threads):
        start_index = i * files_per_thread
        end_index = min((i + 1) * files_per_thread, len(files))
        thread_files = files[start_index:end_index]
        thread = threading.Thread(target=group_data_frames, args=(), kwargs={'files': thread_files})
        threads.append(thread)

    # start threads
    for thread in threads:
        thread.start()

    # wait for threads to finish
    for thread in threads:
        thread.join()


# Step 3
def interpolate_partition(partition):
    # fill in missing values from both the forward and backward directions - use the nearest non-missing value in either direction to interpolate the missing value.
    sorted_partition = partition.sort_values("start_time")
    return sorted_partition.interpolate(method='nearest', limit_direction='both')


def remove_small_files(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir + "/" + filename)
        # Only use it if there are at least 4032 lines (indicating either many tasks or one job running for 2 weeks)
        # 4032 = 1*12(h)*24(day)*14
        if os.path.isfile(file_path) and sum(1 for _ in open(file_path)) < 4032:
            print(file_path)
            os.remove(file_path)


# Step 3 Instead of having blocks where the function is applied to each block, we can decorate functions with
# @delayed and have the functions themselves be lazy.
@delayed
def prepare_data_frame(filename):
    # When we use the @delayed, we are creating multiple delayed tasks that will each independently create a Dask DataFrame
    # object by executing the same code -> the final result of computing multiple delayed tasks that each create a Dask DataFrame object will
    # be a single Dask DataFrame object that includes the results of all the tasks, as if they had been executed sequentially.
    print(filename)
    filepath = os.path.join("daskJobFilesLong/", filename)
    data_frame = dd.read_csv(filepath, blocksize="10GB")
    data_frame = data_frame.drop_duplicates()
    data_frame = data_frame.drop(["scheduling_class"], axis=1)
    data_frame = data_frame[data_frame.max_CPU_usage != 0]
    data_frame = data_frame[data_frame.max_mem_usage != 0]
    data_frame['nr_of_tasks'] = 1
    data_frame['start_time'] = (data_frame[
                                    'start_time'] / 300000000).round() * 300000000  # some start times are not int the 5 min interval - handle them
    number_of_nan = data_frame.isna().sum().compute()
    rows = len(data_frame)
    empty_cols = number_of_nan[number_of_nan == rows].index.tolist()
    data_frame[empty_cols] = data_frame[empty_cols].fillna(0)
    data_frame = data_frame.map_partitions(interpolate_partition)
    # To get the result, call compute. Notice that this runs faster than the original code.
    computed_data_frame = data_frame.compute()
    computed_data_frame.to_csv('sortedJobFiles/' + filename, index=False)


# Step 3 - sort the files, so we get rid of the individual task rows and columns we do not need
def prepocessFiles(files):
    tasks = [prepare_data_frame(filename) for filename in files]
    # tasks are delayed objects - we compute the results by calling dask.compute on the list of delayed objects.
    compute(*tasks)


# Step 3 - sort the files, so we get rid of the individual task rows and columns we do not need
# It is done by multiple threads since, some files take a long time and this code can easily be done in parallel
def process_files_threaded(num_threads):
    # only take files that have not already been sorted
    files = [filename for filename in os.listdir("daskJobFilesLong") if
             filename.endswith('.csv') and not os.path.isfile(
                 'sortedJobFiles/' + filename)]
    # each thread should roughly deal with the same amount of files
    files_per_thread = (len(files) + num_threads - 1) // num_threads

    # create threads
    threads = []
    for i in range(num_threads):
        start_index = i * files_per_thread
        end_index = min((i + 1) * files_per_thread, len(files))
        thread_files = files[start_index:end_index]
        thread = threading.Thread(target=prepocessFiles, args=(), kwargs={'files': thread_files})
        threads.append(thread)

    # start threads
    for thread in threads:
        thread.start()

    # wait for threads to finish
    for thread in threads:
        thread.join()


# Step 2.5
def copy_larger_files():
    for filename in os.listdir("daskJobFiles"):
        file_path = os.path.join("daskJobFiles/", filename)
        # Only use it if there are at least 4032 lines (indicating either many tasks or one job running for 2 weeks)
        # 4032 = 1*12(h)*24(day)*14
        if os.path.isfile(file_path) and sum(1 for _ in open(file_path)) > 4032:
            # Copy the file to the destination directory
            shutil.copy(file_path, os.path.join("daskJobFilesLong", filename))


# Step 2
def iterate_one_partition(partition):
    for index, row in partition.iterrows():
        job_id = "{:.0f}".format(row[2])
        check_file = os.path.isfile("./daskJobFiles/" + job_id + ".csv")
        if check_file:
            with open("daskJobFiles/" + job_id + ".csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)


# Step 2
def fill_files():
    dask_df = dd.read_csv(dir + "part-*.csv.gz",
                          compression='gzip', header=None, assume_missing=True)
    dask_df.map_partitions(iterate_one_partition).compute()


# Step 1 - the jobfile includes the job with the scheduling class of jobs that have tasks with high priority (9-11)
# Hive was used to create the file
def create_relevant_files():
    data_frame = pd.read_csv(jobs_file, sep=" ")
    column_names = ["start_time", "end_time", "job_id", "task_index", "machine_id", "mean_CPU_usage",
                    "canonical_mem_usage", "assigned_mem_usage", "unmapped_page_cache_mem_usage",
                    "total_page_cache_mem_usage", "max_mem_usage", "mean_disk_IO_time",
                    "mean_local_disk_space_used", "max_CPU_usage", "max_disk_IO_time", "CPI", "MAI",
                    "sample_portion", "aggregation_type ", "sampled_CPU_usage", "nr_of_tasks", "scheduling_class"]
    for index, row in data_frame.iterrows():
        f = open("daskJobFiles/" + str(row[0]) + ".csv", 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(column_names)
        f.close()

def main():
    create_relevant_files()
    fill_files()
    copy_larger_files()  # remove small short
    process_files_threaded(12)
    print("preprocessed")
    remove_small_files(
        "sortedJobFiles")  # four files only have one line, remove them and files that are too short (35) - 3758 files
    group_data_frames_threaded(12)
    remove_small_files("sortedGroupedJobFiles")
    # We should now have 2261 job files left that have been preprocessed, sorted and filtered
    add_scheduling_class()
    correlation()

if __name__ == "__main__":
    main()

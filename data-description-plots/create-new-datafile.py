import pandas as pd

dir = "../Desktop/data/task_usage/"
files = 500
csv_file = "data_set_CPU.csv"


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
    data_frame = data_frame.drop(columns=["job_id", "task_index", "canonical_mem_usage", "assigned_mem_usage",
                                          "unmapped_page_cache_mem_usage",
                                          "total_page_cache_mem_usage", "max_mem_usage", "mean_disk_IO_time",
                                          "mean_local_disk_space_used", "max_disk_IO_time", "CPI", "MAI",
                                          "sample_portion", "aggregation_type "])
    data_frame.to_csv(csv_file, sep='\t', encoding='utf-8')

    # add the other files
    for file_index in file_indices:
        data = pd.read_csv(dir + "part-" + str(file_index).zfill(5) + "-of-00500.csv.gz",
                           compression='gzip', header=None)
        data.columns = column_names
        # data_frame = pd.concat([data_frame, data])

        data = data.drop(columns=["job_id", "task_index", "canonical_mem_usage", "assigned_mem_usage",
                                  "unmapped_page_cache_mem_usage",
                                  "total_page_cache_mem_usage", "max_mem_usage", "mean_disk_IO_time",
                                  "mean_local_disk_space_used", "max_disk_IO_time", "CPI", "MAI",
                                  "sample_portion", "aggregation_type "])
        data.to_csv(csv_file, mode='a', header=False)


if __name__ == "__main__":
    main()

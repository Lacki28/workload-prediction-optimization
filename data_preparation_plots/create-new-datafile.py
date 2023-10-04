    import pandas as pd


    def main():
        create_task_usage()
        create_task_events()
        create_task_constraints()
        create_job_events()
        create_machine_events()
        create_machine_attributes()


    def append_other_files(dir, csv_file):
        # add the other files
        file_indices = list(range(1, 500, 1))
        for file_index in file_indices:
            data = pd.read_csv(dir + "part-" + str(file_index).zfill(5) + "-of-00500.csv.gz",
                               compression='gzip', header=None)
            #data.columns = column_names
            data.to_csv(csv_file, mode='a', sep=',', header=False, index=False)


    def create_task_usage():
        # specify the names of the columns
        csv_file = "task_usage.csv"
        dir = "../../Desktop/data/task_usage/"
        column_names = ["start_time", "end_time", "job_id", "task_index", "machine_id", "mean_CPU_usage",
                        "canonical_mem_usage", "assigned_mem_usage", "unmapped_page_cache_mem_usage",
                        "total_page_cache_mem_usage", "max_mem_usage", "mean_disk_IO_time",
                        "mean_local_disk_space_used", "max_CPU_usage", "max_disk_IO_time", "CPI", "MAI",
                        "sample_portion", "aggregation_type ", "sampled_CPU_usage"]
        # read the first file and add the column names
        data_frame = pd.read_csv(dir + "part-00000-of-00500.csv.gz", compression='gzip', header=None)
        #data_frame.columns = column_names
        data_frame.to_csv(csv_file, sep=',', encoding='utf-8', index=False)
        append_other_files(dir, column_names, csv_file)


    def create_task_events():
        # specify the names of the columns
        csv_file = "task_events.csv"
        dir = "../../Desktop/data/task_events/"
        column_names = ["time", "missing_info", "job_id", "task_index", "machine_id", "event_type", "user",
                        "scheduling_class", "priority", "CPU_request", "mem_request", "disk_space_request",
                        "different_machines_restriction"]
        # read the first file and add the column names
        data_frame = pd.read_csv(dir + "part-00000-of-00500.csv.gz", compression='gzip', header=None)
        #data.columns = column_names
        data_frame.to_csv(csv_file, sep=',', encoding='utf-8', index=False)
        append_other_files(dir, column_names, csv_file)


    def create_task_constraints():
        # specify the names of the columns
        csv_file = "task_constraints.csv"
        dir = "../../Desktop/data/task_constraints/"
        column_names = ["time", "job_id", "task_index", "comparison_operator", "attribute_name", "attribute_value"]
        # read the first file and add the column names
        data_frame = pd.read_csv(dir + "part-00000-of-00500.csv.gz", compression='gzip', header=None)
        #data.columns = column_names
        data_frame.to_csv(csv_file, sep=',', encoding='utf-8', index=False)
        append_other_files(dir, column_names, csv_file)


    def create_job_events():
        # specify the names of the columns
        csv_file = "job_events.csv"
        dir = "../../Desktop/data/job_events/"
        column_names = ["time", "missing_info", "job_id", "event_type", "user", "scheduling_class", "job_name",
                        "logical_job_name"]
        # read the first file and add the column names
        data_frame = pd.read_csv(dir + "part-00000-of-00500.csv.gz", compression='gzip', header=None)
        #data.columns = column_names
        data_frame.to_csv(csv_file, sep=',', encoding='utf-8', index=False)
        append_other_files(dir, column_names, csv_file)


    def create_machine_events():
        # specify the names of the columns
        csv_file = "machine_events.csv"
        dir = "../../Desktop/data/machine_events/"
        column_names = ["time", "machine_id", "event_type", "platform_id", "CPUs", "mem", ]
        # read the first file and add the column names
        data_frame = pd.read_csv(dir + "part-00000-of-00001.csv.gz", compression='gzip', header=None)
        #data.columns = column_names
        data_frame.to_csv(csv_file, sep=',', encoding='utf-8', index=False)


    def create_machine_attributes():
        # specify the names of the columns
        csv_file = "machine_attributes.csv"
        dir = "../../Desktop/data/machine_attributes/"
        column_names = ["time", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"]
        # read the first file and add the column names
        data_frame = pd.read_csv(dir + "part-00000-of-00001.csv.gz", compression='gzip', header=None)
        #data.columns = column_names
        data_frame.to_csv(csv_file, sep=',', encoding='utf-8', index=False)


    if __name__ == "__main__":
        main()

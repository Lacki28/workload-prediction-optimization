import pandas as pd

def main():
    unique_jobs()

def unique_jobs():
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"


def unique_machines():
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"
    column_names = ["time", "machine_id", "event_type", "platform_id", "CPUs", "memory"]
    data_frame = pd.read_csv(csv_file, compression='gzip', header=None)
    data_frame.columns = column_names
    print(data_frame.columns)
    print(data_frame.index)
    machines = data_frame.nunique(axis="machine_id")
    print("number of machines is: " + str(machines))


# 10 748 565

if __name__ == "__main__":
    main()

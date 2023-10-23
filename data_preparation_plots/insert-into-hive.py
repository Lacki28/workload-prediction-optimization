from pyhive import hive
import pandas as pd


def create_table_machine_events(conn):
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"
    column_names = ["time", "machine_id", "event_type", "platform_id", "CPUs", "memory"]
    data_frame = pd.read_csv(csv_file, compression='gzip', header=None)
    data_frame.columns = column_names
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE machine_events (time INT, machine_ID INT, event_type INT, platform_ID STRING,cpus DOUBLE,'
        'memory DOUBLE) ROW FORMAT DELIMITED FIELDS TERMINATED BY "," STORED AS TEXTFILE')
    for row in data_frame.itertuples(index=False):
        query = 'INSERT INTO machine_events VALUES ({},{},{},{},{},{})'.format(row.time, row.machine_ID, row.event_type,
                                                                               row.platform_ID, row.cpus, row.memory)
        cursor.execute(query)
    # Load the data from the dataframe into Hive
    conn.commit()
    conn.close()


def create_table_machine_attributes(conn):
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"


def create_table_job_events(conn):
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"


def create_table_task_events(conn):
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"


def create_table_task_constraints(conn):
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"


def create_table_task_usage(conn):
    csv_file = "../Desktop/data/machine_events/part-00000-of-00001.csv.gz"


def main():
    conn = hive.Connection(host='localhost', port=10000, database='default')
    create_table_machine_events(conn)


if __name__ == "__main__":
    main()

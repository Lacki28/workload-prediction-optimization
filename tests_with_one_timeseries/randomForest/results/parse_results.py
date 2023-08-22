def get_max_value_with_prefix(data):


    # Step 1: Group the data entries based on the timestamp
    timestamp_groups = {}
    for key, value in data.items():
        timestamp = key[0]
        max_depth = key[1]
        if timestamp not in timestamp_groups:
            timestamp_groups[timestamp] = {}
        timestamp_groups[timestamp][max_depth] = value
    
    # Step 2: Find the best max depth for each timestamp
    best_max_depths = {}
    for timestamp, max_depth_values in timestamp_groups.items():
        best_max_depth = min(max_depth_values, key=lambda x: max_depth_values[x][0])
        best_max_depths[timestamp] = {
            'max_depth': best_max_depth,
            'values': max_depth_values[best_max_depth]
        }
    
    # Step 3: Print the timestamp, best max depth, and the corresponding values
    print("Best max depth and values for each timestamp:")
    for timestamp, info in best_max_depths.items():
        best_max_depth = info['max_depth']
        values = " & ".join(map(str, info['values']))  # Concatenate values with " & "
        print(f"{timestamp}: {best_max_depth}")
        print(f"{values}")
        print()


def main():
    file_path = 'rf.txt'

    data = {}

    # Read the line from the file
    with open(file_path, 'r') as file:
        for _ in range(3):  # for each sequence length
            for _ in range(2):  # for each hyperparameter
                # sequence length
                file.readline().strip()
                hyperparam = file.readline().strip()
                for i in range(18):
                    line = file.readline().strip()
                    if i % 3 == 0:
                        test_run = line
                        # data[test_run, hyperparam] = cpu_mse
                    if i % 3 == 2:
                        # cpu_line = line
                        values = line.split("&")
                        cpu_mse = [float(val) for val in values]
                        data[test_run, hyperparam] = cpu_mse

            get_max_value_with_prefix(data)


if __name__ == "__main__":
    main()

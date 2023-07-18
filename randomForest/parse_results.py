def get_max_value_with_prefix(full_data, data, prefix):
    min_value = float('inf')
    best_hyperparam = None

    for (key, hyperparam), value in data.items():
        if key == prefix and value[1] < min_value:
            min_value = value[1]
            best_hyperparam = hyperparam

    if best_hyperparam is not None:
        print(best_hyperparam)
        print(full_data[best_hyperparam])



def main():
    file_path = 'rf.txt'

    data = {}
    full_data = {}

    # Read the line from the file
    with open(file_path, 'r') as file:
        for _ in range (20):
            for i in range(6):
                hyperparam = ""
                cpu_mse = 0
                cpu_line = ""
                for i in range(4):
                    line = file.readline().strip()
                    if i == 0:
                        test_run = line
                    if i == 1:
                        hyperparam = line
                    elif i == 3:
                        cpu_line=line
                        values = line.split("&")
                        cpu_mse = [float(val) for val in values]
                data[test_run, hyperparam] = cpu_mse
                full_data[hyperparam] = cpu_line
            print(test_run)
            get_max_value_with_prefix(full_data, data, test_run)


if __name__ == "__main__":
    main()

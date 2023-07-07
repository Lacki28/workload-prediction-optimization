import numpy as np
from matplotlib import pyplot as plt


def remove_outliers(data):
    data_arr = np.array(data, dtype=np.float64)

    Q1 = np.percentile(data_arr, 25)
    Q3 = np.percentile(data_arr, 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return [x for x in data_arr if lower_bound <= x <= upper_bound]


def create_boxplot(loss, name, timestamp, dir):
    loss = remove_outliers(loss)
    print(loss)
    print(name)
    plt.boxplot(loss, labels=[name])
    plt.xlabel('Loss value')
    plt.ylabel('Metric')
    plt.title('Naive benchmark loss values for ' + str(timestamp + 1) + ' ahead prediction')
    plt.savefig('NB_' + dir + '_' + name + '_loss_t' + str(timestamp + 1) + '.png')


def calc_avg(lst):
    result = sum(lst) / len(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def get_avg_loss(dir):
    for timestamp in range(6):
        mae = []
        mse = []
        r2 = []
        nr = []
        total_time = []

        with open(dir + str(timestamp + 1) + ".txt", 'r') as file:
            for line in file:
                test_error_values = line.split(' & ')
                mae.append(float(test_error_values[0]))
                mse.append(float(test_error_values[1]))
                r2.append(float(test_error_values[2]))
                nr.append(float(test_error_values[3]))
                total_time.append(float(test_error_values[4]))
        with open('avg_' + dir + str(timestamp + 1) + ".txt", 'w') as file:
            file.write("mae & mse & r2 & nr & training & total \n")
            file.write(
                calc_avg(mae) + " & " + calc_avg(mse) + " & " + calc_avg(r2) + " & inf & - & " + calc_avg(total_time))
        create_boxplot(mae, "mae", timestamp, dir)
        create_boxplot(mse, "mse", timestamp, dir)
        create_boxplot(r2, "r2", timestamp, dir)


def create_files(file_name):
    with open(file_name + ".txt", 'r') as file:
        # Read each line in the file
        timestamp = 0
        line_counter = 0
        for line in file:
            if (line_counter == 2):
                with open(file_name + str(timestamp + 1) + ".txt", 'a') as file:
                    # Write content to the file
                    file.write(line)
            line_counter += 1
            if (line_counter == 3):
                timestamp += 1
                line_counter = 0
                timestamp = timestamp % 6


def main():
    create_files("training")
    create_files("test")
    get_avg_loss("test")
    get_avg_loss("training")


if __name__ == "__main__":
    main()

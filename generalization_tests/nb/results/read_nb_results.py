import statistics

import numpy as np
from matplotlib import pyplot as plt


def remove_outliers(data):
    cleaned_data = []
    for inner_list in data:
        data_arr = np.array(inner_list, dtype=np.float64)

        Q1 = np.percentile(data_arr, 25)
        Q3 = np.percentile(data_arr, 75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        print(len(data_arr))
        print(len([x for x in data_arr if lower_bound <= x <= upper_bound]))
        cleaned_data.append([x for x in data_arr if lower_bound <= x <= upper_bound])

    return cleaned_data


def create_boxplot(loss, name, timestamp, dir):
    print(dir + " " + name)
    loss = remove_outliers(loss)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    plt.boxplot(loss)
    plt.xlabel('Timestamps ahead', fontsize=20)
    plt.ylabel('Metric ' + name, fontsize=20)
    plt.title('Naive benchmark ' + name, fontsize=22)
    plt.savefig('NB_' + dir + '_' + name + '_loss_t' + str(timestamp + 1) + '.png')
    plt.close()


def calc_avg(lst):
    result = sum(lst) / len(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def calc_std(lst):
    result = statistics.stdev(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def get_avg_loss(dir):
    list_of_mae = []
    list_of_mse = []
    list_of_r2 = []
    list_of_nr = []
    list_of_total_time = []
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
        with open('avg_' + str(timestamp + 1) + ".txt", 'a+') as file:
            file.write(
                "\multirow{2}{*}{NB} & \multirow{2}{*}{" + dir + "} & {avg} & \multirow{2}{*}{-} &\multirow{2}{*}{4}& " +
                calc_avg(mae) + " & " + calc_avg(mse) + " & " + calc_avg(r2) + " & \multirow{2}{*}{$\infty$} & " +
                "\multirow{2}{*}{-}" + " & " "\multirow{2}{*}{" + calc_avg(
                    total_time) + "}\\\\\n")
            file.write(
                "&   & std & & &" + calc_std(mae) + " & " + calc_std(mse) + " & " + calc_std(r2) + " &  &  & \\\\\n")

        # with open('avg_' + dir + str(timestamp + 1) + ".txt", 'w') as file:
        #     file.write("mae & mse & r2 & nr  & total \n")
        #     file.write(
        #         calc_avg(mae) + " & " + calc_avg(mse) + " & " + calc_avg(r2) + " &  $\infty$ & 0 & " + calc_avg(total_time))
        list_of_mae.append(mae)
        list_of_mse.append(mse)
        list_of_r2.append(r2)
        list_of_nr.append(nr)
        list_of_total_time.append(total_time)
    create_boxplot(list_of_mae, "MAE", timestamp, dir)
    create_boxplot(list_of_mse, "MSE", timestamp, dir)
    create_boxplot(list_of_r2, "R2", timestamp, dir)


def create_files(file_name):
    with open(file_name + ".txt", 'r') as file:
        # Read each line in the file
        jobs = 0
        timestamp = 0
        line_counter = 0
        for line in file:
            if (line_counter == 2):
                with open(file_name + str(timestamp + 1) + ".txt", 'a') as file:
                    # Write content to the file
                    file.write(line)
            line_counter += 1
            if (line_counter == 3):
                jobs += 1
                line_counter = 0
                if jobs == 50:
                    jobs = jobs % 50
                    timestamp += 1


def main():
    # create_files("train")
    # create_files("validation")
    # create_files("test")
    # get_avg_loss("train")
    get_avg_loss("validation")
    get_avg_loss("test")


if __name__ == "__main__":
    main()

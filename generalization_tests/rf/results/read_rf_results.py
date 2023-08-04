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
    print(dir +" "+ name)
    loss = remove_outliers(loss)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    plt.boxplot(loss)
    plt.xlabel('Timestamps ahead', fontsize=20)
    plt.ylabel('Metric ' + name, fontsize=20)
    plt.title('Random forest ' + name, fontsize=22)
    plt.savefig('RF_' + dir + '_' + name + '_loss_t' + str(timestamp + 1) + '.png')
    plt.close()


def calc_avg(lst):
    result = sum(lst) / len(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)

def calc_std(lst):
    result = statistics.stdev(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def create_timestamp_files(group):
    mae = []
    mse = []
    r2 = []
    nr = []
    training_time = []
    total_time = []

    with open("rf_"+group + "_.txt", 'r') as file:
        counter = 0
        timestamp = 0
        for line in file:
            if counter % 2 == 1:
                test_error_values = line.split(' & ')
                mae.append(float(test_error_values[0]))
                mse.append(float(test_error_values[1]))
                r2.append(float(test_error_values[2]))
                nr.append(float(test_error_values[3]))
                training_time.append(float(test_error_values[4]))
                total_time.append(float(test_error_values[5]))
                with open("rf_"+group+ str(timestamp + 1) + ".txt", 'a') as file:
                    file.write(
                        test_error_values[0] + " & " + test_error_values[1] + " & " + test_error_values[2] + " & " +
                        test_error_values[3] + " & " + test_error_values[4] + " & " + test_error_values[5])
                timestamp += 1
                timestamp = timestamp % 6
            counter += 1





def get_avg_loss(dir):
    list_of_mae = []
    list_of_mse = []
    list_of_r2 = []
    list_of_nr = []
    list_of_training_time = []
    list_of_total_time = []
    for timestamp in range(6):
        mae = []
        mse = []
        r2 = []
        nr = []
        training_time = []
        total_time = []

        with open("rf_"+dir + str(timestamp + 1) + ".txt", 'r') as file:
            for line in file:
                test_error_values = line.split(' & ')
                mae.append(float(test_error_values[0]))
                mse.append(float(test_error_values[1]))
                r2.append(float(test_error_values[2]))
                nr.append(float(test_error_values[3]))
                training_time.append(float(test_error_values[4]))
                total_time.append(float(test_error_values[5]))
        with open('avg_' + str(timestamp + 1) + ".txt", 'a+') as file:
            file.write(
                "\multirow{2}{*}{RF} & \multirow{2}{*}{" + dir + "} & {avg} & \multirow{2}{*}{individual} &\multirow{2}{*}{1}& " +
                calc_avg(mae) + " & " + calc_avg(mse) + " & " + "{:e}".format(float(calc_avg(r2))) +" & " + calc_avg(nr) + " & " +
                "\multirow{2}{*}{" + calc_avg(training_time) + "}" + " & " "\multirow{2}{*}{" + calc_avg(
                    total_time) + "}\\\\\n")
            file.write(
                "&   & std & & &" + calc_std(mae) + " & " + calc_std(mse) + " & " + "{:e}".format(float(calc_std(r2))) + " & " + calc_std(
                    nr) + " &  & \\\\\n")
        list_of_mae.append(mae)
        list_of_mse.append(mse)
        list_of_r2.append(r2)
        list_of_nr.append(nr)
        list_of_training_time.append(training_time)
        list_of_total_time.append(total_time)
    create_boxplot(list_of_mae, "MAE", timestamp, dir)
    create_boxplot(list_of_mse, "MSE", timestamp, dir)
    create_boxplot(list_of_r2, "R2", timestamp, dir)

def main():
    # create_timestamp_files("test")
    # create_timestamp_files("validation")
    get_avg_loss("test")
    get_avg_loss("validation")


if __name__ == "__main__":
    main()

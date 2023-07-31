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
    plt.boxplot(loss, labels=[name])
    plt.xlabel('Loss value')
    plt.ylabel('Metric')
    plt.title('Random forest loss values for ' + str(timestamp + 1) + ' ahead prediction')
    plt.savefig('RF_' + dir + '_' + name + '_loss_t' + str(timestamp + 1) + '.png')
    plt.close()


def calc_avg(lst):
    result = sum(lst) / len(lst)
    rounded_result = round(result, 5)
    return str(rounded_result)


def create_timestamp_files(group):
    mae = []
    mse = []
    r2 = []
    nr = []
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
                total_time.append(float(test_error_values[4]))
                with open("rf_"+group+"t" + str(timestamp + 1) + ".txt", 'a') as file:
                    file.write(
                        test_error_values[0] + " & " + test_error_values[1] + " & " + test_error_values[2] + " & " +
                        test_error_values[3] + " & " + test_error_values[4] + " & " + test_error_values[5])
                timestamp += 1
                timestamp = timestamp % 6
            counter += 1




def get_avg_loss(group):
    for timestamp in range(6):
        mae = []
        mse = []
        r2 = []
        nr = []
        training_time = []
        total_time = []
        with open("rf_"+group+"t" + str(timestamp + 1) + ".txt", 'r') as file:
            counter = 0
            for line in file:
                if counter % 2 == 1:
                    test_error_values = line.split(' & ')
                    mae.append(float(test_error_values[0]))
                    mse.append(float(test_error_values[1]))
                    r2.append(float(test_error_values[2]))
                    nr.append(float(test_error_values[3]))
                    training_time.append(float(test_error_values[4]))
                    total_time.append(float(test_error_values[5]))
                counter += 1
        with open("avg_"+ group +"_t"+ str(timestamp + 1) + ".txt", 'w') as file:
            file.write("mae & mse & r2 & nr & training & total \n")
            file.write(
                calc_avg(mae) + " & " + calc_avg(mse) + " & " + calc_avg(r2) + " & "+calc_avg(nr)+" & "+calc_avg(training_time)+" & " + calc_avg(total_time))
        create_boxplot(mae, "mae", timestamp, group)
        create_boxplot(mse, "mse", timestamp, group)
        create_boxplot(mse, "nr", timestamp, group)
        create_boxplot(r2, "r2", timestamp, group)


def main():
    create_timestamp_files("test")
    create_timestamp_files("validation")
    get_avg_loss("test")
    get_avg_loss("validation")


if __name__ == "__main__":
    main()

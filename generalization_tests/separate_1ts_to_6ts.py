import numpy as np
from matplotlib import pyplot as plt


def create_boxplot(mae, mse, r2, nr, timestamp, dir):
    data = [mae, mse, nr]

    means = [np.mean(arr) for arr in data]
    stds = [np.std(arr) for arr in data]

    # Set the threshold as a multiple of the standard deviation (e.g., 3 times)
    threshold = 3

    # Remove outliers from each array within the data
    data_without_outlier = []
    for arr, mean, std in zip(data, means, stds):
        data_without_outlier.append([value for value in arr if abs(value - mean) <= threshold * std])

    labels = ['MAE', 'MSE', 'NR']
    plt.boxplot(data_without_outlier, labels=labels)
    plt.xlabel('Loss value')
    plt.ylabel('Metric')
    plt.title('LSTM loss values for ' + str(timestamp + 1) + ' ahead prediction')
    plt.savefig('LSTM_' + dir + '_loss_t' + str(timestamp + 1) + '.png')

    mean = np.mean(r2)
    std = np.std(r2)

    # Set the threshold as a multiple of the standard deviation (e.g., 3 times)
    threshold = 3

    # Remove the outlier from the data
    r2_without_outlier = [value for value in r2 if abs(value - mean) <= threshold * std]

    # Create a boxplot with the updated data
    plt.boxplot(r2_without_outlier, labels=['R2'])
    plt.xlabel('Loss value')
    plt.ylabel('Metric')
    plt.title('LSTM r2 values for ' + str(timestamp + 1) + ' ahead prediction')
    plt.savefig('LSTM_' + dir + '_r2_loss_t' + str(timestamp + 1) + '.png')


def calc_avg(lst):
    return sum(lst) / len(lst)
def get_avg_loss(dir):
    for timestamp in range(6):
        mae = []
        mse = []
        r2 = []
        nr = []
        training_time = []
        total_time = []

        with open(dir + str(timestamp) + ".txt", 'r') as file:
            for line in file:
                test_error_values = line.split(' & ')
                mae.append(float(test_error_values[0]))
                mse.append(float(test_error_values[1]))
                r2.append(float(test_error_values[2]))
                nr.append(float(test_error_values[3]))
                training_time.append(float(test_error_values[4]))
                total_time.append(float(test_error_values[5]))
        with open('avg_'+dir + str(timestamp) + ".txt", 'w') as file:
            file.write("avg MAE "+calc_avg(mae))
            file.write("avg MAE " + calc_avg(mae))
            file.write("avg MSE " + calc_avg(mse))
            file.write("avg MAE " + calc_avg(mae))
            file.write("avg R2 " + calc_avg(r2))
            file.write("avg NR " + calc_avg(nr))
            file.write("avg training_time " + calc_avg(training_time))
            file.write("avg total_time " + calc_avg(total_time))
        create_boxplot(mae, mse, r2, nr, timestamp, dir)
        print(mae)


def create_files():
    with open("nb.txt", 'r') as file:
        # Read each line in the file
        timestamp = 0
        line_counter = 0
        first_10_lines = 0
        for line in file:
            if first_10_lines != 8:
                first_10_lines += 1
            else:
                if (line_counter == 3):
                    with open("training" + str(timestamp) + ".txt", 'a') as file:
                        # Write content to the file
                        file.write(line)
                if (line_counter == 6):
                    with open("test" + str(timestamp) + ".txt", 'a') as file:
                        # Write content to the file
                        file.write(line)
                line_counter += 1
                if (line_counter == 8):
                    line_counter = 0
                    timestamp += 1
                    timestamp = timestamp % 6


def main():
    # create_files()
    get_avg_loss("test")
    get_avg_loss("training")


if __name__ == "__main__":
    main()

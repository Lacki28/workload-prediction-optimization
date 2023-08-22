import time
import warnings
from datetime import timezone, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from matplotlib import ticker
from numpy import array
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.vector_ar.var_model import VAR


def naive_ratio(t, prediction, real_value):
    prediction = array(prediction)
    real_value = array(real_value)
    # Compute the absolute difference between corresponding elements of a and b
    prediction_nr = prediction[t:]
    real_value_nr = real_value[:-t]
    abs_diff_et1 = np.abs(prediction_nr - real_value_nr)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = np.sum(abs_diff_et1)
    et1 = (1 / len(prediction_nr)) * sum_abs_diff_et1
    abs_diff = np.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = np.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / et1


def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, y_test_pred, y_test)
    append_to_file(file_path, "mae & mse & r2 & nr & training & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(r2) + " & " + str(
                       np.round(nr, decimals=5)) + " & " + str(training_time) + " & " + str(
                       round((time.time() - start_time), 2)))


def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")


def plot_results(t, sequence_length, csv, act_cpu, act_mem, pred_cpu, pred_mem, target, size):
    indices = csv.index
    indices = indices[size + t - 1:]
    indices = [str(period) for period in indices]
    # plot forecasts against actual outcomes

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
    axs[0].plot(indices, act_cpu, label='actual ' + target[0], linewidth=1,
                color='orange')
    axs[0].plot(indices, pred_cpu, label='predicted ' + target[0], linewidth=1, color='blue', linestyle='dashed')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel(target[0])
    axs[0].set_title('VAR ' + target[0] + ' prediction h=' + str(sequence_length) + ', t=' + str(t))
    axs[0].legend()
    axs[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    axs[0].xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # Set x-axis tick frequency

    axs[1].plot(indices, act_mem, label='actual ' + target[1], linewidth=1,
                color='orange')
    axs[1].plot(indices, pred_mem, label='predicted ' + target[1], linewidth=1, color='blue', linestyle='dashed')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel(target[1])
    axs[1].set_title('VAR ' + target[1] + ' prediction h=' + str(sequence_length) + ', t=' + str(t))
    axs[1].legend()
    axs[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    axs[1].xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # Set x-axis tick frequency
    plt.savefig('VAR_' + 'h' + str(sequence_length) + '_t' + str(t) + '' + '.png')
    plt.savefig('VAR_' + 'h' + str(sequence_length) + '_t' + str(t) + '' + '.png')


def calculate_prediction_results(t, act_cpu, act_mem, pred_cpu, pred_mem, file_path, start_time, training_time):
    append_to_file(file_path, "TEST ERRORS CPU:")
    append_to_file(file_path, "CPU:")
    calc_MSE_Accuracy(t, act_cpu, pred_cpu, file_path, start_time, training_time)
    append_to_file(file_path, "MEM:")
    calc_MSE_Accuracy(t, act_mem, pred_mem, file_path, start_time, training_time)


def main(t, sequence_length, target):
    file_path = 'VARMA.txt'
    start_time = time.time()
    csv = pd.read_csv("../../sortedGroupedJobFiles/3418324.csv", sep=",", header=0)
    # set correct index
    csv.index = pd.DatetimeIndex(csv["start_time"])
    csv.index = csv.index.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = csv.index[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    csv.index = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                 [first_timestamp + i * increment for i in range(len(csv))]]

    data = csv.loc[:, target]
    size = int(len(data) * 0.7)
    train, test = data[0:size], data[size:len(data)]

    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length) + ", sequence length=" + str(
        sequence_length))
    training_time = round((time.time() - start_time), 2)
    history = train.tail(sequence_length)
    # walk-forward validation
    predictions = list()
    observations = list()
    for x in range(int(len(test) - t + 1)):
        history.iloc[0, 1] += 0.00000003
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model = VAR(history)
        model_fit = model.fit()
        output = model_fit.forecast(model_fit.endog, steps=t)
        predictions.append((output[0, 0], output[0, 1]))
        ith_entry = test.iloc[x + t - 1]
        observations.append((ith_entry[target[0]], ith_entry[target[1]]))
        data_to_add_to_history = data.iloc[size + x:size + x + t]
        history = pd.concat([history, data_to_add_to_history])
        history = history[-sequence_length:]
    act_cpu, act_mem = zip(*observations)
    pred_cpu, pred_mem = zip(*predictions)
    calculate_prediction_results(t, act_cpu, act_mem, pred_cpu, pred_mem, file_path, start_time, training_time)
    plot_results(t, sequence_length, csv, act_cpu, act_mem, pred_cpu, pred_mem, target, size)


if __name__ == "__main__":
    for t in (1, 2, 3, 6):
        for history in (288, 432, 576):  # 72, 288, 576
            main(t, history, ['mean_CPU_usage', 'canonical_mem_usage'])

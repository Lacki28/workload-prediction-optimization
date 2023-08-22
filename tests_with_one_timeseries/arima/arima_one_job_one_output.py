import time
import warnings
from datetime import timezone, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from matplotlib import ticker
from numpy import array
from pmdarima import auto_arima
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


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


def calculate_prediction_results(t, prediction, actual, start_time, training_time, path):
    for i in range(t):
        prediction_values = [arr[i] for arr in prediction]
        actual_values = [arr[i] for arr in actual]
        current_act_cpu_validation = actual_values
        current_pred_cpu_validation = prediction_values
        append_to_file(path, str(i + 1) + " timestamp ahead prediction")

        calc_MSE_Accuracy(t, current_act_cpu_validation, current_pred_cpu_validation,
                          path, start_time, training_time)
        print("DONE")


def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, y_test_pred, y_test)
    append_to_file(file_path, "mae & mse & r2 & nr & training & total")
    append_to_file(file_path,
                   str(mae) + " & " + str(mse) + " & " + str(r2) + " & " + str(
                       np.round(nr, decimals=5)) + " & " + str(round(training_time, 2)) + " & " + str(
                       round((time.time() - start_time), 2)))


def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")


def plot_results(t, sequence_length, df, actual_values_cpu, predictions_cpu, target):
    indices = pd.DatetimeIndex(df["start_time"])
    indices = indices.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = indices[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    indices = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
               [first_timestamp + i * increment for i in range(len(indices))]]
    indices = indices[int(len(df) * 0.7) + t - 1:]
    indices = [str(period) for period in indices]
    for i in range(t):
        current_predictions_cpu = [arr[i] for arr in predictions_cpu]
        current_actual_values_cpu = [arr[i] for arr in actual_values_cpu]
        # current_predictions_cpu = predictions_cpu[:, i]
        # current_actual_values_cpu = actual_values_cpu[:, i]
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
        axs.plot(indices, current_actual_values_cpu, label='actual ' + target, linewidth=1,
                 color='orange')
        axs.plot(indices, current_predictions_cpu, label='predicted ' + target, linewidth=1,
                 color='blue', linestyle='dashed')
        axs.set_xlabel('Time', fontsize=18)
        plt.xticks(rotation=45)  # 'vertical')
        plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
        axs.set_ylabel(target, fontsize=18)
        axs.set_title('ARIMA ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(i + 1), fontsize=20)
        axs.legend(fontsize=16)
        plt.savefig('ARIMA_' + 'h' + str(sequence_length) + '_t' + str(i + 1) + '' + '.png')


def main(t, sequence_length, target):
    file_path = 'ARIMA.txt'
    start_time = time.time()
    csv = pd.read_csv("../../sortedGroupedJobFiles/3418324.csv", sep=",", header=0)
    # set correct index
    csv.index = pd.DatetimeIndex(csv["start_time"])
    csv.index = csv.index.tz_localize(timezone.utc).tz_convert('US/Eastern')
    first_timestamp = csv.index[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
    increment = timedelta(minutes=5)
    csv.index = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in
                 [first_timestamp + i * increment for i in range(len(csv))]]

    data = csv.loc[:, [target]]

    size = int(len(data) * 0.7)
    train, test = data[0:size], data[size:len(data)]

    model = auto_arima(train, maxiter=100)
    order = model.order
    append_to_file(file_path, "t=" + str(t) + ", sequence length=" + str(sequence_length))
    append_to_file(file_path, str(order))
    training_time = round((time.time() - start_time), 2)
    history = train.head(sequence_length)
    # walk-forward validation
    predictions = list()
    observations = list()
    for x in range(int(len(test) - t + 1)):
        # print(history)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            output = model_fit.forecast(steps=t)
        except np.linalg.LinAlgError as e:
            print("Error occurred:", e)
            predictions.append([0, 0, 0, 0, 0, 0])
        else:
            if isinstance(output, pd.Series):
                output = output.values
            predictions.append(output)
        finally:
            observations.append(test.iloc[x:x + t][target].values)
            history = pd.concat([history, pd.DataFrame([test.iloc[x]], columns=history.columns)])
            history = history[-sequence_length:]
    calculate_prediction_results(t, predictions, observations, start_time, training_time, file_path)
    plot_results(t, sequence_length, csv, observations, predictions, target)


if __name__ == "__main__":
    for history in (1, 6, 12):
        main(6, history, 'mean_CPU_usage')

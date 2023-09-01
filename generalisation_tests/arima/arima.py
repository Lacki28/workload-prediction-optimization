import os
import time
import warnings
from datetime import timezone, timedelta

import numpy as np
import pandas as pd
import sklearn.metrics as sm
from pmdarima import auto_arima
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


def naive_ratio(t, prediction, real_value):
    prediction = np.array(prediction)
    real_value = np.array(real_value)
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
        calc_MSE_Accuracy(t, current_act_cpu_validation, current_pred_cpu_validation,
                          path + str(i + 1) + ".txt", start_time, training_time)
        print("DONE")


def calc_MSE_Accuracy(t, y_test, y_test_pred, file_path, start_time, training_time):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = sm.mean_squared_error(y_test, y_test_pred)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    nr = naive_ratio(t, y_test_pred, y_test)
    # append_to_file(file_path, "mae & mse & r2 & nr & training & total")
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


def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content)
            file.write('\n')
    except IOError:
        print("An error occurred while writing to the file.")


def predict(file_path, t, target, sequence_length, selected_files, dir):
    csv_files = read_files(selected_files)
    for csv in csv_files:
        start_time = time.time()
        data = csv.loc[:, [target[0]]]
        train, test = data[0:12], data[12:len(data)]
        model = auto_arima(train, maxiter=100)
        order = model.order
        training_time = time.time() - start_time
        append_to_file(file_path, str(order))
        # walk-forward validation
        predictions = list()
        observations = list()
        history = train.head(sequence_length)
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

        calculate_prediction_results(t, predictions, observations, start_time, training_time, dir)


def read_file_names(file_path, path, index_start, index_end):
    dir = "~/Documents/pythonScripts/new/" + path + "/"
    expanded_path = os.path.expanduser(dir)
    g0 = os.listdir(expanded_path)
    g0 = g0[index_start: index_end]
    g0_files = [expanded_path + filename for filename in g0]
    append_to_file(file_path, "jobs group " + path)
    append_to_file(file_path, str(g0))
    return g0_files


def read_files(training_files):
    training_files_csv = list()
    for file in training_files:
        csv = pd.read_csv(file, sep=",")
        # set correct index
        csv.index = pd.DatetimeIndex(csv["start_time"])
        csv.index = csv.index.tz_localize(timezone.utc).tz_convert('US/Eastern')
        first_timestamp = csv.index[0].replace(year=2011, month=5, day=1, hour=19, minute=0)
        increment = timedelta(minutes=5)
        csv.index = [timestamp.strftime("%Y-%m-%d %H:%M:%S") for timestamp in
                     [first_timestamp + i * increment for i in range(len(csv))]]

        training_files_csv.append(csv)
    return training_files_csv


def main(t, target, sequence_length):
    file_path = 'arima.txt'
    training_files = read_file_names(file_path, "0", 0, 50)
    validation_files = read_file_names(file_path, "0", 50, 100)
    test_files = read_file_names(file_path, "1", 0, 50)

    # predict(order, t, target, sequence_length, training_files,file_path, start_time, dir)
    predict(file_path, t, target, sequence_length, validation_files, "validation")
    predict(file_path, t, target, sequence_length, test_files, "test")


if __name__ == "__main__":
    main(t=6, target=['mean_CPU_usage'], sequence_length=4)

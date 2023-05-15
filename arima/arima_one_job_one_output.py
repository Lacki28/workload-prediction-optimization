import warnings
from datetime import timezone, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from matplotlib import pyplot, ticker
from numpy import array
from pmdarima import auto_arima
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


def naive_ratio(prediction, real_value):
    prediction = array(prediction)
    real_value = array(real_value)
    # Compute the absolute difference between corresponding elements of a and b
    prediction_nr = prediction[1:]
    real_value_nr = real_value[:-1]
    abs_diff_et1 = np.abs(prediction_nr - real_value_nr)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = np.sum(abs_diff_et1)
    et1 = (1 / len(prediction_nr)) * sum_abs_diff_et1
    abs_diff = np.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = np.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / et1


def calc_MSE_Accuracy(y_test, y_test_pred):
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = round(sm.mean_squared_error(y_test, y_test_pred), 5)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    print("Mean absolute error =", mae)
    print("Mean squared error =", mse)
    print("R2 score =", r2)
    nr = naive_ratio(y_test_pred, y_test)
    print("Naive ratio =", nr)


def main(n=2, sequence_length=12, target="mean_CPU_usage"):
    csv = pd.read_csv("../sortedGroupedJobFiles/3418324.csv", sep=",", header=0)
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
    print("n=" + str(n) + ", sequence length=" + str(sequence_length))
    print(order)
    history = train[-sequence_length:].values
    # walk-forward validation
    predictions = list()
    observations = list()
    for t in range(int(len(test) - n + 1)):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast(steps=n)
        predictions.append(output[0])
        observations.append(test.iloc[t + n - 1][target])
        history = np.append(history, test.iloc[t][target])
        history = history[-sequence_length:]
    indices = csv.index
    indices = indices[size + n - 1:]
    indices = [str(period) for period in indices]
    # plot forecasts against actual outcomes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,8))

    plt.subplots_adjust(bottom=0.2)  # Adjust the value as needed
    axs.plot(indices, observations, label='actual ' + target, linewidth=1, color='orange')
    axs.plot(indices, predictions, label='predicted ' + target, linewidth=1, color='blue', linestyle='dashed')
    axs.set_xlabel('Time')
    plt.xticks(rotation=45)  # 'vertical')
    plt.gca().xaxis.set_major_locator(ticker.IndexLocator(base=12 * 24, offset=0))  # print every hour
    axs.set_ylabel(target)
    axs.set_title('ARIMA ' + target + ' prediction h=' + str(sequence_length) + ', t=' + str(n))
    axs.legend()
    plt.savefig('ARIMA_' + 'h' + str(sequence_length) + '_t' + str(n) + '' + '.png')
    plt.show()
    calc_MSE_Accuracy(observations, predictions)


if __name__ == "__main__":
    main()
import numpy as np
from matplotlib import pyplot as plt

x = ['1', '2', '3', '4', '5', '6']
num_columns = 5
bar_width = 0.15
column_names = ["ARIMA", "NB", "RF", "LSTM", "Transformer"]
colors =['C0', 'C1', 'C2', 'C3', 'C4']
# MSE
values_x1 = [5.28e-08, 5.21e-08, 5.55e-08, 5.25e-08, 5.41e-08]
values_x2 = [9.12e-08, 9.02e-08, 8.00e-08, 7.44e-08, 7.37e-08]
values_x3 = [1.11e-07, 1.10e-07, 9.46e-08, 8.94e-08, 8.87e-08]
values_x4 = [1.24e-07, 1.23e-07, 1.03e-07, 1.01e-07, 9.95e-08]
values_x5 = [1.43e-07, 1.42e-07, 1.15e-07, 1.15e-07, 1.13e-07]
values_x6 = [1.66e-07, 1.65e-07, 1.23e-07, 1.23e-07, 1.28e-07]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            color=colors[i],label=f'Data {column_names[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=16)
plt.ylabel('MSE', fontsize=14)
plt.title('MSE of the prediction of job 3418324.', fontsize=14)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('MSE_univar.png')
plt.close()
# r2
values_x1 = [0.72667, 0.72978, 0.71282, 0.7283, 0.72007]
values_x2 = [0.52777, 0.53203, 0.58552, 0.61476, 0.61848]
values_x3 = [0.42303, 0.4267, 0.50958, 0.53651, 0.53985]
values_x4 = [0.35774, 0.36193, 0.46459, 0.47731, 0.48319]
values_x5 = [0.25741, 0.26112, 0.40023, 0.40251, 0.4117]
values_x6 = [0.13555, 0.13938, 0.35775, 0.35759, 0.33212]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'Data {column_names[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('R2 score', fontsize=12)
plt.title('R2 score of the prediction of job 3418324.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('R2_univar.png')
plt.close()


# NR arima rf lstm trans
values_x1 = [0.57532, 0.67374, 0.57452, 0.58288]
values_x2 = [0.7918, 0.8661, 0.8057, 0.80934]
values_x3 = [1.00306, 1.00758, 0.99891, 0.98877]
values_x4 = [1.26954, 1.17355, 1.19668, 1.18547]
values_x5 = [1.74742, 1.37891, 1.4995, 1.48976]
values_x6 = [2.71e+15, 1.93644, 2.77483, 2.13904]
num_columns = 4
column_names = ["ARIMA", "RF", "LSTM", "Transformer"]
values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
colors =['C0', 'C2', 'C3', 'C4']

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            color=colors[i],label=f'Data {column_names[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
max_y_value=3
plt.ylim(0, max_y_value)
plt.ylabel('Naïve Ratio score', fontsize=12)
plt.title('Naïve Ratio score of the prediction of job 3418324.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('NR_univar.png')
plt.close()









# TIME
train = [34.54, 0, 0.33, 432.54, 646.16]

total = [52.28, 0.38, 0.34, 437.85, 651.14]

# x = ['train', 'total']
num_columns = 5
bar_width = 0.15
column_names = ["ARIMA", "NB", "RF", "LSTM", "Transformer"]

# X-axis positions for each bar group
x = np.arange(len(column_names))


# Width of the bars
bar_width = 0.35

# Create the bar plot
plt.bar(x, train, width=bar_width, label='Train', align='center', alpha=0.7)
# Adjust the x-axis position for "Total" values
x_total = x + bar_width

# Create the bar plot for "Total" values
plt.bar(x_total, total, width=bar_width, label='Total', align='center', alpha=0.7)

# Set X-axis labels and their positions
plt.xticks(x + bar_width / 2, column_names)

# Set plot title and labels
plt.xlabel('Prediction method', fontsize=12)
plt.ylabel('Time in seconds', fontsize=12)
plt.title('Time of the prediction.', fontsize=12)
plt.legend(fontsize=9.7)
plt.savefig('time_univar.png')
plt.close()


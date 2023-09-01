import numpy as np
from matplotlib import pyplot as plt

x = ['1', '2', '3', '4', '5', '6']
num_columns = 4
bar_width = 0.15
column_names = ["Validation 1ts", "Test 1ts","Validation 6ts", "Test 6ts"]
colors = ['C4', 'C4', 'C2', 'C2']
alphas = [0.6,0.8, 0.6, 0.8]


values_x1 = [0.01601, 0.02034, 0.01033, 0.01983]
values_x2 = [0.01923, 0.02705, 0.01793, 0.03073]
values_x3 = [0.02171, 0.03269, 0.03084, 0.04507]
values_x4 = [0.02526, 0.03836, 0.04617, 0.06052]
values_x5 = [0.03248, 0.0447, 0.06658, 0.07842]
values_x6 = [0.0373, 0.04973, 0.09387, 0.09752]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'{column_names[i]}', alpha=alphas[i], color=f'{colors[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Comparison of the mean of the MSE.', fontsize=11)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('mean_comparison_lstm_gen.png')
plt.close()



x = ['1', '2', '3', '4', '5', '6']
num_columns = 4
bar_width = 0.15
column_names = ["Validation 1ts", "Test 1ts","Validation 6ts", "Test 6ts"]
colors = ['C4', 'C4', 'C2', 'C2']
alphas = [0.6,0.8, 0.6, 0.8]


values_x1 = [0.06511, 0.05899, 0.03958, 0.06369]
values_x2 = [0.07503, 0.08457, 0.06845, 0.1032]
values_x3 = [0.08228, 0.10788, 0.12169, 0.15381]
values_x4 = [0.09487, 0.13119, 0.18633, 0.21001]
values_x5 = [0.12515, 0.15599, 0.27575, 0.27608]
values_x6 = [0.145, 0.17797, 0.40007, 0.35116]


values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'{column_names[i]}', alpha=alphas[i], color=f'{colors[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Comparison of the standard deviation of the MSE.', fontsize=11)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('std_comparison_lstm_gen.png')
plt.close()


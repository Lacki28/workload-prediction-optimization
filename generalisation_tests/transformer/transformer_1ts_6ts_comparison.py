import numpy as np
from matplotlib import pyplot as plt

x = ['1', '2', '3', '4', '5', '6']
num_columns = 4
bar_width = 0.15
column_names = ["Validation 1ts", "Test 1ts","Validation 6ts", "Test 6ts"]
colors = ['C4', 'C4', 'C2', 'C2']
alphas = [0.6,0.8, 0.6, 0.8]


values_x1 = [0.00843,0.0178,0.03043,0.02782]
values_x2 = [0.0129,0.02548,0.03238,0.03534]
values_x3 = [0.01589,0.03152,0.03633,0.03778]
values_x4 = [0.01919,0.03737,0.03924,0.04655]
values_x5 = [0.02227,0.04235,0.04083,0.04932]
values_x6 = [0.0256,0.04725,0.04375,0.05202]


values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'{column_names[i]}', alpha=alphas[i], color=f'{colors[i]}')

plt.xlabel('Number of timestamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Comparison of the mean of the MSE.', fontsize=11)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('mean_comparison_transformer_gen.png')
plt.close()



x = ['1', '2', '3', '4', '5', '6']
num_columns = 4
bar_width = 0.15
column_names = ["Validation 1ts", "Test 1ts","Validation 6ts", "Test 6ts"]
colors = ['C4', 'C4', 'C2', 'C2']
alphas = [0.6,0.8, 0.6, 0.8]



values_x1 = [0.03274,0.05676,0.13299,0.09385]
values_x2 = [0.04833,0.08435,0.13468,0.12744]
values_x3 = [0.05823,0.10967,0.15232,0.13683]
values_x4 = [0.06959,0.13396,0.15899,0.18077]
values_x5 = [0.08111,0.15575,0.16525,0.19201]
values_x6 = [0.09411,0.17869,0.17731,0.20721]


values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'{column_names[i]}', alpha=alphas[i], color=f'{colors[i]}')

plt.xlabel('Number of timestamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Comparison of the standard deviation of the MSE.', fontsize=11)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('std_comparison_transformer_gen.png')
plt.close()


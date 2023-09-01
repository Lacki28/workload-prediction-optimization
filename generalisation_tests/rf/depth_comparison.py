import numpy as np
from matplotlib import pyplot as plt

x = ['1', '2', '3', '4', '5', '6']
num_columns = 4
bar_width = 0.15
column_names = ["4", "5", "6", "7"]
colors =['C0', 'C1', 'C2', 'C3']

# MSE
values_x1 = [0.01421,0.00967,0.00893,0.00878]
values_x2 = [0.01716,0.01289,0.01237,0.01236]
values_x3 = [0.01947,0.01513,0.01462,0.01461]
values_x4 = [0.02186,0.01754,0.01728,0.01739]
values_x5 = [0.02437,0.01985,0.01936,0.01945]
values_x6 = [0.02716,0.02268,0.02181,0.02177]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            color=colors[i],label=f'Max depth {column_names[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Average MSE of multiple validation jobs.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('MSE_univar.png')
plt.close()
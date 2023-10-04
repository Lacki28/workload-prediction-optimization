import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import tensor

#
x = ['1', '2', '3',  '6']


values_x1 = [ 0.72667, 0.72978, 0.66661,0.68186, 0.72284]
values_x2 = [0.52777 , 0.53203, 0.58768,0.60779, 0.6114]
values_x3 = [0.42303 , 0.4267, 0.50382,0.53817, 0.53817]
values_x4 = [0.42303 , 0.4267, 0.50382,0.53817, 0.53817]
values_x5 = [0.42303 , 0.4267, 0.50382,0.53817, 0.53817]
values_x6 = [ 0.1402, 0.13938 , 0.35841, 0.37075, 0.34547]


values = [values_x1, values_x2, values_x3, values_x6]
num_columns = 5
column_names=["ARIMA", "NB", "RF", "LSTM", "Transformer"]
# column_colors=["royalblue", "darkorange", "green", "crimson",  "purple"]
# column_colors = plt.cm.Set2(range(5))
x_indexes = np.arange(len(x))
bar_width = 0.15

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1)/2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width, label=f'Data {column_names[i]}')

plt.xlabel('Number of timestamps ahead',fontsize=12)
plt.ylabel('R2', fontsize=12)
plt.title('R2 of the prediction of job 3418324.', fontsize=12)


plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.show()



values_x1 = [ 5.28e-08, 5.21e-08 , 6.43e-08, 6.15e-08, 5.36e-08]
values_x2 = [ 9.12e-08, 9.02e-08,7.95e-08,  7.57e-08,7.50e-08 ]
values_x3 = [ 1.11e-07,1.10e-07 ,9.56e-08  , 9.20e-08,8.90e-08 ]
values_x6 = [ 1.56e-07, 1.65e-07,1.23e-07 ,  1.21e-07, 1.26e-07]
values = [values_x1, values_x2, values_x3, values_x6]
num_columns = 5
column_names=["ARIMA", "NB", "RF", "LSTM", "Transformer"]
# column_colors=["royalblue", "darkorange", "green", "crimson",  "purple"]
# column_colors = plt.cm.Set2(range(5))
x_indexes = np.arange(len(x))
bar_width = 0.15

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1)/2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width, label=f'Data {column_names[i]}')

plt.xlabel('Number of timestamps ahead',fontsize=12)
plt.ylabel('R2', fontsize=12)
plt.title('R2 of the prediction of job 3418324.', fontsize=12)


plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.show()
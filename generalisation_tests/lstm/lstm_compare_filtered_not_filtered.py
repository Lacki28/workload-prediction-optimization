import numpy as np
from matplotlib import pyplot as plt

x = ['1', '2', '3', '4', '5', '6']
num_columns = 2
bar_width = 0.15
column_names = ["Best result with max r2", "Best result with min loss"]
colors = ['C4', 'C0']

# MSE - 50
values_x1 = [0.04421, 0.03334]
values_x2 = [0.09694, 0.0487]
values_x3 = [0.1127, 0.04636]
values_x4 = [0.14887, 0.05973]
values_x5 = [0.15362, 0.07882]
values_x6 = [0.18186, 0.11091]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'{column_names[i]}', alpha=0.5, color=f'{colors[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('MSE of the prediction of job 3418324, with 50 epochs training, without filter.', fontsize=11)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('50ep_minmax_univar.png')
plt.close()

colors = ['C0', 'C1']
column_names = ["Trained without filter", "Trained with savgol filter"]
# MSE - 50
values_x1 = [0.03334, 0.01706]
values_x2 = [0.0487, 0.02156]
values_x3 = [0.04636, 0.02572]
values_x4 = [0.05973, 0.03035]
values_x5 = [0.07882, 0.03599]
values_x6 = [0.11091, 0.04291]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            label=f'{column_names[i]}', alpha=0.5, color=f'{colors[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('MSE of the prediction of job 3418324, with 50 epochs training.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('50ep_univar.png')
plt.close()

# 100
values_x1 = [0.01205, 0.01437]
values_x2 = [0.02241, 0.01949]
values_x3 = [0.02249, 0.02207]
values_x4 = [0.04148, 0.03245]
values_x5 = [0.04298, 0.04038]
values_x6 = [0.06511, 0.04715]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            alpha=0.75, label=f'{column_names[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('MSE of the prediction of job 3418324, with 100 epochs training.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('100ep_univar.png')
plt.close()

# filtered
column_names = ["100 epochs", "150 epochs"]

values_x1 = [0.01437, 0.01601]
values_x2 = [0.01949, 0.01923]
values_x3 = [0.02207, 0.02171]
values_x4 = [0.03245, 0.02526]
values_x5 = [0.04038, 0.03248]
values_x6 = [0.04715, 0.0373]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))
colors = ['C1', 'C1']
alpha = [0.75, 1]

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            alpha=alpha[i], label=f'{column_names[i]}', color=f'{colors[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('MSE of the prediction of job 3418324, with 100/150 epochs training.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('150_filtered_univar.png')
plt.show()
plt.close()

column_names = ["100 epochs", "150 epochs"]
colors = ['C0', 'C0']
alpha = [0.75, 1]

# values_x1 = [0.01416]
# values_x2 = [0.01828]
# values_x3 = [0.02003]
# values_x4 = [0.02978]
# values_x5 = [0.03712]
# values_x6 = [0.04381]

#NOT FILTERED 100, 150

values_x1 = [0.01205, 0.01712]
values_x2 = [0.02241, 0.02721]
values_x3 = [0.02249, 0.03224]
values_x4 = [0.04148, 0.05454]
values_x5 = [0.04298, 0.07413]
values_x6 = [0.06511, 0.0888]

values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
x_indexes = np.arange(len(x))

for i in range(num_columns):
    plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
            alpha=alpha[i], label=f'{column_names[i]}', color=f'{colors[i]}')

plt.xlabel('Number of time stamps ahead', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('MSE of the prediction of job 3418324, with 100 epochs training.', fontsize=12)
plt.xticks(x_indexes, x)
plt.legend(fontsize=9.7)
plt.savefig('150ep_not_filtered.png')
plt.close()

# x = ['1', '2', '3', '4', '5', '6']
# num_columns = 4
# bar_width = 0.15
# column_names = ["Data test", "Trained without filter", "Data test savgol", "Trained with savgol filter"]
#
# # R2
# values_x1 = [-23711205.52648, -11260.62711, -20916526.86882, -9887.04137]
# values_x2 = [-20908404.84991, -9968.85368, -20878972.15157, -9874.903]
# values_x3 = [-24906838.48666, -11913.55048, -21202900.66608, -10032.66678]
# values_x4 = [-26666713.72506, -12764.25618, -21207210.59865, -10034.34826]
# values_x5 = [-26825702.90205, -12826.84156, -22065784.13316, -10446.97745]
# values_x6 = [-29016405.71899, -14382.63744, -21972238.96262, -10756.77357]
# #
# # NR
# # values_x1 = [0.9532,0.89503,0.95381,0.89071]
# # values_x2 = [0.98463, 0.97619,0.97478,0.94722]
# # values_x3 = [ 1.00015,1.00017,1.00009 ,1.00149]
# # values_x4 = [1.01753 ,1.02191, 1.03092,1.06841]
# # values_x5 = [ 1.03621,1.0488, 1.07132,1.16609 ]
# # values_x6 = [ 1.15321,1.10178,1.43221,1.88345]
#
#
# values = [values_x1, values_x2, values_x3, values_x4, values_x5, values_x6]
# x_indexes = np.arange(len(x))
#
# for i in range(num_columns):
#     plt.bar(x_indexes + (i - (num_columns - 1) / 2) * bar_width, [values[j][i] for j in range(len(x))], width=bar_width,
#             label=f'{column_names[i]}')
#
# plt.xlabel('Number of time stamps ahead', fontsize=12)
# plt.ylabel('R2', fontsize=12)
# plt.title('R2 of the prediction of job 3418324.', fontsize=12)
# plt.xticks(x_indexes, x)
# plt.legend(fontsize=9.7)
# plt.savefig('R2_univar.png')
# plt.close()

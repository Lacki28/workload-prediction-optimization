# workload-prediction-optimization

## General description
This repo includes the code for the master thesis: [Towards Accurate Time Series  Predictions for Cloud Workloads](https://repositum.tuwien.at/handle/20.500.12708/189045) 

The predictions using only one time series can be found in the directory: test_with_one_timeseries

The predictions for multiple jobs are in the directory generalization_tests

The data was prepared using the files in the data_preparation_plots and the prepared jobs are in sortedGroupedJobFiles. 
The clustered data can be found in clustered_jobs, where on can see the jobs belonging to cluster 0 in folder 0 and 
jobs belonging to cluster -1 are in the folder 1.

Please be aware that the initial code has been relocated, and as a result, certain paths might be inaccurate. 
Additionally, I plan to enhance and release an improved version of the generalization part that will be easier to adapt/reuse.

## How to use test and training files
For general prediction of one timeseries, one can either use the clustered jobs in the clustered_jobs dir, or
the jobs in sortedGroupedJobFiles.

For training and testing on multiple jobs, one can either use the jobs in the clustered_jobs directory, or from the 
sortedGroupedJobFiles. 
To split the data sortedGroupedJobFiles into test and training, one can use this simple code:
```python
import os

file_list = os.listdir( "./sortedGroupedJobFiles")
file_list.sort()  

num_files = len(file_list)

train_ratio = 0.6
test_ratio = 0.2
validation_ratio = 0.2

train_index = int(train_ratio * num_files)
test_index = train_index + int(test_ratio * num_files)

train_files = file_list[:train_index]
validation_files = file_list[train_index:test_index]
test_files = file_list[test_index:]
```

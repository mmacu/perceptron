import csv
import numpy as np
regr_train_files=[
    'data.activation.train.100.csv',
    'data.activation.train.1000.csv',
    'data.activation.train.10000.csv',
    'data.activation.train.500.csv',
    'data.cube.train.100.csv',
    'data.cube.train.1000.csv',
    'data.cube.train.10000.csv',
    'data.cube.train.500.csv'
]
regr_test_files=[
    'data.activation.test.100.csv',
    'data.activation.test.1000.csv',
    'data.activation.test.10000.csv',
    'data.activation.test.500.csv',
    'data.cube.test.100.csv',
    'data.cube.test.1000.csv',
    'data.cube.test.10000.csv',
    'data.cube.test.500.csv'
]
def csv_data_read(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        data=list(reader)
        return np.float64(np.array(data[1:-1]))
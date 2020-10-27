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
class_train_files=[
'data.simple.train.100.csv',
'data.simple.train.1000.csv',
'data.simple.train.10000.csv',
'data.simple.train.500.csv',
'data.three_gauss.train.100.csv',
'data.three_gauss.train.1000.csv',
'data.three_gauss.train.10000.csv',
'data.three_gauss.train.500.csv'

]
class_test_files=[
'data.simple.test.100.csv',
'data.simple.test.1000.csv',
'data.simple.test.10000.csv',
'data.simple.test.500.csv',
'data.three_gauss.test.100.csv',
'data.three_gauss.test.1000.csv',
'data.three_gauss.test.10000.csv',
'data.three_gauss.test.500.csv'

]
pres_regr_train=[
'data.linear.train.100.csv',
'data.linear.train.1000.csv',
'data.linear.train.10000.csv',
'data.linear.train.500.csv',
'data.multimodal.train.100.csv',
'data.multimodal.train.1000.csv',
'data.multimodal.train.10000.csv',
'data.multimodal.train.500.csv',
'data.square.train.100.csv',
'data.square.train.1000.csv',
'data.square.train.10000.csv',
'data.square.train.500.csv'
]
pres_regr_test=[
'data.linear.test.100.csv',
'data.linear.test.1000.csv',
'data.linear.test.10000.csv',
'data.linear.test.500.csv',
'data.multimodal.test.100.csv',
'data.multimodal.test.1000.csv',
'data.multimodal.test.10000.csv',
'data.multimodal.test.500.csv',
'data.square.test.100.csv',
'data.square.test.1000.csv',
'data.square.test.10000.csv',
'data.square.test.500.csv'
]
pres_class_train=[
'data.XOR.train.100.csv',
'data.XOR.train.1000.csv',
'data.XOR.train.10000.csv',
'data.XOR.train.500.csv',
'data.circles.train.100.csv',
'data.circles.train.1000.csv',
'data.circles.train.10000.csv',
'data.circles.train.500.csv',
'data.noisyXOR.train.100.csv',
'data.noisyXOR.train.1000.csv',
'data.noisyXOR.train.10000.csv',
'data.noisyXOR.train.500.csv'

]
pres_class_test=[
'data.XOR.test.100.csv',
'data.XOR.test.1000.csv',
'data.XOR.test.10000.csv',
'data.XOR.test.500.csv',
'data.circles.test.100.csv',
'data.circles.test.1000.csv',
'data.circles.test.10000.csv',
'data.circles.test.500.csv',
'data.noisyXOR.test.100.csv',
'data.noisyXOR.test.1000.csv',
'data.noisyXOR.test.10000.csv',
'data.noisyXOR.test.500.csv'
]
def csv_data_read(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)
        data=list(reader)
        return np.float64(np.array(data[1:-1]))


from mlxtend.data import loadlocal_mnist

def read_train_mnist():
    X, y = loadlocal_mnist(
        images_path='/Users/marek/marek_files/priv/mini/perceptron/mnist/train-images.idx3-ubyte',
        labels_path='/Users/marek/marek_files/priv/mini/perceptron/mnist/train-labels.idx1-ubyte')

    return X,y




from data_reader import csv_data_read, class_train_files, class_test_files, pres_class_train, pres_class_test
from nn_visualisation import *

from nn_model import *
from data_reader import read_train_mnist





#,act_f=Sigm,act_fprim=Sigmprim)
input_size=1
output_size=1
path='projekt1/classification/'
path1='projekt1-oddanie/clasification/'

#train_f_name=path+class_train_files[5]
#test_f_name=path+class_test_files[5]
#pliki na oddaniu:
train_f_name=path1+pres_class_train[5]
test_f_name=path1+pres_class_test[5]
np.random.seed(323)
train_data=csv_data_read(train_f_name)
test_data=csv_data_read(test_f_name)




m1=nn_model([2,75,75,75,4],with_bias=True,act_f=ReLU0,act_fprim=ReLU0prim,
            learn_ratio=0.001,noise_level=0.1,classifier=True,change_m_ratio=0.90)


epochs=50
vis=Visualization(m1)
learning_error=m1.fit(train_data,epochs=epochs,vis=vis)
plt.ioff()
vis.add_drawing(m1,learning_error,train_data,test_data)

print('Mean error over test data set: '+str(m1.score(test_data)))
plt.show()


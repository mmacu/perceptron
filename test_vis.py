

from data_reader import csv_data_read, regr_train_files, regr_test_files
from nn_visualisation import *

from nn_model import *






#,act_f=Sigm,act_fprim=Sigmprim)
input_size=1
output_size=1
path='projekt1/regression/'
train_f_name=regr_train_files[5]
test_f_name=regr_test_files[5]
np.random.seed(343)
train_data=csv_data_read(path+train_f_name)
test_data=csv_data_read(path+test_f_name)



#m1=nn_model([1,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
#            learn_ratio=0.6)
#m1=nn_model([1,3,3,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
#            learn_ratio=0.05)
m1=nn_model([1,3,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
            learn_ratio=0.05,noise_level=0.2)
#ładny do cube:
#m1=nn_model([1,2,3,2,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
#            learn_ratio=0.6)
#też dobry do cube
#m1=nn_model([1,4,4,4,1],with_bias=True,
#            learn_ratio=0.2,bias=0.5)
#m1=nn_model([1,4,4,1],with_bias=True,
#            learn_ratio=0.1,bias=0.5)

#epochs=max(1,int(60000/len(train_data)))
epochs=150
vis=Visualization(m1)
learning_error=m1.fit(train_data,epochs=epochs,vis=vis)
plt.ioff()
vis.add_drawing(m1,learning_error,train_data,test_data)
print('Mean error over test data set: '+str(m1.score(test_data)))
plt.show()



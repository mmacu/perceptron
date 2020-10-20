

from data_reader import csv_data_read, regr_train_files, regr_test_files
from nn_visualisation import *

from nn_model import *






#,act_f=Sigm,act_fprim=Sigmprim)
input_size=1
output_size=1
path='projekt1/regression/'
train_f_name=regr_train_files[1]
test_f_name=regr_test_files[1]
np.random.seed(1974)
train_data=csv_data_read(path+train_f_name)
test_data=csv_data_read(path+test_f_name)


#ładny do cube:
#m1=nn_model([1,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
#            learn_ratio=0.6)
m1=nn_model([1,3,4,3,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
            learn_ratio=0.6)
#m1=nn_model([1,4,4,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
#            learn_ratio=0.6)
#m1=nn_model([1,2,3,2,1],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
#            learn_ratio=0.6)
#też dobry do cube:
#m1=nn_model([1,4,6,4,1],with_bias=True,
#            learn_ratio=0.2,bias=1)
#m1=nn_model([1,3,4,3,1],with_bias=True,
#            learn_ratio=0.2,bias=0.5)
#epochs=max(1,int(60000/len(train_data)))
epochs=5
learning_error=m1.fit(train_data,epochs=epochs)


plt.plot(learning_error)
plt.title("Learning process")
plt.show()
results=np.zeros(len(test_data))
test_error=np.zeros(len(test_data))
di=np.zeros(input_size)
for i in range(len(test_data)):

    di[0]=test_data[i][0]
    results[i]=m1.evaluate(di)

    test_error[i]=np.linalg.norm(results[i]-test_data[i][1])

plt.scatter(test_data.T[0],test_data.T[1],marker='.',label='Test data')
plt.scatter(test_data.T[0],results,marker='.',label='Regression')
plt.scatter(train_data.T[0],train_data.T[1],marker='.',label='Train data')
plt.legend()
plt.show()

vis=Visualization(m1)
vis.draw_solution()

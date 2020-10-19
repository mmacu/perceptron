

from data_reader import csv_data_read, regr_train_files, regr_test_files
from nn_visualisation import *

from nn_model import *






#,act_f=Sigm,act_fprim=Sigmprim)
input_size=1
output_size=1
path='projekt1/regression/'
train_f_name=regr_train_files[4]
test_f_name=regr_test_files[4]

train_data=csv_data_read(path+test_f_name)
test_data=csv_data_read(path+test_f_name)

input_range=(min(train_data.T[0])-0.2*(max(train_data.T[0])-min(train_data.T[0])),
             max(train_data.T[0])+0.2*(max(train_data.T[0])-min(train_data.T[0])))
output_range=(min(train_data.T[1])-0.2*(max(train_data.T[1])-min(train_data.T[1])),
             max(train_data.T[1])+0.2*(max(train_data.T[1])-min(train_data.T[1])))

m1=nn_model([1,3,4,3,1],input_range=input_range,output_range=output_range,with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
            learn_ratio=0.1)
#great for cube:
#m1=nn_model([1,3,4,3,1],input_range=input_range,output_range=output_range,with_bias=True,
#            learn_ratio=0.2)
epochs=max(1,int(100000/len(train_data)))

learning_error=np.zeros(epochs*len(train_data))
input=np.zeros(input_size)
expected=np.zeros(output_size)
for i in range(epochs*len(train_data)):
    input[0]=train_data[i%len(train_data)][0]
    result=m1.evaluate(input)
    expected[0]=train_data[i%len(train_data)][1]
    m1.backprop(input,result,expected)
    learning_error[i]=np.linalg.norm(result-expected)


plt.plot(learning_error)
plt.title("Learning process")
plt.show()
results=np.zeros(len(test_data))
test_error=np.zeros(len(test_data))
for i in range(len(test_data)):
    input[0]=test_data[i][0]
    results[i]=m1.evaluate(input)

    test_error[i]=np.linalg.norm(results[i]-test_data[i][1])

plt.scatter(test_data.T[0],test_data.T[1],marker='.',label='Test data')
plt.scatter(test_data.T[0],results,marker='.',label='Regression')
plt.scatter(train_data.T[0],train_data.T[1],marker='.',label='Train data')
plt.legend()
plt.show()

vis=Visualization(m1)
vis.draw_solution()

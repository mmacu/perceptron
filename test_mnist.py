import pandas

from data_reader import csv_data_read, class_train_files, class_test_files, pres_class_train, pres_class_test, \
    read_test_mnist
from nn_visualisation import *

from nn_model import *
from data_reader import read_train_mnist





#,act_f=Sigm,act_fprim=Sigmprim)


np.random.seed(323)

X,y=read_train_mnist()
Xt,yt=read_test_mnist()
y=y+1
yt=yt+1

XY=np.zeros((len(X),len(X[0])+1),dtype=np.float64)
XYt=np.zeros((len(Xt),len(Xt[0])+1),dtype=np.float64)
for i in range(len(X)):
    XY[i][0:len(X[i])]=X[i]

    XY[i][-1]=y[i]

for i in range(len(Xt)):
    XYt[i][0:len(Xt[i])] = Xt[i]

    XYt[i][-1] = yt[i]


m1=nn_model([len(X[0]),800,400,400,10],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
            learn_ratio=0.0005,with_noise=False,noise_level=0.001,classifier=True,change_m_ratio=0.99)


epochs=1

vis=SimpleVis(m1)
learning_error=m1.fit(XY,epochs=epochs,vis=vis)

plt.plot(learning_error)


print('Mean error over test data set: '+str(m1.score(XYt)))
plt.show()


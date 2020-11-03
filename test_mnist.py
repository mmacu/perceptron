import pandas

import pandas
import pickle
from data_reader import csv_data_read, class_train_files, class_test_files, pres_class_train, pres_class_test, \
    read_test_mnist
from nn_visualisation import *

from nn_model import *
from data_reader import read_train_mnist





#,act_f=Sigm,act_fprim=Sigmprim)

path='projekt1/classification/'
path1='projekt1-oddanie/clasification/'

#train_f_name=path+class_train_files[5]
#test_f_name=path+class_test_files[5]
#pliki na oddaniu:
#train_f_name=path1+pres_class_train[5]
#test_f_name=path1+pres_class_test[5]
np.random.seed(323)
#train_data=csv_data_read(train_f_name)
#test_data=csv_data_read(test_f_name)


X,y=read_train_mnist()
Xt,yt=read_test_mnist()


XY=np.zeros((len(X),len(X[0])+1),dtype=np.float64)
XYt=np.zeros((len(Xt),len(Xt[0])+1),dtype=np.float64)
for i in range(len(X)):
    XY[i][0:len(X[i])]=X[i]

    XY[i][-1]=y[i]

for i in range(len(Xt)):
    XYt[i][0:len(Xt[i])] = Xt[i]

    XYt[i][-1] = yt[i]




m1=nn_model([len(X[0]),800,400,200,10],with_bias=True,act_f=Sigm2,act_fprim=Sigmprim2,
            learn_ratio=0.0005,with_noise=False,classifier=True,change_m_ratio=0.99)


epochs=30

vis=SimpleVis(m1)

learning_error=m1.fit(XY,epochs=epochs,vis=vis)

plt.plot(learning_error)

sc=m1.score(XYt)
print('Średni błąd na zbiorze testowym: '+str(sc))
print('Efektywność na zbiorze testowym: '+str(1-sc))

plt.show()

"""k=0
for i in range(1000):
    r=m1.evaluate(Xt[i])
    if r[0]!=yt[i]:
        k+=1
        px=np.array(Xt[i])
        px.shape=(28,28)
        plt.imshow(px, cmap='gray', vmin=0, vmax=255)
        plt.title(str(k)+" Rezultat:"+ str(r[0]-1)+" Oczekiwany: "+str(yt[i]-1))
        plt.show()"""

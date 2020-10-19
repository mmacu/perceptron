from nn_visualisation import *

from nn_model import *






m1=nn_model([2,4,4,2])


input=np.array([0.5,0.5])
expected=np.array([-0.5,0.2])
iterations=10000
learning_error=np.zeros(iterations)
for i in range(10000):
    input[0]=np.random.random()*2-1
    result=m1.evaluate(input)
    expected[0]=-input[0]
    expected[1]=input[0]/2
    m1.backprop(input,result,expected)
    learning_error[i]=np.linalg.norm(result-expected)

plt.plot(learning_error)
plt.show()




vis=Visualization(m1)
vis.draw_solution()

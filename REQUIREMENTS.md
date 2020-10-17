# Perceptron
A simple perceptron implementation to solve classification and regression problems.

## Project requirements:

### 2.1 Part One (15 points)
During the first three weeks, you should build a program that allows you to solve the classification and regression problem using the multilayer perceptron and visualize its operation, including training the network with the error backpropagation algorithm. It is also required to visualize the network training results for the regression problem for the R → R function and the vector classification problem in R2.

#### 2.1.1 Data sets
in the first class there will be a set of data sets on which to conduct experiments to test the effectiveness of the network,
in the week when the final commissioning takes place, it will be necessary to demonstrate the operation of the program on other, previously not shared data sets.
#### 2.1.2 Implementation Notes
The implementation should be done at the basic level, so that the code shows understanding of the principles of network operation and training, libraries implementing neural networks cannot be used.

#### 2.1.3 Elements to be implemented
possibility of initiating a (repeatable) learning process with a given seed of a random number generator
easy configuration of the number of layers in the network and neurons in the layer, the presence of biases (on the day of commissioning, you will have to quickly adapt the network architecture)
visualization of the training set and the effects of classification and regression
visualization of the error propagated in subsequent learning iterations (on each of the weights)
visualization of weight values ​​in successive learning iterations
#### 2.1.4 Items to be tested
Effect of activation function on network performance - check the sigmoid function and any two other activation functions within the network. Note: the output activation function must be selected according to the type of problem.
Effect of the number of hidden layers in the network and their number. Explore different numbers of layers from 0 to 4, several different architectures
Effect of the error measure at the network output on the learning efficiency. Look for two measures of error for classification and two for regression.
The results and conclusions must be described in the project report.

#### 2.2 Part two (5 points)
Using the implementation created in the first part, by modifying only the network parameters (the number of layers, neurons, activation function, etc.), but not extending the implementation with new functionalities, teach the network to recognize digits from the MNIST set. You can modify the program code to optimize it. A technique that will definitely improve performance will be the departure from the object-oriented representation of neurons in favor of arrays of values ​​in each of the layers.

The assessment from this stage depends on the effectiveness of the network on the test set:

|Score|points|
|-----|------|
|5    |   97%|
|4    |   94%|
|3    |   90%|
|2    |   80%|
|1    |   65%|
|0    |    0%|

The MNIST set is available here: http://yann.lecun.com/exdb/mnist/ In the process of learning the network, we ONLY use the set marked as training, and we use the test set only to assess the quality of the already learned network.

The network submitted for performance evaluation must allow the learning process to be repeated. This means that you have to provide all the necessary program parameters to repeat the training. If the training uses a pseudorandom number generator, you must also provide a seed generator so that the training process can be repeated exactly.

Attention. Learning on a large set takes a correspondingly long time. As part of the initial experiments, you can select a subset of the training data and learn only from them, and only after the initial recognition do full learning.

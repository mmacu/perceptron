import numpy as np

from nn_visualisation import Visualization
def RelU(x:np.float64):
    return np.float64(max(0,x))

def RelUprim(x:np.float64):
    if x>0:
        return 1.0
    else:
        return 0.01
    

def ReLU0(x:np.float64):
    y=np.float64(min(max(-1.0,x),1.0))
    return y

def ReLU0prim(x:np.float64):
    if x>-1.0 and x<1:
        return 1.0
    else:
        return 0.01


def Sigm(x:np.float64):
    return 1.0/(1.0+np.exp(-x*2.0))

def Sigm2(x:np.float64):
    return (2.0/(1.0+np.exp(-2.0*x)))-1.0

def Sigmprim(x:np.float64):
    return np.float64(2.0*x*(1.0-x))

def Sigmprim2(x:np.float64):
    if (x<1 and x>-1):
        return 2.0*(1.0+x)*(1.0-x)
    else:
        return 0.0

def quadr_err(expected,result):
    return 2*(expected - result)

def linear_err(expected,result):
#data is normalized, so we assume 0 if less than 1%
    if expected - result > 0.01:
        return 1
    else:
        if expected - result < -0.01:
            return -1
        else:
            return 0

class nn_model:
    def __init__(self,layersizes,input_range=(-1,1),output_range=(-1,1),act_f=ReLU0,act_fprim=ReLU0prim,learn_ratio=0.1,change_m_ratio=0.5,
                 with_bias=True,bias=0.5,with_noise=True,err_m=quadr_err):


        self.num_of_layers=len(layersizes)
        self.err_m=err_m
        self.layers=layersizes
        if with_bias:
            self.layers[0]+=1
        self.weights = list()
        self.actvalues = list()
        self.v_values=list()
        self.change_momentum=list()
        self.change_momentum_ratio=change_m_ratio
        self.input_range=input_range
        self.output_range=output_range
        self.act_f=np.vectorize(act_f)
        self.act_fprim=np.vectorize(act_fprim)
        self.learn_ratio=learn_ratio
        self.with_bias=with_bias
        self.bias=bias
        self.with_noise=with_noise
        self.noise_level=0.7
        for i in range(self.num_of_layers-1):
            #self.weights.append(np.zeros((self.layers[i+1],self.layers[i]))+0.5)
            self.weights.append(np.random.rand(self.layers[i+1],self.layers[i]))
            self.weights[-1]=self.weights[-1]*2.0-1.0 #second shape arg is number of columns in matrix - this is source layer
            self.change_momentum.append(np.zeros((self.layers[i+1],self.layers[i])))
        for i in range(self.num_of_layers):
            self.actvalues.append(np.zeros(self.layers[i]))
            self.v_values.append(np.zeros(self.layers[i]))
    def set_data_ranges(self,train_data):
        input_size = self.layers[0] - int(self.with_bias)
        output_size = self.layers[-1]
        data_size=len(train_data[0])
        inp=train_data.T[0:input_size].flatten()
        outp=train_data.T[(data_size-output_size):data_size].flatten()
        self.input_range = (min(inp) - 0.2 * (max(inp) - min(inp)),
                       max(inp) + 0.2 * (max(inp) - min(inp)))
        self.output_range = (min(outp) - 0.2 * (max(outp) - min(outp)),
                        max(outp) + 0.2 * (max(outp) - min(outp)))

    def check_bias_and_normalise(self,input_user):
        input_user = (input_user - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        if self.with_bias:
            input=np.zeros(self.layers[0])
            input[0:len(input_user)]=input_user
            input[self.layers[0]-1]=self.bias
            return input
        else:
            return input_user
    def output_normalise(self,output):
        return (output - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

    def output_denormalise(self,output):
        return output*(self.output_range[1]-self.output_range[0])+self.output_range[0]
    def backprop(self,input_user,result,expected):
        #backprop should be called after evaluation has been performed.
        #last layer is somewhat different that hidden ones

        input=self.check_bias_and_normalise(input_user)
        result=self.output_normalise(result)
        expected=self.output_normalise(expected)

        llidx=self.num_of_layers-1
        for l in range(self.num_of_layers-1,0,-1):
            if l==llidx:
                out_delta=2.0*self.err_m(expected,result)*self.act_fprim(self.actvalues[l]) #delta vector on last layer
            else:
                out_delta=2.0*((self.weights[l].T.dot(prev_out_delta)))*self.act_fprim(self.actvalues[l]) #inner and first
            if self.with_noise==True:
                noise=np.random.random(len(out_delta))
                out_delta+=out_delta*(noise*2-1)*self.noise_level
            #if len(out_delta)>1:
                #no_change=np.random.randint(len(out_delta))
                #out_delta[no_change]=0.0
            dW=np.zeros((self.layers[l],self.layers[l-1]))
            for i in range(self.layers[l]):
                dW[i]=out_delta[i]*self.actvalues[l-1]*self.learn_ratio #rows in dW matrix equal in size to source layer size.
                #no_change=np.random.randint(len(dW[i]))
                #dW[i][no_change]=0
            self.change_momentum[l - 1]=(1-self.change_momentum_ratio)*dW +self.change_momentum_ratio*self.change_momentum[l-1]
            self.weights[l-1]+=self.change_momentum[l-1]
            prev_out_delta=out_delta
            """if l==llidx:
                prev_out_delta=(expected-result)*self.act_fprim(self.actvalues[l]) #delta vector on last layer
            else:
                prev_out_delta=((self.weights[l].T.dot(prev_out_delta)))*self.act_fprim(self.actvalues[l])""" #inner and first

    def evaluate(self,input_user):
        input=self.check_bias_and_normalise(input_user)
        self.actvalues[0]=input
        for l in range(1,self.num_of_layers):
            self.v_values[l]=self.weights[l-1].dot(self.actvalues[l-1])
            self.actvalues[l]=self.act_f(self.v_values[l])

        return self.output_denormalise(self.actvalues[self.num_of_layers-1])


    def fit(self,train_data,epochs=5,vis=None,test_data=None):
        self.set_data_ranges(train_data)
        learning_error = np.zeros(epochs)
        input_size=self.layers[0]-int(self.with_bias)
        output_size=self.layers[-1]
        data_size=len(train_data[0])
        data_length=len(train_data)
        input = np.zeros(input_size)
        expected = np.zeros(output_size)
        np.random.shuffle(train_data)
        epoch_error=0.0
        for i in range(epochs * data_length):
            k = i%data_length
            input = train_data[k][0:input_size]
            result = self.evaluate(input)
            expected = train_data[k][(data_size-output_size):data_size]
            self.backprop(input, result, expected)
            epoch_error+=np.linalg.norm(expected-result)**2
            if i%data_length==data_length-1:
                learning_error[int(i/data_length)] = epoch_error/data_length
                epoch_error=0.0

                if vis!=None:
                    vis.add_drawing(self,learning_error,train_data,test_data)


        return learning_error


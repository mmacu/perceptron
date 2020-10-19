import numpy as np

def ReLU0(x:np.float64):
    return min(max(-1,x),1)

def ReLU0prim(x:np.float64):
    return max(np.float64(x>=-1 and x<=1),0.01)

class nn_model:
    def __init__(self,layersizes,input_range=(-1,1),output_range=(-1,1),act_f=ReLU0,act_fprim=ReLU0prim,learn_ratio=0.3,change_m_ratio=0.2):


        self.num_of_layers=len(layersizes)
        self.layers=layersizes
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
        for i in range(self.num_of_layers-1):
            self.weights.append(np.random.rand(self.layers[i+1],self.layers[i]))
            self.weights[-1]=self.weights[-1]*2-1 #second shape arg is number of columns in matrix - this is source layer
            self.change_momentum.append(np.zeros((self.layers[i+1],self.layers[i])))
        for i in range(self.num_of_layers):
            self.actvalues.append(np.zeros(self.layers[i]))
            self.v_values.append(np.zeros(self.layers[i]))

    def backprop(self,input,result,expected):
        #backprop should be called after evaluation has been performed.
        #last layer is somewhat different that hidden ones
        input=(input-self.input_range[0])/(self.input_range[1]-self.input_range[0])
        result=(result-self.output_range[0])/(self.output_range[1]-self.output_range[0])
        expected=(expected-self.output_range[0])/(self.output_range[1]-self.output_range[0])
        llidx=self.num_of_layers-1
        for l in range(self.num_of_layers-1,0,-1):
            if l==llidx:
                out_delta=(expected-result)*self.act_fprim(self.actvalues[l]) #delta vector on last layer
            else:
                out_delta=((self.weights[l].T.dot(prev_out_delta)))*self.act_fprim(self.actvalues[l]) #inner and first

            dW=np.zeros((self.layers[l],self.layers[l-1]))
            for i in range(self.layers[l]):
                dW[i]=out_delta[i]*self.actvalues[l-1]*self.learn_ratio #rows in dW matrix equal in size to source layer size.
            self.weights[l-1]+=dW+self.change_momentum_ratio*self.change_momentum[l-1]
            prev_out_delta=out_delta
            self.change_momentum[l-1]=dW

    def evaluate(self,input):

        self.actvalues[0]=(input-self.input_range[0])/(self.input_range[1]-self.input_range[0])
        for l in range(1,self.num_of_layers):
            self.v_values[l]=self.weights[l-1].dot(self.actvalues[l-1])
            self.actvalues[l]=self.act_f(self.v_values[l])
        return self.actvalues[self.num_of_layers-1]*(self.output_range[1]-self.output_range[0])+self.output_range[0]





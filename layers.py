'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        self.data = X@self.weights+self.biases # n x n2
        if self.activation == 'relu':
            return relu_of_X(self.data)
        elif self.activation == 'softmax': 
            return softmax_of_X(self.data)

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        if self.activation == 'relu':
            temp = gradient_relu_of_X(self.data, delta) # n x n2
            out = temp@self.weights.T # n x n1
            self.weights = self.weights - lr*(1.0/activation_prev.shape[0])*(activation_prev.T)@temp
            self.biases = self.biases - lr*(1.0/activation_prev.shape[0])*np.sum(temp, axis=0, keepdims=True)
            return out
        elif self.activation == 'softmax': 
            temp = gradient_softmax_of_X(self.data, delta)
            out = temp@self.weights.T
            self.weights = self.weights - lr*(1.0/activation_prev.shape[0])*(activation_prev.T@temp)
            self.biases = self.biases - lr*(1.0/activation_prev.shape[0])*np.sum(temp, axis=0, keepdims=True)
            return out
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO


        if self.activation == 'relu':
            X1 = np.expand_dims(X,axis=1) # n x 1 x d1 x r1 x c1
            W1 = np.expand_dims(self.weights, axis=0) # 1 x d2 x d1 x rf x cf
            out = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col)) # n x d2 x r2 x c2
            for col in range(0, self.in_col-self.filter_col+1, self.stride):
                for row in range(0,self.in_row-self.filter_row+1, self.stride):
                    out[:,:,int(row/self.stride), int(col/self.stride)] = np.expand_dims(self.biases,axis=0) + np.sum(X1[:,:,:,row:row+self.filter_row, col:col+self.filter_col]*W1,axis=(2,3,4))
            self.data = out
            return relu_of_X(out)

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        if self.activation == 'relu':
            inp_delta = gradient_relu_of_X(self.data, delta) # n x d2 x r2 x c2
            temp = np.expand_dims(inp_delta, axis=2) # n x d2 x 1 x r2 x c2
            out = np.zeros((self.data.shape[0] ,self.in_depth, self.in_row, self.in_col)) # n x d1 x r1 x c1
            W1 = np.expand_dims(self.weights, axis=0) # 1 x d2 x d1 x rf x cf
            X1 = np.expand_dims(activation_prev, axis=1) # n x 1 x d1 x r1 x c1
            dE_dF = np.zeros((self.out_depth, self.in_depth, self.filter_row, self.filter_col)) # d2 x d1 x rf x cf
            for col in range(0, self.in_col-self.filter_col+1, self.stride):
                for row in range(0,self.in_row-self.filter_row+1, self.stride):
                    dE_dF = dE_dF + (1.0/X1.shape[0])*np.sum(X1[:,:,:,row:row+self.filter_row, col:col+self.filter_col]*np.expand_dims(temp[:,:,:,int(row/self.stride),int(col/self.stride)],axis=(3,4)), axis=0)
                    out[:,:,row:row+self.filter_row,col:col+self.filter_col] = out[:,:,row:row+self.filter_row,col:col+self.filter_col] + np.sum(W1[:,:,:,0:self.filter_row, 0:self.filter_col]*np.expand_dims(temp[:,:,:,int(row/self.stride),int(col/self.stride)],axis=(3,4)),axis=1)
            dE_dB = np.sum(temp,axis=(0,2,3,4))*(1.0/X1.shape[0])
            self.weights = self.weights - lr*dE_dF
            self.biases = self.biases - lr*dE_dB
            return out
            
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        out = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col)) # n x d2 x r2 x c2
        for col in range(0, self.in_col-self.filter_col+1, self.stride):
            for row in range(0,self.in_row-self.filter_row+1, self.stride):
                out[:,:,int(row/self.stride), int(col/self.stride)] = np.mean(X[:,:,row:row+self.filter_row, col:col+self.filter_col],axis=(2,3))
        return out
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        out = np.zeros((delta.shape[0] ,self.in_depth, self.in_row, self.in_col)) # n x d1 x r1 x c1
        for col in range(0, self.in_col-self.filter_col+1, self.stride):
            for row in range(0,self.in_row-self.filter_row+1, self.stride):
                out[:,:,row:row+self.filter_row,col:col+self.filter_col] = out[:,:,row:row+self.filter_row,col:col+self.filter_col] + np.expand_dims(delta[:,:,int(row/self.stride),int(col/self.stride)],axis=(2,3))*(1.0/(self.filter_row*self.filter_col))
        return out
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        out = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col)) # n x d2 x r2 x c2
        for col in range(0, self.in_col-self.filter_col+1, self.stride):
            for row in range(0,self.in_row-self.filter_row+1, self.stride):
                out[:,:,int(row/self.stride), int(col/self.stride)] = np.amax(X[:,:,row:row+self.filter_row, col:col+self.filter_col],axis=(2,3))
        return out
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        #delta: n x d2 x r2 x c2
        out = np.zeros((delta.shape[0] ,self.in_depth, self.in_row, self.in_col)) # n x d1 x r1 x c1
        X1 = activation_prev # n x d1 x r1 x c1
        for col in range(0, self.in_col-self.filter_col+1, self.stride):
            for row in range(0,self.in_row-self.filter_row+1, self.stride):
                max_idx = X1[:,:,row:row+self.filter_row,col:col+self.filter_col].reshape((X1.shape[0],self.in_depth,-1)).argmax(-1) # n x d1
                max_indices = np.unravel_index(max_idx,(self.filter_row,self.filter_col)) # rf,cf
                ind_1 = np.expand_dims(np.arange(out.shape[0]),axis=1)
                ind_2 = np.expand_dims(np.arange(out.shape[1]),axis=0)
                ind_3 = max_indices[0]
                ind_4 = max_indices[1]
                out[:,:,row:row+self.filter_row,col:col+self.filter_col][ind_1,ind_2,ind_3,ind_4] = out[:,:,row:row+self.filter_row,col:col+self.filter_col][ind_1,ind_2,ind_3,ind_4] + delta[:,:,int(row/self.stride),int(col/self.stride)]
        return out
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        return X.reshape((X.shape[0],-1))

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(activation_prev.shape)
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    return np.maximum(X,0)
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    return (delta*(X>0))
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    return np.exp(-1*X)/np.sum(np.exp(-1*X),axis=1,keepdims=True)
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    act = softmax_of_X(X) # activations
    return (np.sum(delta*act,axis=1,keepdims=True)-delta)*act
    # END TODO

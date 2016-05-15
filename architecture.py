import json

import numpy as np

import theano
import theano.tensor as T

from layers.convpool import ConvPool
from layers.fc import FC
from layers.softmax import SoftMax

from utils.load import shared_dataset

class Architecture(object):

    def __init__ (self, specs):
        self.specs = specs
        # Number of layers in DeepNet
        nl = len(self.specs["layers"])

        # list would hold layers (ConvPool, FC, SoftMax etc.) 
        self.layer = [None] * nl
        
        # List holds the shape of output of a layer
        self.lout_shape = [None] * nl
        
        # INIT_FIRST_LAYER
        self.init_first_layer()
        
        # INIT_Subsequent_layer
        for i in range(1, nl):
            self.init_layer(i)


        self.cost = self.layer[nl-1].negative_log_likelihood(self.y)
        self.prob_y = self.layer[nl-1].prob_y
        self.pred_y = self.layer[nl-1].pred_y
        
        self.params = []
        for i in range(0, nl):
            self.params += self.layer[i].params

        self.batch_size = self.specs["meta"]["batch_size"]
        self.nl = nl

    def init_first_layer(self):
        
        layer_type = self.specs["layers"][0]["type"]
        input_shape = self.specs["meta"]["input_shape"]
        # Dimensions of X: (batch_size, channels, height, weight)
        self.X = T.tensor4('Input_Image')
        self.y = T.ivector('Target_Class')

        if layer_type == "convpool":
            n_filters = self.specs["layers"][0]["n_filters"]
            poolsize = self.specs["layers"][0]["poolsize"]
 
            self.layer[0] = ConvPool(self.X, 
                                    filter_shape = (n_filters,
                                                    input_shape[0],3,3),
                                    poolsize = poolsize)

            self.lout_shape[0] = (n_filters, (input_shape[1]-3+1)//2,
                                        (input_shape[2]-3+1)//2)
        elif layer_type == "fc":
            # For FC, input_shape should be of rasterized image
            units = self.specs["layers"][0]["units"]
            self.layer[0] = FC(self.X.flatten(2), 
                                fan_in = np.prod(input_shape),
                                fan_out = units)
            self.lout_shape[0] = (units,)

            
        elif layer_type == "softmax":
            # For SoftMax, input_shape should be of rasterized image
            units = self.specs["layers"][0]["units"]
            self.layer[0] = SoftMax(self.X.flatten(2), 
                                n_in = np.prod(input_shape),
                                n_out = units)
            self.lout_shape[0] = (units,)
        else:
            # TODO: Throw Exception
            print 'Invalid Layer Type'

        #print self.lout_shape[0]

    def init_layer(self, index):
        
        #json_index = "layer" + str(index)
        layer_type = self.specs["layers"][index]["type"]
        in_shape = self.lout_shape[index - 1]

        if layer_type == "convpool":
            n_filters = self.specs["layers"][index]["n_filters"]
            poolsize = self.specs["layers"][index]["poolsize"]
            
            self.layer[index] = ConvPool (self.layer[index - 1].output,
                                filter_shape =  (n_filters, in_shape[0], 3, 3),
                                poolsize =  poolsize)
            self.lout_shape[index] = (n_filters, (in_shape[1]-3+1)//2,
                                        (in_shape[2]-3+1)//2)
        
        elif layer_type == "fc":
            # If previous layer is ConvPool, need to flatter the input
            # Otherwise (in case of fc), we're good to go
            units = self.specs["layers"][index]["units"]
            self.layer[index] = FC(self.layer[index-1].output.flatten(2),
                                    fan_in = np.prod(in_shape), 
                                    fan_out = units)
            self.lout_shape[index] = (units,)
        
        elif layer_type == "softmax":
            units = self.specs["layers"][index]["units"]
            self.layer[index] = SoftMax(
                                self.layer[index-1].output.flatten(2),
                                n_in = np.prod(in_shape),
                                n_out = units)
            self.lout_shape[index] = (units,)
            
        #print self.lout_shape[index]

    def load (self, train, valid, test):
        """
            train : (X, y) 
            valid : (X, y)
            test  : (X, y)
            where X is a 4D numpy matrix and y is a numpy vector/python list
        """
        self.train_set_x, self.train_set_y = shared_dataset(train)
        self.valid_set_x, self.valid_set_y = shared_dataset(valid)
        self.test_set_x, self.test_set_y = shared_dataset(test)

        n_train = self.train_set_x.get_value(borrow=True).shape[0]
        n_valid = self.valid_set_x.get_value(borrow=True).shape[0]
        n_test = self.test_set_x.get_value(borrow=True).shape[0]

        self.n_train_batches = n_train // self.batch_size
        self.n_valid_batches = n_valid // self.batch_size
        self.n_test_batches = n_test // self.batch_size

    def get_train_func(self, index, learning_rate):
        '''
        get_train_func returns a theano func which optimizer could use to
        update parameters of the model
        '''
        grads = T.grad(self.cost, self.params)

        updates = [(param_i, param_i - learning_rate * grad_i)
                    for param_i, grad_i in zip(self.params, grads)]

        givens  = {self.X : self.train_set_x[index*self.batch_size :
                                            (index+1)*self.batch_size],
                    self.y : self.train_set_y[index*self.batch_size :
                                            (index+1)*self.batch_size]}
        return theano.function([index], self.cost, updates=updates,
                                givens=givens)

    def get_valid_func(self, index):
        givens = {self.X : self.valid_set_x[index*self.batch_size :
                                            (index+1)*self.batch_size],
                    self.y : self.valid_set_y[index*self.batch_size :
                                            (index+1)*self.batch_size]}
        return theano.function(
                [index],
                self.layer[self.nl-1].errors(self.y),
                givens=givens)

    def get_test_func(self, index):
        givens = {self.X : self.test_set_x[index*self.batch_size :
                                            (index+1)*self.batch_size],
                    self.y : self.test_set_y[index*self.batch_size :
                                            (index+1)*self.batch_size]}
        return theano.function(
                [index],
                self.layer[self.nl-1].errors(self.y),
                givens=givens)

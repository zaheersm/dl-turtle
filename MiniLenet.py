import numpy as np

import theano
import theano.tensor as T

from layers.convpool import ConvPool
from layers.fc import FC
from layers.softmax import SoftMax

from utils.load import shared_dataset


import six.moves.cPickle as pickle

class MiniLenet(object):
    
    def __init__ (self, image_shape, nfilters = [20, 50], batch_size = 500):
        
        '''
        image_shape : Tuple | ( no_channels, image_height, image_width)
        nfilters    : List  | [ no_filters_layer0, no_filters_layer1 ]
                    Where layer0 is the first ConvPool layer right after input
        batch_size: Scalar  | No of images to see at an instance for updating gradient
        '''     
        rng = np.random.RandomState(1234)
        # Dimensions of X: (batch_size, channels, height, width)
        self.X = T.tensor4('Input_Image')
    
        # Dimensions of y: (batch_size, ) 
        self.y = T.ivector('Target_Class')
        
        self.layer0 = ConvPool(self.X, input_shape = image_shape, 
                                filter_shape = (nfilters[0],
                                                image_shape[0], 5, 5),
                                poolsize = (2,2))
        
        l0_out_shape = (nfilters[0], (image_shape[1]-5+1)//2, 
                        (image_shape[2]-5+1)//2)
        
        self.layer1 = ConvPool(self.layer0.output,
                                input_shape = (batch_size, l0_out_shape[0],
                                                l0_out_shape[1], 
                                                l0_out_shape[2]),
                                filter_shape = (nfilters[1], nfilters[0], 5, 5),
                                poolsize = (2,2)) 
        
        l1_out_shape = (nfilters[1], (l0_out_shape[1]-5+1)//2, 
                        (l0_out_shape[2]-5+1)//2)
        
        self.layer2 = FC(self.layer1.output.flatten(2),
                        fan_in =  np.prod(l1_out_shape),
                        fan_out = 500)
        
        self.layer3 = SoftMax(self.layer2.output, 500, 10)
        
        self.cost = self.layer3.negative_log_likelihood(self.y)
        
        self.prob_y = self.layer3.prob_y
        self.pred_y = self.layer3.pred_y

        self.params = self.layer0.params + self.layer1.params + \
                        self.layer2.params + self.layer3.params

        self.batch_size = batch_size

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
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] // self.batch_size
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] // self.batch_size
            
if __name__ == '__main__':
    ml = MiniLenet((3,32,32),(10, 20), 500)

    train = pickle.load(open('processed_data/training_set.pkl', 'rb'))
    valid = pickle.load(open('processed_data/validation_set.pkl', 'rb'))
    test = pickle.load(open('processed_data/test_set.pkl', 'rb'))
    train = (train['trainX'], train['trainY'])
    valid = (valid['validX'], valid['validY'])
    test = (test['testX'], test['testY'])

    ml.load(train, valid, test)

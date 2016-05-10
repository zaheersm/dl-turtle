import theano
import theano.tensor as T

import numpy as np

'''
    FC represents a fully-connected layer
'''
class FC(object):
    
    def __init__(self, input, fan_in, fan_out):
        # Random number generator for initializing weight vectors 
        rng = np.random.RandomState(1234)
        high = np.sqrt (6./ (fan_in + fan_out))
        low = -high
        self.W = theano.shared(name = 'FC.W',
                                value = np.asarray(rng.uniform(low = low, 
                                                    high = high,
                                                    size = (fan_in, fan_out))),
                                borrow = True)
        self.b = theano.shared(name = 'FC.b',
                                value = np.zeros((fan_out,)),
                                borrow = True)
        
        self.output = T.tanh( T.dot(input, self.W) + self.b)
        
        self.params = [self.W, self.b]
        
        self.input = input

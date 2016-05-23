import numpy as np

import theano
import theano.tensor as T
'''
    FC represents a fully-connected layer
'''
class FC(object):
    
    def __init__(self, input, fan_in, fan_out):
        # Random number generator for initializing weight vectors 
        rng = np.random.RandomState(1234)
        W_bound = np.sqrt (6./ (fan_in + fan_out))
       
        W_init = np.array(rng.uniform( low = -W_bound,
                                        high = W_bound,
                                        size = (fan_in, fan_out)),
                            dtype = theano.config.floatX)

        self.W = theano.shared(name = 'FC.W',
                                value = W_init,
                                borrow = True)
        self.b = theano.shared(name = 'FC.b',
                                value = np.zeros((fan_out,), 
                                                dtype = theano.config.floatX),
                                borrow = True)
        
        self.output = T.tanh( T.dot(input, self.W) + self.b)
        
        self.params = [self.W, self.b]
        
        self.input = input

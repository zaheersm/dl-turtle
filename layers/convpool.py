import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

class ConvPool(object):
    
    def __init__ (self, input, input_shape, filter_shape, poolsize):
        
        # Initializing a random number generator for initializing 
        # weight filters
        
        rng = np.random.RandomState(1234)
        
        # fan_in : n_in_maps * filter_width * filter_height 
        fan_in = np.prod(filter_shape[1:])
    
        # fan_out : (n_out_maps * filter_width * filter_height)/poolsize
        fan_out = (filter_shape[0] *  np.prod(filter_shape[2:]) // \
                    np.prod(poolsize))

        W_bound = np.sqrt(6./ (fan_in + fan_out))

        W_init = rng.uniform(low = -W_bound, high = W_bound,
                            size = filter_shape)
        b_init = np.zeros((filter_shape[0],), dtype = np.float64)

        self.W = theano.shared(
                        name = 'Conv.W',
                        value = W_init,
                        borrow = True)
        self.b = theano.shared(
                        name = 'Conv.b',
                        value = b_init,
                        borrow = True)

        convolution = conv2d(input, self.W)
        
        # Adding bias term to convolved output         
        convolution = convolution + self.b.dimshuffle('x', 0, 'x', 'x')

        pool = downsample.max_pool_2d(input = convolution,
                                            ds=poolsize,
                                            ignore_border=True)
        self.output = T.tanh(pool)
        
        self.params = [self.W, self.b]
        
        self.input = input

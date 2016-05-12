import numpy as np

import theano
import theano.tensor as T

def shared_dataset(data, borrow = True):
    """ Loads data into a shared_variable
    Data : Tuple of numpy arrays that is (X, y)
         : Where X is a 4D matrix (no_images, no_channels, height, width)
            and y is a list of labels (integers)
    """
    #abc = np.asarray(data[0], dtype = theano.config.floatX)
    shared_x = theano.shared(np.asarray(data[0], 
                                        dtype=theano.config.floatX),
                            borrow=True)

    shared_y = theano.shared(np.asarray(data[1],
                                        dtype=theano.config.floatX),
                            borrow=True)
    
    """
    Data is stored on GPU as floats therefore, class labels are also stored
    as floats. However, during our computation, we need
    them as Integers therefore casting into integer before returing
    """
    return shared_x, T.cast(shared_y, 'int32')

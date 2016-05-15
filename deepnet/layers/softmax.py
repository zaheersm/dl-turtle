import numpy as np

import theano
import theano.tensor as T

class SoftMax(object):

    def __init__(self, input, n_in, n_out):
        
        # Init Weights and bias
        self.W = theano.shared(
                        name = 'SoftMax.W',
                        value = np.zeros((n_in, n_out), dtype= np.float64),
                        borrow = True)
        
        self.b = theano.shared(
                        name = 'SoftMax.b',
                        value = np.zeros((n_out,), dtype = np.float64),
                        borrow = True)
        
        # Probability distribution over n_out classes
        self.prob_y = T.nnet.softmax((T.dot(input, self.W) + self.b))
        
        # Prediction | Class with maximum probability 
        self.pred_y = T.argmax(self.prob_y, axis = 1)
        
        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        log_probs = T.log(self.prob_y)
        # Indexing the log_probs of true label
        log_probs = log_probs[T.arange(y.shape[0]), y]
        return -T.mean(log_probs)

    def errors(self, y):
        return T.mean(T.neq(self.pred_y, y))

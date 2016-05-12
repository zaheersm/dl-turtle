"""
DISCLAIMER:
train method below is taken from deeplearning.net tutorial for early stopping 
[http://deeplearning.net/tutorial/gettingstarted.html#early-stopping]

It has been slightly rebuffed inorder to integrate it with other parts of this
project
"""

from __future__ import print_function

import sys
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np
import theano.tensor as T

def train(model, learning_rate = 0.1, n_epochs = 200, 
            persist_name = 'model_params.pkl'):
        # Symbolic variable to represent the index of minibatch
        index = T.lscalar()
        test_model = model.get_test_func(index)
        validate_model = model.get_valid_func(index)
        train_model = model.get_train_func(index, learning_rate)
        print('... training')
        # early-stopping parameters
        patience = 1000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(self.n_train_batches, patience // 2)
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(model.n_train_batches):
                iter = (epoch - 1) * model.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)
                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(model.n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, model.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
            
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(model.n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, 
                                model.n_train_batches,test_score * 100.))
                        # Save Model
                        with open(persist_name, 'wb') as f:
                            pickle.dump(model.params, f)

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file  ran for %.2fm' 
                % ((end_time - start_time) / 60.)))

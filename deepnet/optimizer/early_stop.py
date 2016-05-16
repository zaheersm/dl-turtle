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
import json

import numpy as np
import theano.tensor as T

from PIL import Image
import base64
import cStringIO

def get_top_three(data):
    n = data.shape[0]

    values = np.zeros((n,3), dtype = np.float64)
    indices = np.zeros((n,3), dtype = np.int64)

    for i in range(3):
        indices[:, i] = np.argmax(data, axis = 1)
        values[:, i] = np.max(data, axis = 1)
        data[np.arange(n), indices[:, i]] = -1
    return (values, indices)

def train(handler, model, learning_rate = 0.1, n_epochs = 200, 
            persist_name = 'model_params.pkl'):
        # Symbolic variable to represent the index of minibatch
        index = T.lscalar()
        test_model = model.get_test_func(index)
        validate_model = model.get_valid_func(index)
        train_model = model.get_train_func(index, learning_rate)
        indices = T.ivector()
        samples_prob = model.get_samples_prob(indices)
        test_size = model.test_set_x.get_value().shape[0]
        print('... training')
        # early-stopping parameters
        patience = 1000
        patience_increase = 2
        improvement_threshold = 0.995
        #validation_frequency = min(model.n_train_batches, patience // 2)
        validation_frequency = 5
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(model.n_train_batches):
                if handler.train == False:
                    return
                
                iter = (epoch - 1) * model.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)
                if (iter + 1) % validation_frequency == 0:
                    rindices = np.array(np.random.randint(0, test_size, 3),
                                        dtype=np.int32)
                    max_probs, max_labels = get_top_three(
                                                samples_prob(rindices))
                    sample_images = model.test_set_x.get_value()[rindices]
                    sample_images = sample_images.reshape(len(rindices), 28, 28)
                    image_ary = []
                    for i in range(len(sample_images)):
                        image = np.uint8(sample_images[i]*255)
                        _buffer = cStringIO.StringIO()
                        Image.fromarray(image).convert('L').save(_buffer,
                                                            format = 'JPEG')
                        image_str = 'data:image/jpeg;base64,' + \
                                    base64.b64encode(_buffer.getvalue())
                        image_ary += [image_str]
                    info = {"images":image_ary,
                            "probs":max_probs.tolist(),
                            "labels":max_labels.tolist(),
                            "iteration": cost_ij.item()}
                    info_json = json.dumps(info)
                    handler.client.send(info_json)
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

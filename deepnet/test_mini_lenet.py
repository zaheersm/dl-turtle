"""
Sample file to setup MiniLenet and train it

Assuming you've dataset in appropriate format within processed_data/
"""

import six.moves.cPickle as pickle

import theano
import theano.tensor as T

from mini_lenet import MiniLenet
from optimizer.early_stop import train

if __name__ == '__main__':
    ml = MiniLenet((3,32,32),(10, 20), 500)

    train_set = pickle.load(open('processed_data/training_set.pkl', 'rb'))
    valid_set = pickle.load(open('processed_data/validation_set.pkl', 'rb'))
    test_set = pickle.load(open('processed_data/test_set.pkl', 'rb'))
    train_set = (train_set['trainX'], train_set['trainY'])
    valid_set = (valid_set['validX'], valid_set['validY'])
    test_set = (test_set['testX'], test_set['testY'])

    ml.load(train_set, valid_set, test_set)
    train(ml, 0.1, persist_name='cifar_params.pkl')

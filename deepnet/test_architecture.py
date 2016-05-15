'''
An example of using generic architectures by specifying them
in JSON format
'''
import six.moves.cPickle as pickle
import json

from optimizer.early_stop import train
from architecture import Architecture

if __name__ == '__main__':
    
    ''' 
    # Specs for Logisitic Regression [SoftMax]
    specs = '{"meta": {"dataset":"MNIST", "batch_size":20, ' + \
            '"input_shape":[1,28,28]}, ' + \
            '"layers":[{"type":"softmax", "units":10}]}'
    # Specs for MLP : FC -> Softmax
    specs = '{"meta": {"dataset":"MNIST", "batch_size":20, ' + \
            '"input_shape":[1,28,28]}, ' + \
            '"layers":[{"type":"fc", "units":500},'+\
            '{"type":"softmax", "units": 10}]}'
    '''
    # Specs for ConvNet: ConvPool -> ConvPool -> FC -> SoftMax
    specs = '{"meta": {"dataset":"MNIST", "batch_size":500, ' + \
            '"input_shape":[1,28,28]}, ' + \
            '"layers":[{"type":"convpool", "n_filters":30, ' + \
            '"poolsize":[2,2]}, ' + \
            '{"type":"convpool", "n_filters":75, '+ \
            '"poolsize":[2,2]}, ' + \
            '{"type":"fc", "units":500}, ' + \
            '{"type":"softmax", "units": 10}]}'

    specs = json.loads(specs) 
    net = Architecture(specs)
    

    dataset = specs["meta"]["dataset"]
    train_set = pickle.load(open(dataset + '/training_set.pkl', 'rb'))
    valid_set = pickle.load(open(dataset + '/validation_set.pkl', 'rb'))
    test_set = pickle.load(open(dataset + '/test_set.pkl', 'rb'))
    
    train_set = (train_set['trainX'], train_set['trainY'])
    valid_set = (valid_set['validX'], valid_set['validY'])
    test_set = (test_set['testX'], test_set['testY'])

    net.load(train_set, valid_set, test_set)
    train(net, 0.1)

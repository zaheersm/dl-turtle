import json
import cPickle as pickle

from architecture import Architecture
from optimizer.early_stop import train

class Handler:
    
    def __init__(self, client):
        #set client_connection to currently connected node
        self.client_connection = client
        #initialize temporary cache
        self.cache = None
        #initialize Convolution Neural Network
        self.model = None
    
    def handle_request(self, req):
    #if client is sending some data add it to handler cache and 
    #await next command
        if(req[0:4] == 'data'):
            if(self.cache == None):
                self.cache = req[4:]
            else:
                self.cache += req[4:]
          
        else:
            #delegating request to mapped methods
            self.switcher(req)
    
    def switcher(self, command):
        return {
            'start' : self.start,
            'stop' : self.stop,
            'load weights' : self.load_weights,
            'save_weights' : self.save_weights
        }[command]()
    
    def start(self):
        train(self.model, 0.1)
      
    def stop(self):
        #self.model.stop_training()
        a = 1

    def load_weights(self, weights = None):
        #self.model.load_weights(weights)
        a = 1
    
    def save_weights(self):
        #self.client_connection.send(self.serialize(self.model.save_weights()))
        a = 1
    
    def serialize(self, data):
        #to be implemented
        a = 1
    
    def create_model(self, specs):
        specs = json.loads(specs)
        self.model = Architecture(specs)
        dataset = specs["meta"]["dataset"]

        train_set = pickle.load(open(dataset + '/training_set.pkl', 'rb'))
        valid_set = pickle.load(open(dataset + '/validation_set.pkl', 'rb'))
        test_set = pickle.load(open(dataset + '/test_set.pkl', 'rb'))

        train_set = (train_set['trainX'], train_set['trainY'])
        valid_set = (valid_set['validX'], valid_set['validY'])
        test_set = (test_set['testX'], test_set['testY'])

        self.model.load(train_set, valid_set, test_set)

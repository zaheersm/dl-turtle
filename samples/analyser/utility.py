from models import *

class Handler:

    def __init__(self):
		
		#initialize temporary cache
        self.cache = None
		#initialize Convolution Neural Network
        #self.model = new CNN() //constructor needs datasets to initialize

    def handler(self, req):
		
		#if client is sending some data add it to handler cache and await next command
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
            }[command](...)

    def start(self):
        self.model.begin_training()
        
    def stop(self):
        self.model.stop_training()

    def load_weights(self, weights = None):
        self.model.load_weights(weights)

    def save_weights(self):
        return self.serialize(self.model.save_weights())

    def serialize(self, data):
        #to be implemented
		
	def deserialize(self, stream):
		#to be implemented

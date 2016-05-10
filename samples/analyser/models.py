import time
import numpy as np

class CNN:
	def __init__(self, client, train_set = None, validation_set = None, test_set = None):
		#initialize convolution neural network with data
		
		self.train_set = train_set
		self.validation_set = validation_set
		self.test_set = test_set
		
		#set initial state to set
		self.state = 'SET'
		
		#set client_connection to connected node
		self.client_connection = client
	def begin_training(self):
		for i in range(1000):
			time.sleep(0.05)
			self.client_connection.send('iteration ' + str(int(np.random.rand() * 5000)))
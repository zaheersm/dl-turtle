class CNN:

    def __init__(self, train_set, validation_set, test_set, client):
		#initialize convolution neural network with data
		
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        
		#set initial state to set
        self.state = 'SET'
		
		#set client to connected node
		self.client = client
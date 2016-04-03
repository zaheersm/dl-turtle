#!/usr/bin/env python
import math

import graphlab
import numpy as np
import theano
import theano.tensor as T

sales = graphlab.SFrame('kc_house_data.gl/')
N = sales.shape[0]

# Features to be used:
#    sqft_living, floors, bedrooms and bathrooms

sqft_living = sales.select_column('sqft_living').to_numpy().reshape(N,1)
floors = sales.select_column('floors').to_numpy().reshape(N,1).astype(float)
bedrooms = sales.select_column('bedrooms').to_numpy().reshape(N,1)
bathrooms = sales.select_column('bathrooms').to_numpy().reshape(N,1)

# Target Output: price
price = sales.select_column('price').to_numpy()

bias = np.ones((N,1),dtype = 'float64')

# Concatenating bias with feature vectors to build feature matrix
features = np.concatenate((bias, sqft_living, floors, bedrooms, bathrooms), axis = 1)

# Initializng weight vectors
init_weights = np.array([-47000.0, 1.0, 1.0, 1.0, 1.0])
w = theano.shared(init_weights, 'w')

# Theano speicific symbolic equations
X = T.dmatrix('X')
y = T.dvector('y')

prediction = T.dot(X, w)
cost = T.sum(T.pow(y - prediction, 2))
grad = T.grad(cost, w)

learning_rate = 7e-12
train = theano.function([X, y], outputs = [cost, grad], updates = [(w, w - learning_rate * grad)])
test = theano.function([X], prediction)

# Training
cost = 0
grad = []

# Training | 10000 iterations (Room for improvement)
for i in range(10000):
    cost, grad = train(features, price)

print 'Weight Vector after training (10000 iterations): '    
print w.get_value()
print 'Cost after training: '
print cost


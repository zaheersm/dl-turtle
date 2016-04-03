#!/usr/bin/env python

import math

import graphlab
import numpy as np
import theano
import theano.tensor as T

sales = graphlab.SFrame('kc_house_data.gl/')

X_val = sales.select_column('sqft_living').to_numpy()
y_val = sales.select_column('price').to_numpy()

# Initializing slope with 1 and intercept with -47000
m = theano.shared(1.0, 'm')
c = theano.shared(-47000.0, 'c')
X = T.dvector('X')
y = T.dvector('y')

N = X_val.shape[0]

prediction = T.dot(X,m) + c 
cost = T.sum(T.pow(y - prediction, 2))
gradm = T.grad(cost, m)
gradc = T.grad(cost, c)

learning_rate = 7e-12
converged = False
tolerance = 2.5e7

train = theano.function([X,y], outputs = [cost, gradm, gradc], 
                               updates = [(m, m - learning_rate * gradm), 
                                          (c, c - learning_rate * gradc)])
test = theano.function([X], prediction)

print 'Initial Weights: '
print m.get_value(), ' ', c.get_value()

while not converged:
    
    cost, grad_m, grad_c = train(X_val, y_val)
    
    gradient_sum_squares = (grad_m * grad_m) + (grad_c*grad_c)
    gradient_magnitude = math.sqrt(gradient_sum_squares)
    
    if (gradient_magnitude < tolerance):
        converged = True
print 'Weights of the fitted line: '
print 'Slope: ', m.get_value(), ' Intercept: ', c.get_value() 


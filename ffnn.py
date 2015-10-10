#!/usr/local/bin/python3
#
# This program calculates the output of a two layer feed-forward
# neural network given the input pattern x, and weight matrices W1 and W2
#
# Author: Olafur Jon Thoroddsen (olafurjt13@ru.is)


# ----- Begin Importing modules ----- #

import numpy as np

# ----- End Importing modules ----- #


# ----- Begin Function Definitions ----- #

def sigmoid(a):
	try:
		tmp = [1/(1 + np.exp(-b)) for b in a]
		return np.array(tmp)
	except TypeError:
		return 1/(1+np.exp(-a))

def addOnes(X):
	try:
		return np.insert(X,0,1,axis=1)
	except IndexError:
		return np.insert(X,0,1)

def ffnn(x,W1,W2):
	# Adding a row of ones to the input to simplify calculations:
	x = addOnes(x)

	# Calculating the hidden layer inputs a1:
	a1 = np.dot(W1,x)

	# Calculating the hidden layer outputs z1:
	z1 = sigmoid(a1)

	# Calculating the output layer inputs a2:
	a2 = np.dot(W2,addOnes(z1))

	# Calculating the output of the network y:
	y = sigmoid(a2)

	return [y,z1,a1]


# ----- End Function Definitions ----- #



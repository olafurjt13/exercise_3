#!/usr/local/bin/python3
#
# This program calculates the output of a two layer neural network
# given the input pattern x, and weight matrices W1 and W2
#
# Author: Ólafur Jón Thoroddsen (olafurjt13@ru.is)


# ----- Begin Importing modules ----- #

import numpy as np
import sys

# ----- End Importing modules ----- #

USAGE = """USAGE: ffnn.py "input pattern" "weight matrix 1" "weight matrix 2"
	Example: ffnn.py x W1 W2
	Where x is a Dx1 dimensional vector
	W1 is Dx(Mh+1) dimensional matrix
	W2 is (Mh+1)xMo dimensional matrix
"""

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

# ----- End Function Definitions ----- #


if len(sys.argv) != 4:
	print(USAGE)
	exit(1)

script, x_input, W1_input, W2_input = sys.argv

x = np.loadtxt(x_input)
W1 = np.loadtxt(W1_input)
W2 = np.loadtxt(W2_input)

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

print(y)

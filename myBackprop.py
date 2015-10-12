#!/usr/local/bin/python3
#
# This program calculates the gradient of the cross-entropy
# error function for a two layer nerual network given an 
# input pattern x, target vector t and weight matrices W1 and W2
#
# Author: Olafur Jon Thoroddsen


# ----- Begin Importing modules ----- #

import numpy as np
from ffnn import ffnn, sigmoid

# ----- End Importing modules ----- #


# ----- Begin Function Definitions ----- #

def outputDeltas(t,y):
	return np.array(y - t)

def networkDeltas(deltaks,ays,W):

	a = np.matrix([])
	i = 0
	for values in ays:
		a = np.insert(a,i,sigmoid(values))
		i += 1
	a = a.T

	W = np.delete(W,0)
	deltajs = np.matrix([])
	i = 0
	for value in a:
		deltajs = np.insert(deltajs,i,(sigmoid(value) * (1 - sigmoid(value)) * np.dot(W.T[i],deltaks)))
		i += 1
	deltajs = deltajs.T
	return np.matrix(deltajs)

def myBackprop(x,t,W1,W2):

	y,z,a = ffnn(x,W1,W2)
	# y: Network output
	# z: hidden layer output
	# a: hidden layer input

	# Compute output Deltas
	deltak = outputDeltas(t,y)

	# Computing network Deltas
	deltaj = networkDeltas(deltak,a,W2)

	# Computing the 
	dEn_dw1 = deltaj * np.insert(x,0,1)
	dEn_dw2 = deltak * np.insert(z,0,1)

	return [dEn_dw1, dEn_dw2, y]
# ----- End Function Definitions ----- #

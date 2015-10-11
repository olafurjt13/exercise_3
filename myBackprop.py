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
	a = np.array([sigmoid(value) for value in ays])
	W = np.delete(W,0)
	deltajs = [sigmoid(a[i]) * (1 - sigmoid(a[i])) * np.dot(W.T[i],deltaks) for i in range(len(a))]

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

	print("delta_k: " + str(deltak))
	print("delta_j: " + str(deltaj))

	dEn_dw1 = deltaj.T * np.matrix(np.insert(x,0,1).T)
	dEn_dw2 = np.matrix(deltak.T) * np.matrix(np.insert(z,0,1).T)

	return [dEn_dw1, dEn_dw2, y]
# ----- End Function Definitions ----- #

#!/usr/local/bin/python3
#
# This program calculates the gradient of the cross-entropy
# error function for a two layer nerual network given an 
# input pattern x and weight matrices W1 and W2
# 
# It is assumed that there is only one output variable y
#
# Author: Olafur Jon Thoroddsen


# ----- Begin Importing modules ----- #

import numpy as np
import sys
from ffnn import ffnn, sigmoid

# ----- End Importing modules ----- #


# ----- Begin Function Definitions ----- #

def outputDeltas(t,y):
	return np.array(y - t)

def networkDeltas(deltaks,ays,W):
	a = np.array([sigmoid(value) for value in ays])
	W = np.delete(W,0)
	deltajs = [sigmoid(a[i]) * (1 - sigmoid(a[i])) * np.dot(W.T[i],deltaks) for i in range(len(a))]
	
	return deltajs

def myBackprop(x,t,W1,W2):

	y,z,a = ffnn(x,W1,W2)	# y: Network output
							# z: hidden layer output
							# a: hidden layer input


	# Compute output Deltas
	deltak = outputDeltas(t,y)
	print("Delta k's: " + str(deltak))

	# Computing network Deltas
	deltaj2 = networkDeltas(deltak,a,W2)
	print("Delta j's: " + str(deltaj2))
	
	


# ----- End Function Definitions ----- #

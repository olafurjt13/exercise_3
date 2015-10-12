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
	#print("outputDeltas t: " + str(np.shape(t)))
	#print("outputDeltas y: " + str(np.shape(y)))
	return np.array(y - t)

def networkDeltas(deltaks,ays,W):
	#print("networkDeltas deltaks: " + str(np.shape(deltaks)))
	#print("networkDeltas ays: " + str(np.shape(ays)))
	#print("networkDeltas W: " + str(np.shape(W)))

	a = np.matrix([])
	i = 0
	for values in ays:
		a = np.insert(a,i,sigmoid(values))
		i += 1
	a = a.T
	#print("networkDeltas a(shape): " + str(np.shape(a)))
	#print("networkDeltas a(value): " + str(a))
	#print("networkDeltas a(type):  " + str(type(a)))

	#print("networkDeltas W(value): " + str(W))
	W = np.delete(W,0)
	#print("networkDeltas W(value): " + str(W))
	#print("networkDeltas W(shape): " + str(np.shape(W)))
	deltajs = np.matrix([])
	i = 0
	for value in a:
		#print("networkDeltas a(j): " + str(value))
		deltajs = np.insert(deltajs,i,(sigmoid(value) * (1 - sigmoid(value)) * np.dot(W.T[i],deltaks)))
		i += 1
	deltajs = deltajs.T
	#print("networkDeltas deltajs: " + str(np.shape(deltajs)))
	#deltajs = [sigmoid(a[0,i]) * (1 - sigmoid(a[0,i])) * np.dot(W.T[i],deltaks) for i in range(len(a))]
	return np.matrix(deltajs)

def myBackprop(x,t,W1,W2):

	#print("myBackprop x(shape): " + str(np.shape(x)))
	#print("myBackprop x(type): " + str(type(x)))
	#print("myBackprop t(shape): " + str(np.shape(t)))
	#print("myBackprop t(type): " + str(type(t)))
	#print("myBackprop W1(shape): " + str(np.shape(W1)))
	#print("myBackprop W1(type): " + str(type(W1)))
	#print("myBackprop W2(shape): " + str(np.shape(W2)))
	#print("myBackprop W2(type): " + str(type(W2)))

	y,z,a = ffnn(x,W1,W2)
	#print("finished ffnn successfully")
	# y: Network output
	# z: hidden layer output
	# a: hidden layer input

	# Compute output Deltas
	#print("Entering outputDeltas")
	deltak = outputDeltas(t,y)
	#print("Finished calculating outputDeltas")

	# Computing network Deltas
	#print("Entering networkDeltas")
	deltaj = networkDeltas(deltak,a,W2)
	#print("Finished calculating networkDeltas")

	#print("myBackprop T(type): " + str(type(x)))
	#print("myBackprop T(shape): " + str(np.shape(x)))

	#print("myBackprop deltaj.T x np.inert(x,0,1).T (shapes): " + str(np.shape(deltaj)) + " x " + str(np.shape(np.insert(x,0,1))))
	#print("myBackprop np.insert(deltaj.T,0,deltaj[0,0]) x np.inert(x,0,1).T (values): " + str(deltaj) + " x " + str(np.insert(x,0,1)))

	dEn_dw1 = deltaj * np.insert(x,0,1)
	dEn_dw2 = deltak * np.insert(z,0,1)

	return [dEn_dw1, dEn_dw2, y]
# ----- End Function Definitions ----- #

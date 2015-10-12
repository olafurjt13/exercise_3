#!/usr/local/bin/python3
#
# This program trains a feed-forward neural network using gradient decent.
# It takes as it's inputs the training data X, the targets T, initial weight
# matrices W1i and W2i, number of iterations Niter and the learning rate eta.
#
# Author: Olafur Jon Thoroddsen

import numpy as np
import numpy.linalg as LA
from myBackprop import myBackprop

def trainNN(X,T,W1i,W2i,Niter,eta):
	trainData = np.concatenate((X,T),axis=1)
	W1 = W1i
	W2 = W2i
	print("Orignal weight matrices: " + str(W1) + '\n' + str(W2))
	for i in range(Niter):
		for data in trainData:
			dEn_w1, dEn_w2, y = myBackprop(data[0,0:2].T,np.matrix(data[0,2]),W1,W2)
			W1 = W1 - eta * dEn_w1
			W2 = W2 - eta * dEn_w2

	print("Weight matrices after " + str(Niter) + " iterations of training: " + str(W1) + '\n' + str(W2))

	return [W1, W2]
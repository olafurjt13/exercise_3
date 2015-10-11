#!/usr/local/bin/python3
#
# This program trains a feed-forward neural network using gradient decent.
# It takes as it's inputs the training data X, the targets T, initial weight
# matrices W1i and W2i, number of iterations Niter and the learning rate eta.
#
# Author: Olafur Jon Thoroddsen

import numpy as np
from myBackprop import myBackprop
from ffnn import ffnn


def trainNN(X,T,W1i,W2i,Niter,eta):
	
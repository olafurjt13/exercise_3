#!/usr/local/bin/python3
#
# This program calculates the gradient of the cross-entropy
# error function for a two layer nerual network given an 
# input pattern x and weight matrices W1 and W2
# 
# Author: Olafur Jon Thoroddsen


# ----- Begin Importing modules ----- #

import numpy as np
import sys
from ffnn, sigmoid import ffnn

# ----- End Importing modules ----- #

def myBackprop(x,t,W1,W2):
	y,z,a = ffnn(x,W1,W2)

	if (np.size(t) > 1):
		print("Target vector larger then 1")
		i = 0
		deltak = []
		for ts in t:
			dektak[i] = y[i] - ts
			i += 1
	else:
		deltak = y - t
		print("Target vector is a single number")


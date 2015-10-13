#!/usr/local/bin/python3
#
# ----- Begin Importing modules ----- #

import numpy as np
import sys
from ffnn import ffnn
from myBackprop import myBackprop

# ----- End Importing modules ----- #


def openData(Nx2_datafile):
	array = ([item.strip('\n').split(' ') for item in Nx2_datafile])
	for item in array:
		item[0] = float(item[0])
		item[1] = float(item[1])
	return np.matrix(array)


USAGE = """USAGE: tester.py "input pattern" "weight matrix 1" "weight matrix 2"
	Example: tester.py x W1 W2
	Where x is a Dx1 dimensional vector
	W1 is Dx(Mh+1) dimensional matrix
	W2 is (Mh+1)xMo dimensional matrix
"""


if len(sys.argv) != 4:
	#print(USAGE)
	exit(1)

script, x_input, W1_input, W2_input = sys.argv

x = np.matrix(np.loadtxt(x_input))
W1 = np.matrix(np.loadtxt(W1_input))
W2 = np.matrix(np.loadtxt(W2_input))

y,z,a = ffnn(x.T,W1,W2)

print("Network output: " + str(y) +'\n'+ "Hidden layer input: " + str(a) + '\n'+"Hidden layer output: " + str(z))
print("This configuration has " + str(np.shape(W2)[1]) + " neurons in the hidden layer")

t = 1	# Synthesizing the target value so that we get some error when running the script :)

dEn_w1, dEn_w2, y = myBackprop(x.T,t,W1,W2)


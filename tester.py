#!/usr/local/bin/python3
#
# ----- Begin Importing modules ----- #

import numpy as np
import sys
from ffnn import ffnn

# ----- End Importing modules ----- #


USAGE = """USAGE: tester.py "input pattern" "weight matrix 1" "weight matrix 2"
	Example: tester.py x W1 W2
	Where x is a Dx1 dimensional vector
	W1 is Dx(Mh+1) dimensional matrix
	W2 is (Mh+1)xMo dimensional matrix
"""


if len(sys.argv) != 4:
	print(USAGE)
	exit(1)

script, x_input, W1_input, W2_input = sys.argv

x = np.loadtxt(x_input)
W1 = np.loadtxt(W1_input)
W2 = np.loadtxt(W2_input)

y,z,a = ffnn(x,W1,W2)

print("Network output: " + str(y) +'\n'+ "Hidden layer input: " + str(a) + '\n'+"Hidden layer output: " + str(z))
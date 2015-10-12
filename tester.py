#!/usr/local/bin/python3
#
# ----- Begin Importing modules ----- #

import numpy as np
import sys
from ffnn import ffnn
from myBackprop import myBackprop
from trainNN import trainNN

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

x = np.loadtxt(x_input)
W1 = np.loadtxt(W1_input)
W2 = np.loadtxt(W2_input)

#y,z,a = ffnn(x,W1,W2)

##print("Network output: " + str(y) +'\n'+ "Hidden layer input: " + str(a) + '\n'+"Hidden layer output: " + str(z))
##print("This configuration has " + str(len(W2)-1) + " neurons in the hidden layer")

t = 1	# Synthesizing the target value so that we get some error when running the script :)

#dEn_w1, dEn_w2, y = myBackprop(x,t,W1,W2)

##print("dEn_w1: " + str(dEn_w1))
##print("dEn_w2: " + str(dEn_w2))


x1_data = open('x1_data.txt','r')
x2_data = open('x2_data.txt','r')

x1 = openData(x1_data)
x2 = openData(x2_data)
X = np.matrix(np.concatenate((x1,x2),axis=0))
T = np.matrix(np.concatenate((np.ones(np.size(x1,axis=0)),np.zeros(np.size(x2,axis=0))),axis=0)).T
W1i = np.matrix(np.random.rand(np.shape(np.matrix(W1))[0],np.shape(np.matrix(W1))[1]))
W2i = np.matrix(np.random.rand(np.shape(np.matrix(W2))[0],np.shape(np.matrix(W2))[1]))


Niter = 100
eta = 0.01
#print("X: " + str(np.shape(X)))
#print("T: " + str(np.shape(T)))
#print("W1i: " + str(np.shape(W1i)))
#print("W2i: " + str(np.shape(W2i)))

W1, W2 = trainNN(X,T,W1i,W2i,Niter,eta)


















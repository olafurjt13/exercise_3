#!/usr/local/bin/python3
#
# ----- Begin Importing modules ----- #

import numpy as np
import sys
from ffnn import ffnn
from myBackprop import myBackprop
from trainNN import trainNN
import matplotlib.pyplot as plt

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

#dEn_w1, dEn_w2, y = myBackprop(x.T,t,W1,W2)



x1_data = open('x1_data.txt','r')
x2_data = open('x2_data.txt','r')

x1 = openData(x1_data)
x2 = openData(x2_data)
X = np.matrix(np.concatenate((x1,x2),axis=0))
T = np.matrix(np.concatenate((np.ones(np.size(x1,axis=0)),np.zeros(np.size(x2,axis=0))),axis=0)).T
W1i = np.matrix(np.random.rand(np.shape(np.matrix(W1))[0],np.shape(np.matrix(W1))[1]))
W2i = np.matrix(np.random.rand(np.shape(np.matrix(W2))[0],np.shape(np.matrix(W2))[1]))

#W1i = np.matrix(np.zeros([2,3]))
#W2i = np.matrix(np.zeros([1,3]))

outputfile = open('Training optimization data.txt','w')

outputfile.write(str(W1i))
outputfile.write(str(W2i))
outputfile.write('Niter eta Training_MCR_C1 Training_MCR_C2 Test_MCR_C1 Test_MCR_2\n')
for Niter in [5,10,15,20,25,30,35]:
	for eta in [0.0001,0.0005,0.001,0.0015,0.002,0.0025,0.0030]:
		#Niter = 25
		#eta = 0.0020


		W1_trained, W2_trained = trainNN(X,T,W1i,W2i,Niter,eta)
		#W1file = open('W1_trained.txt','w')
		#W2file = open('W2_trained.txt','w')
		#W1file.write(str(W1).strip('[]'))
		#W2file.write(str(W2).strip('[]'))


		# Evaluating the training error

		y_training_class1=[]
		for exes in x1:
			y,z,a = ffnn(exes.T,W1_trained,W2_trained)
			y_training_class1.append(y[0,0])

		y_training_class2=[]
		for exes in x2:
			y,z,a = ffnn(exes.T,W1_trained,W2_trained)
			y_training_class2.append(y[0,0])

		MCR_1 = 0
		for value in y_training_class1:
			if value < 0.5:
				MCR_1 += 1

		MCR_2 = 0
		for value in y_training_class2:
			if value > 0.5:
				MCR_2 += 1

		print("Testing MCR:")
		print("Class 1: " + str(MCR_1) + '\n' + "Class 2: " + str(MCR_2))


		# Evaluating test error:
		x1_1000_data = open('x1_1000.txt')
		x2_1000_data = open('x2_1000.txt')

		x1_1000 = openData(x1_1000_data)
		x2_1000 = openData(x2_1000_data)


		classX1eval = []
		classX2eval = []
		for exes in x1_1000:
			y,z,a = ffnn(exes.T,W1_trained,W2_trained)
			classX1eval.append(y[0,0])

		for exes in x2_1000:
			y,z,a = ffnn(exes.T,W1_trained,W2_trained)
			classX2eval.append(y[0,0])

		misclassification_1 = 0
		for value in classX1eval:
			if value < 0.5:
				misclassification_1 += 1

		misclassification_2 = 0
		for value in classX2eval:
			if value > 0.5:
				misclassification_2 += 1

		print("Training MCR:")
		print("Class 1: " + str(misclassification_1) + '\n' + "Class 2: " + str(misclassification_2))

		outputfile.write(str(Niter) + ' ' + str(eta) + ' ' + str(MCR_1) + ' ' + str(MCR_2) + ' ' + str(misclassification_1) + ' ' + str(misclassification_2) + '\n')
		# plt.plot(x1_1000[:,0].T,x1_1000[:,1].T,'bo',alpha=0.5)
		# plt.plot(x2_1000[:,0].T,x2_1000[:,1].T,'ro',alpha=0.5)
		# plt.grid()
		# plt.show()



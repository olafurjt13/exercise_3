#!/usr/local/bin/python3
#
# ----- Begin Importing modules ----- #

import numpy as np
from ffnn import ffnn
from trainNN import trainNN
import matplotlib.pyplot as plt

# ----- End Importing modules ----- #


def openData(Nx2_datafile):
	array = ([item.strip('\n').split(' ') for item in Nx2_datafile])
	for item in array:
		item[0] = float(item[0])
		item[1] = float(item[1])
	return np.matrix(array)


x1_train = open('x1_data.txt')
x2_train = open('x2_data.txt')
x1 = openData(x1_train)
x2 = openData(x2_train)

X = np.matrix(np.concatenate((x1,x2),axis=0))
T = np.matrix(np.concatenate((np.ones(np.size(x1,axis=0)),np.zeros(np.size(x2,axis=0))),axis=0)).T

x1_1000_data = open('x1_1000.txt')
x2_1000_data = open('x2_1000.txt')

x1_1000 = openData(x1_1000_data)
x2_1000 = openData(x2_1000_data)

#W1_file = open('W1_trained.txt')
#W2_file = open('W2_trained.txt')

W1i = np.matrix(np.random.rand(2,3))
W2i = np.matrix(np.random.rand(1,3))




#W1_trained, W2_trained = trainNN(X,T,W1i,W2i,Niter,eta)


#W1_trained = np.matrix([[-0.11711,1.30896,1.52017],[0.39298,0.89084,0.46073]])
#W2_trained = np.matrix([-0.72771,1.38562,0.47514])


# y_training_class1=[]
# for exes in x1:
# 	y,z,a = ffnn(exes.T,W1_trained,W2_trained)
# 	y_training_class1.append(y[0,0])

# y_training_class2=[]
# for exes in x2:
# 	y,z,a = ffnn(exes.T,W1_trained,W2_trained)
# 	y_training_class2.append(y[0,0])

# MCR_1 = 0
# for value in y_training_class1:
# 	if value < 0.5:
# 		MCR_1 += 1

# MCR_2 = 0
# for value in y_training_class2:
# 	if value > 0.5:
# 		MCR_2 += 1

# print("Testing MCR:")
# print("Class 1: " + str(MCR_1) + '\n' + "Class 2: " + str(MCR_2))


# # Evaluating test error:
# x1_1000_data = open('x1_1000.txt')
# x2_1000_data = open('x2_1000.txt')

# x1_1000 = openData(x1_1000_data)
# x2_1000 = openData(x2_1000_data)


# classX1eval = []
# classX2eval = []
# for exes in x1_1000:
# 	y,z,a = ffnn(exes.T,W1_trained,W2_trained)
# 	classX1eval.append(y[0,0])

# for exes in x2_1000:
# 	y,z,a = ffnn(exes.T,W1_trained,W2_trained)
# 	classX2eval.append(y[0,0])

# misclassification_1 = 0
# for value in classX1eval:
# 	if value < 0.5:
# 		misclassification_1 += 1

# misclassification_2 = 0
# for value in classX2eval:
# 	if value > 0.5:
# 		misclassification_2 += 1

# print("Training MCR:")
# print("Class 1: " + str(misclassification_1) + '\n' + "Class 2: " + str(misclassification_2))


# plt.plot(x1[:,0],x1[:,1],'ro',alpha=0.5)
# plt.plot(x2[:,0],x2[:,1],'bo',alpha=0.5)
# plt.grid()
# plt.title('Training data')
# plt.xlabel('x_1')
# plt.ylabel('x_2')
# plt.show()

outputfile = open('Training optimization data.txt','w')

outputfile.write(str(W1i))
outputfile.write(str(W2i))
outputfile.write('Niter eta Training_MCR_C1 Training_MCR_C2 Test_MCR_C1 Test_MCR_2\n')
for Niter in [5,10,15,20,25,30,35]:
	for eta in [0.0001,0.0005,0.001,0.0015,0.002,0.0025,0.0030]:

		W1_trained, W2_trained = trainNN(X,T,W1i,W2i,Niter,eta)

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
		





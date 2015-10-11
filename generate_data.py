#!/usr/bin/python

import numpy as np
from sys import argv

USAGE = '''Usage: generate_data.py "Number of data in data set 1" "Number of data in data set 2" "Name of data set 1" "Name of data set 2" '''

if (len(argv) != 5):
	print USAGE
	exit(1)


script, N1, N2, x1_data, x2_data = argv

mu1 = [1,1]
sigma1 = [[1,0.5],[0.5,1]]

mu2 = [-1,-1]
sigma2 = [[1,-0.5],[-0.5,1]]

x1 = np.random.multivariate_normal(mu1,sigma1,int(N1))
x2 = np.random.multivariate_normal(mu2,sigma2,int(N2))


with open(x1_data+'.txt','w') as datafile:
	for item in x1:
		datafile.write(str(item[0]) + ' ' + str(item[1]) + '\n')


with open(x2_data+'.txt','w') as datafile:
	for item in x2:
		datafile.write(str(item[0]) + ' ' + str(item[1]) + '\n')
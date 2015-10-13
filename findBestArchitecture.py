#!/usr/local/bin/python3

from sys import argv

script, inputfile = argv

data = []
with open(inputfile) as Optimization_datafile:
	file_lines = Optimization_datafile.readlines()[7:]
	for lines in file_lines:
		data.append(lines.strip('\n').split(' '))



	i = 0
	optimal = 0
	sum_of_errors = 1000
	for thing in data:
		if ((float(thing[4]) + float(thing[5])) < sum_of_errors):
			sum_of_errors = float(thing[4]) + float(thing[5])
			optimal = i
		i += 1

	print("The optimal settings are:\n" + "Niter: " + str(data[optimal][0]) + "\n" + "eta: " + str(data[optimal][1]))
	print("The sum of errors is: " + str(sum_of_errors))
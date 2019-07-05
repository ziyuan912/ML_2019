from numpy import random
import numpy as np

def sign(n):
	new = np.zeros(len(n))
	count = 0
	for i in n:
		if i > 0:
			new[count] = int(1)
		else:
			new[count] = int(-1)
		count += 1
	return new

def ein(x,y,data_size):
	threshold = np.zeros(data_size)
	x1 = sorted(x)
	for i in range(data_size - 1):
		threshold[i] = (x1[i] + x1[i + 1])/2
	threshold[data_size - 1] = 1
	Ein = 20
	s = 0
	bestthres = 0
	for i in range(data_size):
		output = y*sign(x - threshold[i])
		count = 0
		for j in output:
			if j == -1:
				count += 1
		if count/data_size < Ein:
			s = 1
			Ein = count/data_size
			bestthres = threshold[i]
		if (data_size - count)/data_size < Ein:
			s = -1
			Ein = (data_size - count)/data_size
			bestthres = threshold[i]
	return(Ein,s,bestthres)

def q17_18():
	data_size = 20
	totalein = 0
	totaleout = 0
	s = 1
	bestthres = 0
	for i in range(5000):
		x = random.uniform(-1,1,data_size)
		noise = sign(random.uniform(-0.2,0.8,data_size))
		y = noise*sign(x)
		(Ein,s,bestthres) = ein(x, y, data_size)
		Eout = 0.5 + 0.3*s*(abs(bestthres) - 1)
		totalein += Ein
		totaleout += Eout
	print(totalein/5000, totaleout/5000)

def q19_20():
	data = open("hw2_train.dat","r")
	dataset = []
	linecount = 0
	x = [[] for i in range(9)]
	y = []
	for line in data:
		for i in range(9):
			x[i].append(float(line.split()[i]))
		y.append(int(line.split()[9]))
		linecount += 1
	Ein = np.zeros(9)
	s = np.zeros(9)
	bestthres = np.zeros(9)
	for i in range(9):
		(Ein[i],s[i],bestthres[i]) = ein(x[i],y,len(y))
	bestdim = -1
	error = len(y)
	for i in range(9):
		if Ein[i] < error:
			bestdim = i
			error = Ein[i]
	data = open("hw2_test.dat","r")
	testdat = []
	linecount = 0
	xtest = []
	ytest = []
	for line in data:
		xtest.append(float(line.split()[bestdim]))
		ytest.append(int(line.split()[9]))
		linecount += 1
	##print(bestdim,s[bestdim],bestthres[bestdim],xtest,ytest)
	ytest = np.array(ytest)
	output = ytest*int(s[bestdim])*sign(xtest - bestthres[bestdim])
	Eout = 0
	for i in output:
		if i == -1:
			Eout += 1
	Eout /= len(ytest)
	print(Ein[bestdim], Eout)



#q17_18()
q19_20()




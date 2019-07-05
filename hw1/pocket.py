import numpy as np
import random

def is_error(w, dataset, r):
	result = None
	#print(r)
	for i in r:
		x = dataset[i][0]
		y = dataset[i][1]
		#print(i, 'x=', x, 'y =',y)
		if x[0] != 1:
			x.insert(0,1)
		x = np.array(x)
		ans = 0
		if np.dot(w.T, x) > 0:
			ans = 1
		else:
			ans = -1
		if ans != y:
			result = x, y
			return result
	return result

def pocketPLA(dataset, r, num):
	w = np.zeros(5)
	wpocket = w
	minerror = checkerror(w, dataset)
	count = 0
	i = 0
	while i < 50:
		x = dataset[r[count]][0]
		y = dataset[r[count]][1]
		if x[0] != 1:
			x.insert(0,1)
		x = np.array(x)
		ans = 0
		if np.dot(w.T, x) > 0:
			ans = 1
		else:
			ans = -1
		if ans != y:
			w += x*y
			newerror = checkerror(w, dataset)
			if minerror == 0:
				return w
			if newerror < minerror:
				wpocket = w
				minerror = newerror
			#print (minerror, newerror, wpocket)
			i += 1
		count += 1
		if count == num-1:
			count = 0
	return w

def  checkerror(w, dataset):
	fault = 0
	for x,y in dataset:
		if x[0] != 1:
			x.insert(0,1)
		x = np.array(x)
		ans = 0
		if np.dot(w.T, x) > 0:
			ans = 1
		else:
			ans = -1
		if ans != y:
			fault += 1
	return fault

data = open("hw1_18_train.dat","r")
tdata = open("hw1_18_test.dat","r")
dataset = []
testdata = []
linecount = 0
testcount = 0
for line in data:
	x = [float(num) for num in line.split()[:4]]
	y = int(line.split()[4])
	linecount += 1
	dataset.append((x,y))
dataset = np.array(dataset)
for line in tdata:
	x = [float(num) for num in line.split()[:4]]
	y = int(line.split()[4])
	testcount += 1
	testdata.append((x,y))
dataset = np.array(testdata)
sums = 0
#r = list(range(linecount))
#print(PLA(dataset, r, linecount))
for i in range(50):
	r = list(range(linecount))
	random.seed(i)
	random.shuffle(r)
	w = pocketPLA(dataset, r, linecount)
	err_rate = checkerror(w, testdata)/testcount
	sums += err_rate
	print(i, err_rate)
print(sums/50)

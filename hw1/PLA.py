import numpy as np
import random
import matplotlib.pyplot as plt

def PLA(dataset, r, num):
	w = np.zeros(5).astype(float)
	count = 0
	fincount = 0
	i = 0
	while True:
		x = dataset[r[i]][0]
		y = dataset[r[i]][1]
		x = np.array(x)
		#print(x)
		if np.dot(w.T , x) > 0:
			ans = 1
		else:
			ans = -1
		if ans != y:
			w += x*y
			count += 1
			fincount = 0
		else:
			fincount += 1
		if fincount == num:
			return count
		if i == num-1:
			i = 0
		else:
			i += 1
	return count

data = open("hw1_15_train.dat","r")
dataset = []
linecount = 0
for line in data:
	x = [float(num) for num in line.split()[:4]]
	y = int(line.split()[4])
	linecount += 1
	dataset.append((x,y))
for i in range(linecount):
	dataset[i][0].insert(0,1)
dataset = np.array(dataset)
#print(dataset)
sums = 0
r = list(range(linecount))
array = []
y = np.arange(256)
#print(PLA(dataset, r, linecount))
for i in range(1126):
	r = list(range(linecount))
	random.seed(i*5)
	random.shuffle(r)
	steps = PLA(dataset, r, linecount)
	sums += steps
	#print(steps)
	array.append(steps)
plt.hist(array, bins = 20)
plt.title("number of updates")
plt.show()
print(sums/1126)


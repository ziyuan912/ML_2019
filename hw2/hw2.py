from numpy import random
import numpy as np
import matplotlib.pyplot as plt

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
	threshold = []
	for i in range(data_size - 1):
		threshold.append((x[i] + x[i + 1])/2)
	threshold.append(1)
	threshold = np.array(threshold)
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

#x = 100 + 10 * np.random.randn(10000)
s = 1
bestthres = 0
histogram = []
einsum = 0
eoutsum = 0
for i in range(1000):
	x = random.uniform(-1,1,20)
	x = sorted(x)
	noise = sign(random.uniform(-20,80,20))
	y = sign(noise*x)
	(Ein,s,bestthres) = ein(x, y, 20)
	Eout = 0.5 + 0.3*s*(abs(bestthres) - 1)
	histogram.append(Ein - Eout)
	einsum += Ein
	eoutsum += Eout
histogram = np.array(histogram)
plt.hist(histogram,facecolor = 'green',edgecolor = 'green')
print(einsum/1000, eoutsum/1000)
plt.title('Ein - Eout')
plt.show()





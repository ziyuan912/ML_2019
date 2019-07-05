import numpy as np
import matplotlib.pyplot as plt


def logistic(x,w,y):
	theta = 1/(1+np.exp(y*w.T.dot(x)))
	return x*y*theta
def error(E,x,w,y,length):
	err = 0
	for i in range(length):
		output = np.sign(x[i].dot(w))
		if y[i] != output:
			err += 1
	E.append(err/length)
	return E

data = open("hw3_train.dat","r")
dataset = []
Ein = []
Eout = []
linecount = 0
x = [[] for i in range(1000)]
y = []
data2 = open("hw3_test.dat","r")
linecount2 = 0
x2 = [[] for i in range(3000)]
y2 = []
t = np.arange(2000)
for line in data2:
	x2[linecount2].append(1)
	for i in range(20):
		x2[linecount2].append(float(line.split()[i]))
	y2.append(int(line.split()[20]))
	linecount2 += 1
x2 = np.array(x2)
y2 = np.array(y2)
for line in data:
	x[linecount].append(1)
	for i in range(20):
		x[linecount].append(float(line.split()[i]))
	y.append(int(line.split()[20]))
	linecount += 1
x = np.array(x)
y = np.array(y)
w = np.zeros(21)
for i in range(2000):
	s = np.zeros(21)
	s = logistic(x[i%linecount],w,y[i%linecount])
	w += 0.001*s
	Ein = error(Ein,x,w,y,linecount)
	Eout = error(Eout,x2,w,y2,linecount2)
	print(Ein[-2:-1],Eout[-2:-1])
print(w)

plt.bar(t,Ein)
plt.title('Ein_SGD')
plt.show()
plt.bar(t,Eout)
plt.title('Eout_SGD')
plt.show()
print(Eout[1999])


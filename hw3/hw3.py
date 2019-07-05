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
EinSGD = []
EoutSGD = []
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
wSGD = np.zeros(21)
for i in range(2000):
	s = np.zeros(21)
	sSGD = np.zeros(21)
	for j in range(linecount):
		s += logistic(x[j],w,y[j])
	sSGD = logistic(x[i%linecount],wSGD,y[i%linecount])
	wSGD += 0.001*sSGD
	s /= linecount
	w += 0.01*s
	Ein = error(Ein,x,w,y,linecount)
	Eout = error(Eout,x2,w,y2,linecount2)
	EinSGD = error(EinSGD,x,wSGD,y,linecount)
	EoutSGD = error(EoutSGD,x2,wSGD,y2,linecount2)

fig = plt.subplot()	
fig.bar(t-0.5,Ein,color = 'b',width = 0.5,label = 'logistic regression')
fig.bar(t,EinSGD,color = 'r',width = 0.5, label = 'SGD')
plt.legend()
plt.title('Ein')
plt.show()
fig2 = plt.subplot()
fig2.bar(t-0.5,Eout,color = 'b',width = 0.5,label = 'logistic regression')
fig2.bar(t,EoutSGD,color = 'r',width = 0.5,label = 'SGD')
plt.legend()
plt.title('Eout')
plt.show()


"""Eoutnum = 0
for i in range(linecount2):
	output = np.sign(x2[i].dot(w))
	if y2[i] != output:
		Eoutnum += 1
print(Eoutnum/linecount2)"""


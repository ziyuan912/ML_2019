import numpy as np
import random
import matplotlib.pyplot as plt

train_data = open("hw4_train.dat")
test_data = open("hw4_test.dat")
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
traincnt = 0
testcnt = 0
for line in train_data:
	x = [float(num) for num in line.split()[:-1]]
	y = int(line.split()[-1])
	Xtrain.append(x)
	Ytrain.append(y)
	traincnt += 1

for line in test_data:
	x = [float(num) for num in line.split()[:-1]]
	y = int(line.split()[-1])
	Xtest.append(x)
	Ytest.append(y)
	testcnt += 1
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

def distance(X, center):
	return np.linalg.norm(X-center)

def kNearest(x, X, Y, k):
	dis = []
	for i in X:
		dis.append(distance(i, x))
	dis = np.array(dis)
	nearindex = np.argsort(dis)
	nearindex = nearindex[:k]
	predict = np.sign(np.sum(Y[nearindex]))
	return predict

def Predict(X, Y, Xtrain,Ytrain, K):
	E = []
	for k in K:
		error = 0
		for i in range(len(Y)):
			predict = kNearest(X[i], Xtrain, Ytrain, k)
			if Y[i] != predict:
				error += 1
		E.append(error/len(Y))
	return E

if __name__ == '__main__':
	k = [1, 3, 5, 7, 9]
	Ein = Predict(Xtrain, Ytrain, Xtrain, Ytrain, k)
	Eout = Predict(Xtest, Ytest, Xtrain, Ytrain, k)
	plt.plot(k, Ein)
	plt.title('k vs $E_{in}$')
	plt.xlabel('k')
	plt.ylabel('$E_{in}$')
	plt.show()
	plt.plot(k, Eout)
	plt.title('k vs $E_{out}$')
	plt.xlabel('k')
	plt.ylabel('$E_{out}$')
	plt.show()


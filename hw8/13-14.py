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
	return np.sum(np.square(X-center))

def RBF(x, X, Y, Gamma):
	predict = 0
	for i in range(len(X)):
		predict += Y[i]*np.exp(-1*Gamma*distance(X[i], x))
	return np.sign(predict)

def Predict(X, Y, Xtrain,Ytrain, Gamma):
	E = []
	for r in Gamma:
		error = 0
		for i in range(len(Y)):
			predict = RBF(X[i], Xtrain, Ytrain, r)
			if Y[i] != predict:
				error += 1
		E.append(error/len(Y))
	return E

if __name__ == '__main__':
	Gamma = [0.001, 0.1, 1, 10, 100]
	Ein = Predict(Xtrain, Ytrain, Xtrain, Ytrain, Gamma)
	Eout = Predict(Xtest, Ytest, Xtrain, Ytrain, Gamma)
	xGamma = ['0.001', '0.1', '1', '10', '100']
	plt.plot(xGamma, Ein)
	plt.title('$\gamma$ vs $E_{in}$')
	plt.xlabel('$\gamma$')
	plt.ylabel('$E_{in}$')
	plt.show()
	plt.plot(xGamma, Eout)
	plt.title('$\gamma$ vs $E_{out}$')
	plt.xlabel('$\gamma$')
	plt.ylabel('$E_{out}$')
	plt.show()
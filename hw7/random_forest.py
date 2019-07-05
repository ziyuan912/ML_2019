import decision_tree
import numpy as np
import matplotlib.pyplot as plt
import random

def evaluate(X, Y, Score, root):
	errorrate = 0
	trainnum = X.shape[0]
	for i in range(trainnum):
		Score[i] += decision_tree.ans(X[i], root)
		if np.sign(Score[i]) != Y[i]:
			errorrate += 1
	return errorrate/trainnum

if __name__ == '__main__':
	traindata = open("hw3_train.dat")
	testdata = open("hw3_test.dat")
	Xtrain = []
	Ytrain = []
	Xtest = []
	Ytest = []

	for line in traindata:
		x = [float(num) for num in line.split()[:2]]
		y = int(line.split()[2])
		Xtrain.append(x)
		Ytrain.append(y)
	Xtrain = np.array(Xtrain)
	Ytrain = np.array(Ytrain)

	for line in testdata:
		x = [float(num) for num in line.split()[:2]]
		y = int(line.split()[2])
		Xtest.append(x)
		Ytest.append(y)
	Xtrain = np.array(Xtrain)
	Ytrain = np.array(Ytrain)
	Xtest = np.array(Xtest)
	Ytest = np.array(Ytest)
	trainnum = Xtrain.shape[0]
	treenum = int(trainnum*0.8)
	Ein = []
	Einscore = np.zeros(Xtrain.shape[0])
	Eoutscore = np.zeros(Xtest.shape[0])
	EinG = []
	EoutG = []
	for i in range(30000):
		bagging = [random.randint(0, trainnum-1) for j in range(treenum)]
		Xtree = Xtrain[bagging, :]
		Ytree = Ytrain[bagging]
		root = decision_tree.makeTree(Xtree, Ytree, 10000)
		Ein.append(decision_tree.error(Xtrain, Ytrain, root))
		EinG.append(evaluate(Xtrain, Ytrain, Einscore, root))
		#print(Einscore)
		EoutG.append(evaluate(Xtest, Ytest, Eoutscore, root))

	plt.hist(Ein)
	plt.title('$E_{in}(g_{t})$')
	plt.xlabel('$E_{in}(g_{t})$')
	plt.show()

	plt.plot(EinG)
	plt.title('t vs $E_{in}(G_{t})$')
	plt.xlabel('t')
	plt.ylabel('$E_{in}(G_{t})$')
	plt.show()

	plt.plot(EoutG)
	plt.title('t vs $E_{out}(G_{t})$')
	plt.xlabel('t')
	plt.ylabel('$E_{out}(G_{t})$')
	plt.show()	

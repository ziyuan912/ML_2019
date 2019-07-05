import numpy as np
import matplotlib.pyplot as plt

class Tree:
	def __init__(self, d, theta, s = None):
		self.d = d
		self.theta = theta
		self.s = s
		self.left = None
		self.right = None

def decision_stump(X, Y, i):
	theta = [[]for i in range(2)]
	trainnum = X.shape[0]
	x1 = sorted(X[:, 0])
	x2 = sorted(X[:, 1])
	theta[0].append(x1[0] - 1), theta[1].append(x2[0] - 1)
	for i in range(trainnum - 1):
		theta[0].append((x1[i] + x1[i + 1])/2)
		theta[1].append((x2[i] + x2[i + 1])/2)
	theta[0].append(x1[trainnum - 1] + 1), theta[1].append(x2[trainnum - 1] + 1)
	theta = np.array(theta)
	bestpurity = np.inf
	besttheta = 0
	besti = 0
	for j in range(2):
		for t in theta[j]:
			yleft = []
			yright = []
			for k in range(X.shape[0]):
				if X[k][j] <= t:
					yleft.append(Y[k])
				else:
					yright.append(Y[k])
			yleft = np.array(yleft)
			yright = np.array(yright)
			impurity = len(yleft)*gini_index(yleft) + len(yright)*gini_index(yright)
			if(impurity < bestpurity):
				bestpurity = impurity
				besttheta = t
				besti = j
	return (bestpurity, besttheta, besti)
	

def gini_index(Y):
	if(len(Y) == 0):
		return 1
	muplus = np.mean(Y == 1)
	return (1 - np.square(muplus) - np.square(1 - muplus))

def makeTree(X, Y, h):
	if np.sum(X != X[0, :]) == 0 or np.sum(Y != Y[0]) == 0 or h == 0:
		yplus = np.sum(Y == 1)
		yneg = np.sum(Y == -1)
		if yplus > yneg:
			return Tree(None, None, 1)
		else:
			return Tree(None, None, -1)
	bestpurity, besttheta, i = decision_stump(X, Y, 2)
	root = Tree(i, besttheta)
	xleft = []
	yleft = []
	xright = []
	yright = []
	for k in range(X.shape[0]):
		if X[k][i] <= besttheta:
			xleft.append(X[k])
			yleft.append(Y[k])
		else:
			xright.append(X[k])
			yright.append(Y[k])
	xleft = np.array(xleft)
	yleft = np.array(yleft)
	xright = np.array(xright)
	yright = np.array(yright)
	#print(bestpurity, besttheta, i)
	#print(xleft.shape, xright.shape)
	root.left = makeTree(xleft, yleft, h - 1)
	root.right = makeTree(xright, yright, h - 1)
	return root

def ans(x, root):
	if root.s != None:
		return root.s
	if x[root.d] <= root.theta:
		return ans(x, root.left)
	else:
		return ans(x, root.right)

def error(X, Y, root):
	errorrate = 0
	for i in range(len(Y)):
		if ans(X[i], root) != Y[i]:
			errorrate += 1
	return errorrate/len(Y)
	
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

	root = makeTree(Xtrain, Ytrain, 10000)
	print('Ein:', error(Xtrain, Ytrain, root))
	print('Eout:', error(Xtest, Ytest, root))
	H = np.arange(1, 6)
	Ein = []
	Eout = []
	for h in H:
		rooth = makeTree(Xtrain, Ytrain, h)
		Ein.append(error(Xtrain, Ytrain, rooth))
		Eout.append(error(Xtest, Ytest, rooth))
	plt.bar(H, Ein)
	plt.title('h vs $E_{in}(g_{h})$')
	plt.xlabel('h')
	plt.ylabel('$E_{in}(g_{h})$')
	plt.show()

	plt.bar(H, Eout)
	plt.title('h vs $E_{out}(g_{h})$')
	plt.xlabel('h')
	plt.ylabel('$E_{out}(g_{h})$')
	plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

def opendata(file):
	X = []
	Y = []
	data = open(file,"r")
	for line in data:
		x = [float(num) for num in line.split()[1:]]
		y = float(line.split()[0])
		X.append(x)
		Y.append(y)
	X = np.array(X)
	Y = np.array(Y)
	return (X,Y)

(train_X, train_Y) = opendata("./features.train")
(test_X, test_Y) = opendata("./features.test")

train_Y1 = []
for y in train_Y:
	if(y == 0):
		train_Y1.append(1)
	else:
		train_Y1.append(-1)

Gamma = [-2, -1, 0, 1, 2]
Evalcount = [0 for j in range(5)]

for i in range(100): 
	trainval_X, testval_X, trainval_Y, testval_Y = train_test_split(train_X, train_Y1, test_size = 1000)
	Eval = []
	for gamma in Gamma:
		clf = svm.SVC(C = 0.1, kernel = 'rbf', gamma = pow(10,gamma))
		clf.fit(trainval_X,trainval_Y)
		e = np.sum(clf.predict(testval_X) != testval_Y)/1000
		Eval.append(e)
	best_gamma = np.argmin(Eval)
	Evalcount[best_gamma] += 1


	
Evalcount = np.array(Evalcount)
Gamma = np.array(Gamma)
plt.bar(Gamma,Evalcount)
plt.title("best $\gamma$")
plt.xlabel('$log_{10}\gamma$')
plt.ylabel('count')
plt.show()

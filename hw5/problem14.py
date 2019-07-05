import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

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
test_Y1 = []
for y in train_Y:
	if(y == 4):
		train_Y1.append(1)
	else:
		train_Y1.append(-1)
for y in test_Y:
	if(y == 4):
		test_Y1.append(1)
	else:
		test_Y1.append(-1)

C = [-5, -3, -1, 1, 3]
Ein = []

for c in C: 
	clf = svm.SVC(C = pow(10,c), kernel = 'poly',coef0 = 1, gamma = 1, degree = 2)
	clf.fit(train_X,train_Y1)
	ein = np.sum(clf.predict(train_X) != train_Y1)
	Ein.append(ein/len(train_X))
	
Ein = np.array(Ein)
C = np.array(C)
plt.plot(C,Ein)
plt.title("Ein vs $log_{10}C$")
plt.xlabel('$log_{10}C$')
plt.ylabel('Ein')
plt.show()


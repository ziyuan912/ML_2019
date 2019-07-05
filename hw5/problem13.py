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
for y in train_Y:
	if(y == 2):
		train_Y1.append(1)
	else:
		train_Y1.append(-1)

C = [-5, -3, -1, 1, 3]
W = []

for c in C: 
	clf = svm.SVC(C = pow(10,c), kernel = 'linear')
	clf.fit(train_X,train_Y1)
	w = clf.coef_[0]
	W.append(np.sqrt(np.dot(w,w)))
W = np.array(W)
C = np.array(C)
plt.plot(C,W)
plt.title("$||w||$ vs $log_{10}C$")
plt.xlabel('$log_{10}C$')
plt.ylabel('$||w||$')
plt.show()

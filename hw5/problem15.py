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
	if(y == 0):
		train_Y1.append(1)
	else:
		train_Y1.append(-1)

C = [-2, -1, 0, 1, 2]
Dis = []

for c in C: 
	clf = svm.SVC(C = pow(10,c), kernel = 'rbf', gamma = 80)
	clf.fit(train_X,train_Y1)
	w = 0
	for alpham,Xm in zip(clf.dual_coef_[0],clf.support_vectors_):
		for alphan,Xn in zip(clf.dual_coef_[0],clf.support_vectors_):
			XX = Xn-Xm
			w += alphan*alpham*np.exp(-80*np.dot(XX,XX))
	dis = 1/np.sqrt(w)
	Dis.append(dis)
	
Dis = np.array(Dis)
C = np.array(C)
plt.plot(C,Dis)
plt.title("Distance vs $log_{10}C$")
plt.xlabel('$log_{10}C$')
plt.ylabel('Margin')
plt.show()

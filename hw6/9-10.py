import numpy as np
import matplotlib.pyplot as plt


data = open("hw2_lssvm_all.dat")
linecnt = 0
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []
lamda = [0.05, 0.5, 5, 50, 500]
strlamda = ['0.05', '0.5', '5', '50', '500']

for line in data:
	x = [1]
	x += [float(num) for num in line.split()[:-1]]
	y = int(line.split()[-1])
	if linecnt < 400:
		Xtrain.append(x)
		Ytrain.append(y)
	else:
		Xtest.append(x)
		Ytest.append(y)
	linecnt += 1

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

Ein = []
Eout = []

for l in lamda:
	temp = np.linalg.inv(l*np.identity(11) + np.dot(Xtrain.T,Xtrain))
	w = (temp.dot(Xtrain.T)).dot(Ytrain)
	ypredict = w.dot(Xtrain.T)
	ein = np.sum(np.sign(ypredict) != Ytrain)
	ein /= 400
	Ein.append(ein)
	ypredict = w.dot(Xtest.T)
	eout = np.sum(np.sign(ypredict) != Ytest)
	eout /= (linecnt - 400)
	Eout.append(eout)

plt.plot(strlamda, Ein)
plt.title('$\lambda$ vs $E_{in}(g)$')
plt.xlabel('$\lambda$')
plt.ylabel('$E_{in}(g)$')
plt.show()
plt.plot(strlamda, Eout)
plt.title('$\lambda$ vs $E_{out}(g)$')
plt.xlabel('$\lambda$')
plt.ylabel('$E_{out}(g)$')
plt.show()

print('minimum Ein(g):', min(Ein),'corresponding lambda:', lamda[np.argmin(Ein)])
print('minimun Eout(g):', min(Eout),'corresponding lambda:', lamda[np.argmin(Eout)])
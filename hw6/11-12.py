import numpy as np
import random
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
einG = []
eoutG = []

random.seed(100)

for i in range(len(lamda)):
	G = []
	for t in range(250):
		bootstrap = [random.randint(0, 399) for k in range(400)]
		bootX = Xtrain[bootstrap]
		bootY = Ytrain[bootstrap]
		temp = np.linalg.inv(lamda[i]*np.identity(11) + np.dot(bootX.T,bootX))
		w = (temp.dot(bootX.T)).dot(bootY)
		G.append(w)
	G = np.array(G)
	ypredict = np.zeros(400)
	for t in range(250):
		ypredict += np.sign(Xtrain.dot(G[t]))
	ein = np.sum(np.sign(ypredict) != Ytrain)
	ein /= 400
	#print('1:', ein, lamda[i])
	Ein.append(ein)
	ypredict = np.zeros(linecnt - 400)
	for t in range(250):
		ypredict += np.sign(G[t].dot(Xtest.T))
	eout = np.sum(np.sign(ypredict) != Ytest)
	eout /= (linecnt - 400)
	#print('2:', eout, lamda[i])
	Eout.append(eout)

plt.plot(strlamda, Ein)
plt.title('$\lambda$ vs $E_{in}(G)$')
plt.xlabel('$\lambda$')
plt.ylabel('$E_{in}(G)$')
plt.show()
plt.plot(strlamda, Eout)
plt.title('$\lambda$ vs $E_{out}(G)$')
plt.xlabel('$\lambda$')
plt.ylabel('$E_{out}(G)$')
plt.show()

print('minimum Ein(g):', min(Ein),'corresponding lambda:', lamda[np.argmin(Ein)])
print('minimun Eout(g):', min(Eout),'corresponding lambda:', lamda[np.argmin(Eout)])
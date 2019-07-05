import numpy as np
import matplotlib.pyplot as plt

def decision_stump(X, Y, theta, u):
	predict = np.sign(X - theta)
	error1 = u.dot(Y != predict)/np.sum(u)
	error2 = 1 - error1
	if(error1 > error2):
		return (error2, -1)
	else:
		return (error1, 1)


def adaboost(X, Y, theta, u):
	trainnum = u.shape[0]
	Ein = []
	g = []
	U = []
	for t in range(300):
		minEin = 1
		mintheta = 0
		mins = 1
		mind = 0
		U.append(np.sum(u))
		for i in range(theta.shape[1]):
			(error1, s1) = decision_stump(X[:, 0], Y, theta[0][i], u)
			(error2, s2) = decision_stump(X[:, 1], Y, theta[1][i], u)
			if(error1 < error2 and error1 < minEin):
				minEin = error1
				mintheta = theta[0][i]
				mins = s1
				mind = 0
			elif(error2 < error1 and error2 < minEin):
				minEin = error2
				mintheta = theta[1][i]
				mins = s2
				mind = 1
		epson = np.sqrt((1 - minEin)/minEin)
		ein = 0
		for i in range(trainnum):
			if(mins*np.sign(X[i][mind] - mintheta) != Y[i]):
				u[i] *= epson
				ein += 1
			else:
				u[i] /= epson
		Ein.append(ein/trainnum)
		g.append((mins, mind, mintheta, np.log(epson)))
	t = np.arange(300)
	plt.plot(t, Ein)
	plt.title('t vs $E_{in}(g_{t})$')
	plt.xlabel('t')
	plt.ylabel('$E_{in}(g_{t})$')
	plt.show()
	print("Ein(gT) = ",Ein[299])

	plt.plot(t, U)
	plt.title('t vs $U_{t}$')
	plt.xlabel('t')
	plt.ylabel('$U_{t}$')
	plt.show()
	print("Ut = ",U[299])	
	return g

def predict(X, G, t):
	ans = np.zeros(X.shape[0])
	for i in range(t):
		s = G[i][0]
		d = G[i][1]
		theta = G[i][2]
		alpha = G[i][3]
		ans += alpha*s*np.sign(X[:, d] - theta)
	return np.sign(ans)


data = open("hw2_adaboost_train.dat")
trainnum = 0
X = []
Y = []
t = np.arange(300)

for line in data:
	x = [float(num) for num in line.split()[:2]]
	y = int(line.split()[2])
	X.append(x)
	Y.append(y)
	trainnum += 1

X = np.array(X)
Y = np.array(Y)

theta = [[]for i in range(2)]

x1 = sorted(X[:, 0])
x2 = sorted(X[:, 1])

theta[0].append(x1[0] - 1), theta[1].append(x2[0] - 1)
for i in range(trainnum - 1):
	theta[0].append((x1[i] + x1[i + 1])/2)
	theta[1].append((x2[i] + x2[i + 1])/2)
theta[0].append(x1[trainnum - 1] + 1), theta[1].append(x2[trainnum - 1] + 1)
theta = np.array(theta)

u = np.ones(trainnum)/trainnum

G = adaboost(X, Y, theta, u)

EinG = []
for i in range(300):
	ans = predict(X, G, i+1)
	EinG.append(np.sum(ans != Y)/trainnum)
plt.plot(t, EinG)
plt.title('t vs $E_{in}(G_{t})$')
plt.xlabel('t')
plt.ylabel('$E_{in}(G_{t})$')
plt.show()
print("Ein(GT) = ", EinG[299])

data = open("hw2_adaboost_test.dat")
testX = []
testY = []
for line in data:
	x = [float(num) for num in line.split()[:2]]
	y = int(line.split()[2])
	testX.append(x)
	testY.append(y)

testX = np.array(testX)
testY = np.array(testY)

EoutG = []
for i in range(300):
	ans = predict(testX, G, i+1)
	EoutG.append(np.sum(ans != testY)/testY.shape[0])
plt.plot(t, EoutG)
plt.title('t vs $E_{out}(G_{t})$')
plt.xlabel('t')
plt.ylabel('$E_{out}(G_{t})$')
plt.show()
print("Eout(GT) = ", EoutG[299])


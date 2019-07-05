import numpy as np
import random
import matplotlib.pyplot as plt

train_data = open("hw4_nolabel_train.dat")
Xtrain = []
traincnt = 0
for line in train_data:
	x = [float(num) for num in line.split()]
	Xtrain.append(x)
	traincnt += 1

Xtrain = np.array(Xtrain)

def distance(X, center):
	return np.sum(np.square(X-center))

def find_class(classify, X, cluster):
	for i in range(X.shape[0]):
		dis = []
		for c in cluster:
			dis.append(distance(X[i], c))
		classify[i] = int(np.argmin(dis))

def change_cluster(classify, X, cluster):
	count = np.zeros(len(cluster))
	for i in range(len(cluster)):
		cluster[i] = 0
	for i in range(X.shape[0]):
		cluster[int(classify[i])] += X[i]
		count[int(classify[i])] += 1
	for i in range(len(cluster)):
		cluster[i] /= count[i]


def Kmeans(X, k):
	Ein = []
	for t in range(500):
		cluster = random.sample(list(X), k)
		cluster = np.array(cluster)
		classify = np.zeros(X.shape[0])
		while True:
			find_class(classify, X, cluster)
			temp = cluster
			change_cluster(classify, X, cluster)
			if temp.all == cluster.all:
				break
		ein = 0
		for i in range(X.shape[0]):
			ein += distance(X[i], cluster[int(classify[i])])
		Ein.append(ein/X.shape[0])
	return Ein
		

if __name__ == '__main__':
	K = [2, 4, 6, 8, 10]
	avEin = []
	varEin = []
	for k in K:
		ein = Kmeans(Xtrain, k)
		avein = np.sum(ein)/500
		#print(avein)
		avEin.append(avein)
		varEin.append(np.var(ein))
	plt.plot(K, avEin)
	plt.title('K vs average of $E_{in}$')
	plt.xlabel('K')
	plt.ylabel('average of $E_{in}$')
	plt.show()
	plt.plot(K, varEin)
	plt.title('K vs variance of $E_{in}$')
	plt.xlabel('K')
	plt.ylabel('variance of $E_{in}$')
	plt.show()


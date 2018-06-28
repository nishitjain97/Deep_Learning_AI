import numpy as np

class NeuralNetwork:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.w1 = np.random.randn(3, 2) * 0.01
        self.b1 = 0
        self.w2 = np.random.randn(1, 3) * 0.01
        self.b2 = 0
        self.m = X.shape[1]
        self.Z1 = 0
        self.layer1 = 0
        self.Z2 = 0
        self.layer2 = np.zeros(Y.shape)

        self.forwardprop()

    def sigmoid(self, Z):
        return(1 / (1 + np.exp(-Z)))

    def forwardprop(self):
        self.Z1 = np.dot(self.w1, self.X) + self.b1
        self.layer1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.w2, self.layer1) + self.b2
        self.layer2 = self.sigmoid(self.Z2)

    def backprop(self):
        dz2 = self.layer2 - self.Y
        dw2 = (1 / self.m) * np.dot(dz2, self.layer1.T)
        db2 = (1 / self.m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.w2.T, dz2) * self.sigmoid(self.Z1) * (1 - self.sigmoid(self.Z1))
        dw1 = (1 / self.m) * np.dot(dz1, self.X.T)
        db1 = (1 / self.m) * np.sum(dz1, axis=1, keepdims=True)

        grads = {
            "dw": [dw1, dw2],
            "db": [db1, db2]
        }
        return grads

    def gradient_descent(self, learning_rate=0.001, num_iters=1000):
        print('Cost before gradient descent:', self.cost())
        print('Learning rate:', learning_rate)
        print('Number of iterations:', num_iters)

        for i in range(num_iters):
            grads = self.backprop()
            dw = grads['dw']
            db = grads['db']
            dw1 = dw[0]
            dw2 = dw[1]
            db1 = db[0]
            db2 = db[1]

            self.w1 = self.w1 - learning_rate * dw1
            self.b1 = self.b1 - learning_rate * db1
            self.w2 = self.w2 - learning_rate * dw2
            self.b2 = self.b2 - learning_rate * db2

            self.forwardprop()

            if (i+1) % 100 == 0:
                print('Cost after iteration', i+1, ':', self.cost())

    def cost(self):
        try:
            logged = -1 * self.Y * np.log(self.layer2) - (1 - self.Y) * np.log(1 - self.layer2)
        except:
            return 0
        else:
            return (1 / self.m) * np.sum(logged)

data = []
for i in range(100):
    data = data + [[-j, -(i * j), 0] for j in range(1, 1000, 3)]
    data = data + [[j, (i * j), 1] for j in range(1, 1000, 5)]

data = np.array(data)
X = data[:, :2].T
for i in range(X.shape[0]):
    X[i] = X[i] - np.mean(X[i])
    X[i] = X[i] / np.max(X[i])
Y = data[:, 2].T

net1 = NeuralNetwork(X, Y)
net1.gradient_descent(num_iters=1000)
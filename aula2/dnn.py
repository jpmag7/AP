#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from dataset import Dataset


class DNN:
    
    def __init__(self, dataset, hidden_layers = [2, 1], normalize = False):
        self.X, self.y = dataset.getXy()
        self.X = np.hstack ( (np.ones([self.X.shape[0],1]), self.X ) )
        
        self.h = hidden_layers
        self.Ws = [np.zeros([self.h[0], self.X.shape[1]])]

        for i in range(len(self.h)):
            self.Ws.append(np.zeros([self.h[i], self.h[i-1] + 1]))
        #self.Ws.append(np.zeros([1, self.h[-1] + 1]))
        
        if normalize:
            self.normalized = True
            self.normalize()
        else:
            self.normalized = False


    def setWeights(self, ws):
        self.Ws = []
        for w in ws: self.Ws.append(w)
        

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        
        if self.normalized:
            if np.all(self.sigma != 0): 
                x[1:] = (x[1:] - self.mu) / self.sigma
            else: 
                x[1:] = (x[1:] - self.mu)
        
        a = [x]
        for i in range(len(self.h)-1):
            z = np.dot(self.Ws[i], a[i])
            ai = sigmoid(z)
            ai = np.hstack((1, ai))
            a.append(ai)
        
        z2 = np.dot(self.Ws[-1], a[-1].T)
        y_hat = sigmoid(z2)

        return y_hat



    def costFunction(self, weights=None):
        if weights is not None:
            #self.Ws = weights
            pass

        m = self.X.shape[0]

        a = [self.X]
        for i in range(len(self.h)-1):
            z = np.dot(a[i], self.Ws[i].T)
            ai = sigmoid(z)
            maux = a[-1].shape[0]
            ai = np.hstack((np.ones([maux,1]), ai))
            a.append(ai)

        z = np.dot(a[-1], self.Ws[-1].T)
        y_hat = sigmoid(z)

        # add a small constant value to the predicted values that are zero
        y_hat[y_hat == 0] = 1e-10
        y_hat[y_hat == 1] = 1 - 1e-10
        
        J = -1/m * (np.dot(self.y.T, np.log(y_hat)) + np.dot((1 - self.y).T, np.log(1 - y_hat)))
        return J

    def set_weights_1D(self, weights):
        self.Ws = []
        start = 0
        end = self.h[0] * self.X.shape[1]
        self.Ws.append(weights[:end].reshape((self.h[0], self.X.shape[1])))
        for i in range(len(self.h) - 1):
            start = end
            end = start + (self.h[i]+1) * self.h[i+1]
            self.Ws.append(weights[start:end].reshape((self.h[i+1], self.h[i]+1)))

    def build_model(self):
        from scipy import optimize
        size = self.X.shape[1] * self.h[0] + sum([(self.h[i]+1) * self.h[i+1] for i in range(len(self.h) - 1)])

        weights = np.random.rand(size)
        self.set_weights_1D(weights)
        
        result = optimize.minimize(lambda w: self.costFunction(w), weights, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        
        weights = result.x
        self.set_weights_1D(weights)


    def normalize(self):
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

    

def test():
    print("Test 1")
    ds= Dataset("xnor.data")
    nn = DNN(ds, [2, 1])
    w1 = np.array([[-30,20,20],[10,-20,-20]])
    w2 = np.array([[-10,20,20]])
    nn.setWeights([w1, w2])
    print( nn.predict(np.array([0,0]) ) )
    print( nn.predict(np.array([0,1]) ) )
    print( nn.predict(np.array([1,0]) ) )
    print( nn.predict(np.array([1,1]) ) )
    print("Error:", nn.costFunction())

def test2():
    print("Test 2")
    ds= Dataset("xnor.data")
    nn = DNN(ds, [2, 1], normalize = False)
    nn.build_model()
    print( nn.predict(np.array([0,0]) ) )
    print( nn.predict(np.array([0,1]) ) )
    print( nn.predict(np.array([1,0]) ) )
    print( nn.predict(np.array([1,1]) ) )
    print("Error:", nn.costFunction())
    
test()
test2()

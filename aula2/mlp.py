#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from dataset import Dataset


class MLP:
    
    def __init__(self, dataset, hidden_nodes = 2, normalize = False):
        self.X, self.y = dataset.getXy()
        self.X = np.hstack ( (np.ones([self.X.shape[0],1]), self.X ) )
        
        self.h = hidden_nodes
        self.W1 = np.zeros([hidden_nodes, self.X.shape[1]])
        self.W2 = np.zeros([1, hidden_nodes+1])
        
        if normalize:
            self.normalize()
        else:
            self.normalized = False


    def setWeights(self, w1, w2):
        self.W1 = w1
        self.W2 = w2
        

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        
        if self.normalized:
            if np.all(self.sigma != 0): 
                x[1:] = (x[1:] - self.mu) / self.sigma
            else: 
                x[1:] = (x[1:] - self.mu)
        
        z1 = np.dot(self.W1, x)
        a1 = sigmoid(z1)
        a1 = np.hstack((1, a1))
        z2 = np.dot(self.W2, a1.T)
        y_hat = sigmoid(z2)

        return y_hat



    def costFunction(self, weights=None):
        if weights is not None:
            self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
            self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])

        m = self.X.shape[0]

        z1 = np.dot(self.X, self.W1.T)
        a1 = sigmoid(z1)
        a1 = np.hstack((np.ones([m,1]), a1))
        z2 = np.dot(a1, self.W2.T)
        y_hat = sigmoid(z2)

        # add a small constant value to the predicted values that are zero
        y_hat[y_hat == 0] = 1e-10
        y_hat[y_hat == 1] = 1 - 1e-10

        J = -1/m * (np.dot(self.y.T, np.log(y_hat)) + np.dot((1 - self.y).T, np.log(1 - y_hat)))

        return J




    def build_model(self):
        from scipy import optimize

        size = self.h * self.X.shape[1] + self.h+1
        
        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        weights = result.x
        self.W1 = weights[:self.h * self.X.shape[1]].reshape([self.h, self.X.shape[1]])
        self.W2 = weights[self.h * self.X.shape[1]:].reshape([1, self.h+1])

    def normalize(self):
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

    

def test():
    ds= Dataset("xnor.data")
    nn = MLP(ds, 2)
    w1 = np.array([[-30,20,20],[10,-20,-20]])
    w2 = np.array([[-10,20,20]])
    nn.setWeights(w1, w2)
    print( nn.predict(np.array([0,0]) ) )
    print( nn.predict(np.array([0,1]) ) )
    print( nn.predict(np.array([1,0]) ) )
    print( nn.predict(np.array([1,1]) ) )
    print(nn.costFunction())

def test2():
    ds= Dataset("xnor.data")
    nn = MLP(ds, 5, normalize = False)
    nn.build_model()
    print( nn.predict(np.array([0,0]) ) )
    print( nn.predict(np.array([0,1]) ) )
    print( nn.predict(np.array([1,0]) ) )
    print( nn.predict(np.array([1,1]) ) )
    print(nn.costFunction())
    
#test()
test2()

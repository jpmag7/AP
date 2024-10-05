# -*- coding: utf-8 -*-
"""
@author: miguelrocha
@author: José Magalhães
"""

import numpy as np
from dataset import Dataset
import matplotlib.pyplot as plt

class LogisticRegression:
    
    def __init__(self, dataset, split=0.2, standardize = False, regularization = False, lamda = 1):
        if standardize:
            dataset.standardize()
            dataset.split(split)
            self.X = np.hstack ((np.ones([dataset.nrows(),1]), dataset.Xst_train))
            self.standardized = True
        else:
            dataset.split(split)
            self.X = np.hstack ((np.ones([dataset.nrows(),1]), dataset.X_train))
            self.standardized = False
        self.y = dataset.Y_train
        self.X_test = np.hstack ((np.ones([dataset.nrowsTest(),1]), dataset.X_test))
        self.Y_test = dataset.Y_test
        self.theta = self.theta = np.zeros(self.X.shape[1])
        self.regularization = regularization
        self.lamda = lamda
        self.data = dataset
       

    def printCoefs(self):
        print(self.theta)


    def probability(self, instance):
        x = np.empty([self.X.shape[1]])
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])
        if self.standardized:
            if np.all(self.sigma!= 0): 
                x[1:] = (x[1:] - self.data.mu) / self.data.sigma
            else: x[1:] = (x[1:] - self.mu) 
        ####
        return sigmoid(np.dot(self.theta, x))


    def predict(self, instance):
        ####
        res = self.probability(instance)
        if res >= 0.5: res = 1
        else: res = 0
        return res


    def predictMultiple(self, instances):
        res = []
        for instance in instances:
            res.append(self.predict(instance))
        return res
  

    def costFunction(self, theta = None):
        if theta is None: theta= self.theta
        m = self.X.shape[0]
        # predictions
        p = np.dot(self.X, self.theta)
        h = sigmoid(p)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        # cost function
        cost = (-1/m) * np.sum(self.y * np.log(h) + (1 - self.y) * np.log(1 - h))
        return cost


    def costFunctionReg(self, theta = None, lamda = 1):
        if theta is None: theta= self.theta
        m = self.X.shape[0]
        # predictions
        p = np.dot(self.X, self.theta)
        h = sigmoid(p)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        # cost function
        cost = cost = (-1/m) * np.sum(self.y * np.log(h) + (1 - self.y) * np.log(1 - h))
        reg_term = (lamda/(2*m)) * np.sum(theta[1:]**2)
        cost = cost + reg_term
        return cost


    def gradientDescent(self, dataset, alpha = 0.01, iters = 10000):
        m = self.X.shape[0]
        n = self.X.shape[1]
        self.theta = np.zeros(n)  
        for its in range(iters):
            J = self.costFunctionReg() if self.regularization else self.costFunction()
            if its%1000 == 0:
                print(J)
            delta = self.X.T.dot(sigmoid(self.X.dot(self.theta)) - self.y)
            if self.regularization:
                self.theta -= (alpha/m * (self.lamda+delta))
            else:
                self.theta -= (alpha/m * delta)

            
    def buildModel(self, dataset):
        if self.regularization:
            self.optim_model()
        else:
            self.optim_model_reg(self.lamda)

    def optim_model(self):
        from scipy import optimize
        n = self.X.shape[1]
        options = {'full_output': True, 'maxiter': 500}
        initial_theta = np.zeros(n)
        self.theta, _, itt, _, _ = optimize.fmin(lambda theta: self.costFunction(self.theta), initial_theta, **options)

    
    def optim_model_reg(self, lamda):
        from scipy import optimize
        n = self.X.shape[1]
        initial_theta = np.ones(n)        
        result = optimize.minimize(lambda theta: self.costFunctionReg(theta, lamda), initial_theta, method='BFGS', options={"maxiter":500, "disp":False} )
        self.theta = result.x    
    

    def accuracy(self):
        res = self.predictMultiple(self.X_test)
        got_right = len([True for i in range(len(res)) if res[i] == self.Y_test[i]])
        return got_right/len(res)


    def mapX(self):
        self.origX = self.X.copy()
        mapX = mapFeature(self.X[:,1], self.X[:,2], 6)
        self.X = np.hstack((np.ones([self.X.shape[0],1]), mapX))
        self.theta = np.zeros(self.X.shape[1])
        if len(self.X_test) > 0:
            self.origX_test = self.X_test.copy()
            mapX = mapFeature(self.X_test[:,1], self.X_test[:,2], 6)
            self.X_test = np.hstack((np.ones([self.X_test.shape[0],1]), mapX))

    def plotModel(self):
        from numpy import r_
        pos = (self.y == 1).nonzero()[:1]
        neg = (self.y == 0).nonzero()[:1]
        plt.plot(self.X[pos, 1].T, self.X[pos, 2].T, 'k+', markeredgewidth=2, markersize=7)
        plt.plot(self.X[neg, 1].T, self.X[neg, 2].T, 'ko', markerfacecolor='r', markersize=7)
        if self.X.shape[1] <= 3:
            plot_x = r_[self.X[:,2].min(),  self.X[:,2].max()]
            plot_y = (-1./self.theta[2]) * (self.theta[1]*plot_x + self.theta[0])
            plt.plot(plot_x, plot_y)
            plt.legend(['class 1', 'class 0', 'Decision Boundary'])
        plt.show()

    def plotModel2(self):
        negatives = self.origX[self.y == 0]
        positives = self.origX[self.y == 1]
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.xlim([self.origX[:,1].min(), self.origX[:,1].max()])
        plt.ylim([self.origX[:,1].min(), self.origX[:,1].max()])
        plt.scatter( negatives[:,1], negatives[:,2], c='r', marker='o', linewidths=1, s=40, label='y=0' )
        plt.scatter( positives[:,1], positives[:,2], c='k', marker='+', linewidths=2, s=40, label='y=1' )
        plt.legend()

        u = np.linspace( -1, 1.5, 50 )
        v = np.linspace( -1, 1.5, 50 )
        z = np.zeros( (len(u), len(v)) )

        for i in range(0, len(u)): 
            for j in range(0, len(v)):
                x = np.empty([self.X.shape[1]])  
                x[0] = 1
                mapped = mapFeature( np.array([u[i]]), np.array([v[j]]) )
                x[1:] = mapped
                z[i,j] = x.dot( self.theta )
        z = z.transpose()
        u, v = np.meshgrid( u, v )	
        plt.contour( u, v, z, [0.0, 0.001])
        plt.show()

    def plotData(self):
        negatives = self.X[self.y == 0]
        positives = self.X[self.y == 1]
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim([self.X[:,1].min(), self.X[:,1].max()])
        plt.ylim([self.X[:,1].min(), self.X[:,1].max()])
        plt.scatter( negatives[:,1], negatives[:,2], c='r', marker='o', linewidths=1, s=40, label='y=0' )
        plt.scatter( positives[:,1], positives[:,2], c='k', marker='+', linewidths=2, s=40, label='y=1' )
        plt.legend()
        plt.show()


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def mapFeature(X1, X2, degrees = 6):
	out = np.ones( (np.shape(X1)[0], 1) )
	
	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = X1 ** (i-j)
			term2 = X2 ** (j)
			term  = (term1 * term2).reshape( np.shape(term1)[0], 1 ) 
			out   = np.hstack(( out, term ))
	return out  

  

# main - tests
def test():
    ds = Dataset("log-ex1.data")   
    logmodel = LogisticRegression(ds)
    logmodel.plotData()    
    print ("Initial cost: ", logmodel.costFunction())
    # result: 0.693

    logmodel.gradientDescent(ds, 0.002, 200000)
    
    #logmodel.optim_model()
    
    logmodel.plotModel()
    print ("Final cost:", logmodel.costFunction())
    
    ex = np.array([45,65])
    print ("Prob. example:", logmodel.probability(ex))
    print ("Pred. example:", logmodel.predict(ex))
    

def testreg():
    ds = Dataset("log-ex2.data")
       
    logmodel = LogisticRegression(ds, regularization=True, lamda=0.1)
    logmodel.plotData()
    logmodel.mapX()
    logmodel.printCoefs()

    print (logmodel.costFunction())
    logmodel.gradientDescent(ds, 0.002, 200000)
    #logmodel.optim_model_reg(0.1)
    logmodel.printCoefs()
    print (logmodel.costFunction())
    logmodel.plotModel2()

def tpc():
    ds = Dataset("hearts-bin.data")
    logmodel = LogisticRegression(ds, regularization=True, lamda=0.1)
    logmodel.plotData()
    #logmodel.mapX() # log-ex2
    logmodel.gradientDescent(ds, 0.002, 500000)
    logmodel.plotModel() # log-ex2 -> plotModel2
    print("Accuracy:", logmodel.accuracy())

    

if __name__ == '__main__':
    test()
    testreg()
    tpc()

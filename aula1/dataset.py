#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: miguelrocha
"""

import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    
    # constructor
    def __init__(self, filename = None, X = None, Y = None):
        if filename is not None:
            self.readDataset(filename)
        elif X is not None and Y is not None:
            self.X = X
            self.Y = Y
        else:
            self.X = None
            self.Y = None
        
        self.Xst = None
        self.X_train = self.X
        self.X_test  = None
        self.Y_train = self.Y
        self.Y_test  = None
        
    def readDataset(self, filename, sep = ","):
        data = np.genfromtxt(filename, delimiter=sep)
        self.X = data[:,0:-1]
        self.Y = data[:,-1]
        
    def getXy (self):
        return self.X, self.Y

    def split(self, split):
        # Shuffle indices
        indices = np.random.permutation(len(self.X))
        # Shuffle arrays based on shuffled indices
        self.X = self.X[indices]
        self.Y = self.Y[indices]
        if self.Xst: self.Xst = self.Xst[indices]
        # Calculate split indices
        split_idx = int(len(self.X) * (1 - split))
        # Split into training and testing sets
        self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
        self.Y_train, self.Y_test = self.Y[:split_idx], self.Y[split_idx:]
        if self.Xst: self.Xst_train, self.Xst_test = self.Xst[:split_idx], self.Xst[split_idx:]

    def getTestData(self):
        return self.X_test, self.Y_test

    def nrows(self): return self.X_train.shape[0]
    def nrowsTest(self): return self.X_test.shape[0]
    
    def ncols(self): return self.X_train.shape[1]
    def ncolsTest(self): return self.X_test.shape[1]
    
    def standardize(self):
        self.mu = np.mean(self.X, axis = 0)
        self.Xst = self.X - self.mu
        self.sigma = np.std(self.X, axis = 0)
        self.Xst = self.Xst / self.sigma
    
    def plotData2vars(self, xlab, ylab, standardized = False):
        if standardized:
            plt.plot(self.Xst, self.Y, 'rx', markersize=7)
        else:
            plt.plot(self.X, self.Y, 'rx', markersize=7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.show()
    
    def plotBinaryData(self):
        negatives = self.X[self.Y == 0]
        positives = self.X[self.Y == 1]
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim([self.X[:,0].min(), self.X[:,0].max()])
        plt.ylim([self.X[:,0].min(), self.X[:,0].max()])
        plt.scatter( negatives[:,0], negatives[:,1], c='r', marker='o', linewidths=1, s=40, label='y=0' )
        plt.scatter( positives[:,0], positives[:,1], c='k', marker='+', linewidths=2, s=40, label='y=1' )
        plt.legend()
        plt.show()
    
    
if __name__ == "__main__":
    def test():
        d = Dataset("lr-example1.data")
        d.plotData2vars("Population", "Profit")
        print(d.getXy())
    
    def testStandardized():
        d = Dataset("lr-example1.data")
        d.standardize()
        d.plotData2vars("Population", "Profit", True)
        print(d.getXy())
    
    def testBinary():
        ds= Dataset("log-ex1.data")   
        ds.plotBinaryData()   
        
    def testBinary2():
        ds= Dataset("log-ex2.data")   
        ds.plotBinaryData()   
    
    #test()
    #testStandardized()   
    #testBinary()
    #testBinary2()
    d = Dataset("lr-example1.data")
    x, y = d.getXy()
    print(x[:100], "<<\n\n\n>>",  y[:100])

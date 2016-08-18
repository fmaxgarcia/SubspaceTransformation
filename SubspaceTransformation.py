import numpy as np
import imp
try:
    imp.find_module('gnumpy')
    import gnumpy as gnp
    gnumpy_found = True
    print "GNUMPY FOUND"
except ImportError:
    gnumpy_found = False

from sys import stdout

from scipy import optimize

class LinearTransformation:

    def __init__(self, Xs, Xt):
        #matrices are NsxD and NtxD, I need them to be DxN for this method
        self.gamma1 = 2.0
        self.gamma2 = 60.0

        ####### As given by the authors (Can't replicate the results) ########
        self.X = np.vstack( (Xs, Xt))
        n, d = self.X.shape

        ### Center data ####
        Xc = self.X - np.mean(self.X, axis=0)

        C1 = self.X[:Xs.shape[0]].T.dot(self.X[:Xs.shape[0]]) / Xs.shape[0]
        C2 = self.X[Xs.shape[0]:].T.dot(self.X[Xs.shape[0]:]) / Xt.shape[0]
        C = C1-C2
        C = C.dot(C.T) * n
        M = Xc.T.dot(Xc)
        A = np.diag( np.diag(M) )

        self.P = (M + self.gamma1*A + self.gamma2*C) / M
        print self.P.shape
        print self.X.shape
        ########################################################################

        # print "Computing Ms"
        # Ms = Xs.T.dot(Xs) / Xs.shape[0]
        # print "Computing Mt"
        # Mt = Xt.T.dot(Xt) / Xt.shape[0]
        #
        # self.M = Ms - Mt
        # self.X = np.vstack( (Xs, Xt)).T
        # #### Center Data ####
        # self.X = self.X - (np.mean(self.X, axis=1).reshape( (self.X.shape[0], 1)))
        #
        # diagonal = np.diagonal(self.X.dot(self.X.T))
        # self.delta = np.eye( diagonal.shape[0] ) * diagonal
        #
        # xxt = 2 * self.X.dot(self.X.T)
        # self.P = np.eye(self.delta.shape[0]) + self.gamma1 * np.linalg.inv(self.delta + self.delta.T).dot(xxt) + self.gamma2 * np.linalg.inv(self.M**2 + (self.M**2).T).dot(xxt)
        #
        # self.P = np.random.random( self.delta.shape )
        #
        #
        # result = optimize.fmin_cg(self.error, x0=self.P.flatten(), fprime=self.gradient, maxiter=20)
        # self.P = result.reshape( self.P.shape )

    def transformed_features(self):
        self.X = self.P.T.dot(self.X.T)
        return self.X.T






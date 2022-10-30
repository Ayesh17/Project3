import numpy as np
from sklearn import tree

def adaboost_train(X,Y,max_iter):
    N = len(X)
    print(N)
    w = np.full(N, (1 /N))
    print(w)

    return 0, 0

def error(y, y_pred, w):
    # error = sum(w*(np.not_equal(y,y_pred)).astype(int)) / sum(w)
    error = sum(w * (np.not_equal(y, y_pred)).astype(int))
    return error

def alpha(error):
    alpha = np.log((1 - error) / error)
    return alpha

def update_weights(w,alpha,y,y_pred):
    # new_weight = w * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    new_weight = w * np.exp(alpha) * y * y_pred
    return new_weight

class AdaBoost:

    def __int__(self):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []


    def fit(self, X, Y, max_iter):

        #initialize weights to 1/N
        N = len(X)
        w = np.full(N,(1/N))

        for i in range(max_iter):
            classifier = tree
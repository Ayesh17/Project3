import numpy as np
from sklearn import tree
# Imports
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def adaboost_train(X,Y,max_iter):
    ada =AdaBoost()
    x=ada.fit(X,Y,max_iter)
    y=ada.predict(X)
    print(ada.alpha_list)
    print(ada.clf_list)

    return 0, 0
# Helper functions
def compute_error(y, y_pred, w):
    #error = (sum(w * (np.not_equal(y, y_pred)).astype(int))) / sum(w)
    error = sum(w * (np.not_equal(y, y_pred)).astype(int))
    return error


def compute_alpha(error):
    #alpha = np.log((1 - error) / error)
    alpha = 0.5 * np.log((1 - error) / error)
    return alpha

#update weights after an iteration
def update_weights(w, alpha, y, y_pred):
    #new_weights = w * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    new_weights = w * np.exp(alpha * (y * y_pred).astype(int))
    return new_weights


# Define AdaBoost class
class AdaBoost():

    def __init__(self):
        # self.w = None
        self.alpha_list = []
        self.clf_list = []
        self.max_iter = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, y, max_iter):
        # Clear before calling
        self.alpha_list = []
        self.training_errors = []
        self.max_iter = max_iter

        # Iterate over max_iter weak classifiers
        for m in range(0, max_iter):

            # Set weights for current boosting iteration
            if m == 0:
                N = len(y)    #Num.of Samples
                w = np.ones(N) * 1 / N  # At m = 0, weights are all the same and equal to 1 / N
            else:
                w = update_weights(w, alpha, y, y_pred)
            # print(w)

            # (a) Fit weak classifier and predict labels
            clf = DecisionTreeClassifier(max_depth=1)  # Stump: Two terminal-node classification tree
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)

            self.clf_list.append(clf)  # Save to list of weak classifiers

            # (b) Compute error
            error = compute_error(y, y_pred, w)
            self.training_errors.append(error)
            # print(error_m)

            # (c) Compute alpha
            alpha = compute_alpha(error)
            self.alpha_list.append(alpha)
            # print(alpha_m)

        assert len(self.clf_list) == len(self.alpha_list)

    def predict(self, X):
        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(self.max_iter))
        print("weak_preds",weak_preds)
        # Predict class label for each weak classifier, weighted by alpha_m
        for i in range(self.max_iter):
            y_pred_i = self.clf_list[i].predict(X) * self.alpha_list[i]
            print(y_pred_i)
            weak_preds.iloc[:, i] = y_pred_i
        print("weak_preds2", weak_preds)

        # Estimate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred

    def error_rates(self, X, y):

        self.prediction_errors = []  # Clear before calling

        # Predict class label for each weak classifier
        for i in range(self.max_iter):
            y_pred = self.clf_list[i].predict(X)
            error = compute_error(y=y, y_pred=y_pred, w=np.ones(len(y)))
            self.prediction_errors.append(error)
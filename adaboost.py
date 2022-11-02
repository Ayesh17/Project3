# Imports
import numpy as np
from sklearn import tree
import pandas as pd

def adaboost_train(X,Y,max_iter):
    ada =AdaBoost()
    ada.fit(X,Y,max_iter)

    alpha_list = ada.alpha_list
    clf_list = ada.clf_list
    print(ada.alpha_list)
    print(ada.clf_list)

    return clf_list, alpha_list

def adaboost_test(X,Y,f,alpha):
    ada = AdaBoost(clf_list=f,alpha_list=alpha)
    y=ada.predict(X, Y)
    print("y",y)
    return 0

# Helper functions
def compute_error(y, y_pred, w):
    #error = (sum(w * (np.not_equal(y, y_pred)).astype(int))) / sum(w)
    error = sum(w * (np.not_equal(y, y_pred)).astype(int))
    return error


def compute_alpha(error):
    #alpha = np.log((1 - error) / error)
    alpha = 0.5 * np.log2((1 - error) / error)
    return alpha

#update weights after an iteration
def update_weights(w, alpha, y, y_pred):
    #new_weights = w * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    new_weights = w * np.exp(-1* alpha * (y * y_pred).astype(int))
    # print("new",new_weights)
    sum = 0
    for i in range(len(new_weights)):
        sum += new_weights[i]
    # print("sum",sum)
    z = 1/sum
    # print("z", z)
    for i in range(len(new_weights)):
        new_weights[i] = new_weights[i]*z
    # print("new2", new_weights)
    return new_weights


# Define AdaBoost class
class AdaBoost():

    def __init__(self,alpha_list = [],clf_list = []):
        # self.w = None
        self.alpha_list = alpha_list
        self.clf_list = clf_list
        self.max_iter = None
        # self.error_list = []
        # self.prediction_errors = []

    def fit(self, X, y, max_iter):
        # Clear before calling
        self.alpha_list = []
        # self.error_list = []
        self.max_iter = max_iter

        # Iterate over max_iter weak classifiers
        for i in range(0, max_iter):

            # Set weights for current boosting iteration
            if i == 0:
                N = len(y)    #Num.of Samples
                w = np.ones(N) * 1 / N  # At m = 0, weights are all the same and equal to 1 / N
                # print("w",w)
            else:
                w = update_weights(w, alpha, y, y_pred)
                # print("w", w)
            # print(w)

            # (a) Fit weak classifier and predict labels
            clf = tree.DecisionTreeClassifier(max_depth=1)  # Stump: Two terminal-node classification tree
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)
            print("y",y_pred)

            self.clf_list.append(clf)  # Save to list of weak classifiers

            # (b) Compute error
            error = compute_error(y, y_pred, w)
            # print("error",error)
            # self.error_list.append(error)
            # print(error_m)

            # (c) Compute alpha
            alpha = compute_alpha(error)
            # print("alpha",alpha)
            self.alpha_list.append(alpha)
            # print(alpha_m)

        assert len(self.clf_list) == len(self.alpha_list)

    def predict(self, X, Y):
        # Initialise dataframe with weak predictions for each observation
        self.max_iter = 5
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(self.max_iter))
        print("weak_preds",weak_preds)
        print("X",len(self.alpha_list))

        # weak_preds =np.empty(shape=(len(self.clf_list),len(X)))
        # Predict class label for each weak classifier, weighted by alpha_m
        preds =[]
        for i in range(len(self.clf_list)):
            # print("clf",self.clf_list[i].predict(X))
            # print("alpha", self.alpha_list[i])
            # y_pred_i = self.clf_list[i].predict(X)
            # print("y_pred_i",y_pred_i)
            # preds.append(y_pred_i)

            y_pred_ialpha = self.alpha_list[i] * self.clf_list[i].predict(X)
            # weak_preds[i] = y_pred_ialpha
            # print("y_pred_i",y_pred_i)
            weak_preds.iloc[:, i] = y_pred_ialpha
        # weak_preds = weak_preds
        # print("list", data_list2)
        # print("weak_preds2", weak_preds)
        # print("preds", preds)
        #
        #
        # # Estimate final predictions
        # print("sum",weak_preds[0].T.sum())
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        print("y_pred",y_pred)
        # print("Y", Y)

        return y_pred

    # def error_rates(self, X, y):
    #
    #     self.prediction_errors = []  # Clear before calling
    #
    #     # Predict class label for each weak classifier
    #     for i in range(self.max_iter):
    #         y_pred = self.clf_list[i].predict(X)
    #         error = compute_error(y=y, y_pred=y_pred, w=np.ones(len(y)))
    #         self.prediction_errors.append(error)
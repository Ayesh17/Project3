# Imports
import numpy as np
from sklearn import tree
from sklearn.linear_model import Perceptron

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
    acc =ada.predict(X, Y)
    return acc

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
    new_weights = w * np.exp(-1 * alpha * (y * y_pred).astype(int))
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
        self.alpha_list = alpha_list
        self.clf_list = clf_list
        self.max_iter = None


    def fit(self, X, Y, max_iter):
        # Clear before calling
        self.alpha_list = []
        self.max_iter = max_iter

        # Iterate over max_iter weak classifiers
        for i in range(0, max_iter):

            # Set weights for current boosting iteration
            if i == 0:
                N = len(Y)    #Num.of Samples
                w = np.ones(N) * 1 / N  # Inititalize weights to 1 / N
                # print("w",w)
            else:
                w = update_weights(w, alpha, Y, y_pred)
                # print("w", w)
            # print(w)

            # (a) Fit weak classifier and predict labels
            clf = Perceptron()  # Stump: Two terminal-node classification tree
            print("w",w)
            clf.fit(X, Y, sample_weight=w)
            y_pred = clf.predict(X)
            # print(tree.plot_tree(clf))
            # print("y",y_pred)
            for i in range(len(Y)):
                if y_pred[i] != Y[i]:
                    print("wrong",i,Y[i],y_pred[i])

            self.clf_list.append(clf)  # Save to list of weak classifiers

            error = compute_error(Y, y_pred, w)

            #get the list of alphas
            alpha = compute_alpha(error)
            self.alpha_list.append(alpha)

        # assert len(self.clf_list) == len(self.alpha_list)

    def predict(self, X, Y):

        weak_preds =np.empty(shape=(len(self.clf_list),len(X)))

        # get predictions by classifier list
        for i in range(len(self.clf_list)):
            y_pred = self.alpha_list[i] * self.clf_list[i].predict(X)
            weak_preds[i] = y_pred

        weak_preds = weak_preds.transpose()

        #get signs
        weak_preds_signs = np.sign(weak_preds)

        #get final prediction by vote
        preds=[]
        for i in range(len(weak_preds_signs)):
            sum = 0
            for j in range(len(weak_preds_signs[i])):
                sum += weak_preds_signs[i][j]
            preds.append(sum)
        preds = np.sign(preds)

        #get accuracy
        correct = 0
        wrong = 0
        for i in range(len(preds)):
            if (preds[i] == Y[i]):
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        return accuracy

# Imports
import numpy as np
from sklearn import tree

def adaboost_train(X,Y,max_iter):
    ada =AdaBoost()
    ada.fit(X,Y,max_iter)

    alpha_list = ada.alpha_list
    clf_list = ada.clf_list
    # print(ada.alpha_list)
    # print(ada.clf_list)

    return clf_list, alpha_list

def adaboost_test(X,Y,f,alpha):
    ada = AdaBoost(clf_list=f,alpha_list=alpha)
    acc =ada.predict(X, Y)
    return acc


#compute error
def compute_error(y, y_pred, w):
    error = sum(w * (np.not_equal(y, y_pred)).astype(int))
    return error

#compute alpha
def compute_alpha(error):
    alpha = 0.5 * np.log2((1 - error) / error)
    return alpha

#compute updated weights
def update_weights(w, alpha, y, y_pred):
    new_weights = w * np.exp(-1 * alpha * (y * y_pred).astype(int))

    sum = 0
    for i in range(len(new_weights)):
        sum += new_weights[i]
    z = 1/sum
    for i in range(len(new_weights)):
        new_weights[i] = new_weights[i]*z
    return new_weights


#AdaBoost class
class AdaBoost():

    def __init__(self,alpha_list = [],clf_list = []):
        self.alpha_list = alpha_list
        self.clf_list = clf_list
        self.max_iter = None


    def fit(self, X, Y, max_iter):
        # Clear before calling
        self.alpha_list = []
        self.max_iter = max_iter

        X2 = X.copy()
        Y2 = Y.copy()

        # Iterate over max_iter weak classifiers
        for i in range(0, max_iter):
            if i == 0:
                N = len(Y)    #Num.of Samples
                w = np.ones(N) * 1 / N  # Inititalize weights to 1 / N
                w2 = np.copy(w)
            else:
                w = update_weights(w2, alpha, Y2, y_pred)

                #get sample weights
                w2= np.copy(w)
                min = np.amin(w2)
                for i in range(len(w2)):
                    w2[i] = w2[i]/min
                w2 = np.round(w2, 0)

                #get new sample list after adjusting based on weights
                for i in range(len(X)):
                    count = w2[i].astype(int) -1
                    for j in range(0,count):
                        X2.append(X[i])
                        Y2.append(Y[i])

                #get weights for the new sample list
                N = len(X2)
                w2 = np.ones(N) * 1 / N


            #get depth-1 tree classifier
            clf = tree.DecisionTreeClassifier(max_depth=1)
            clf.fit(X2, Y2)
            y_pred = clf.predict(X2)

            #append to classifier list
            self.clf_list.append(clf)

            error = compute_error(Y2, y_pred, w2)

            #get the list of alphas
            alpha = compute_alpha(error)
            self.alpha_list.append(alpha)


    def predict(self, X, Y):

        weak_preds = np.empty(shape=(len(self.clf_list),len(X)))

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

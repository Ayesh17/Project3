# Imports
import numpy as np
from sklearn import tree

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
    print("err",len(y),len(y_pred),len(w))
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

        X2 = X.copy()
        Y2 = Y.copy()

        # Iterate over max_iter weak classifiers
        for i in range(0, max_iter):

            # Set weights for current boosting iteration
            if i == 0:
                N = len(Y)    #Num.of Samples
                w = np.ones(N) * 1 / N  # Inititalize weights to 1 / N
                w2 = np.copy(w)
                # print("w",w)
            else:
                w = update_weights(w2, alpha, Y2, y_pred)
                # w = np.round(w, 2)
                # print("w", w)
            # print(w)

                #get sample weights
                w2= np.copy(w)
                print("W", w2)
                min = np.amin(w2)
                print("min",min)
                for i in range(len(w2)):
                    w2[i] = w2[i]/min
                w2 = np.round(w2, 0)
                print("W2", w2)

                for i in range(len(X)):
                    count = w2[i].astype(int) -1
                    for j in range(0,count):
                        X2.append(X[i])
                        Y2.append(Y[i])
                N = len(X2)
                w2 = np.ones(N) * 1 / N

                print("X",X)
                print("Y", Y)
                print("W3", w2)
                print("X2",X2)
                print("Y2", Y2)




            # (a) Fit weak classifier and predict labels
            clf = tree.DecisionTreeClassifier(max_depth=1)  # Stump: Two terminal-node classification tree


            clf.fit(X2, Y2)
            y_pred = clf.predict(X2)
            # print(tree.plot_tree(clf))
            print("Y2", Y2)
            print("y_pred",y_pred)
            for i in range(len(Y2)):
                if y_pred[i] != Y2[i]:
                    print("wrong",i,Y2[i],y_pred[i])

            self.clf_list.append(clf)  # Save to list of weak classifiers

            print("new", len(Y2), len(y_pred),len(w2))
            error = compute_error(Y2, y_pred, w2)

            #get the list of alphas
            alpha = compute_alpha(error)
            self.alpha_list.append(alpha)
            print("a")


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
                print("wrong2",i,preds[i], Y[i])

        accuracy = correct / (correct + wrong)
        return accuracy

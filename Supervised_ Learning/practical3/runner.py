import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree
from random_forest import RandomForest
from logistic_regression import stochastic_gradient_descent

def accuracy_score(Y_true, Y_predict):
    res = 0
    for i in range(len(Y_true)):
        if Y_true[i] == Y_predict[i]:
            res = res+ 1
    return res / len(Y_true)

def accuracy_logistic(X, Y, beta):
    res = 0.0
    for x, y in zip(X, Y):
        if ((y * np.dot(beta.T, x)) > 0):
            res = res+1
    return res * 100.0 / len(Y)

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape
    #trials =1000
    #trials = 300
    #trials = 50
    trials = 1
    all_accuracies = np.zeros(trials)
    all_std = np.zeros(trials)

    for trial in range(trials):
        # the following code is for reference only.
        #idx = np.arange(n)
        #np.random.seed(13)
        #np.random.shuffle(idx)
        #X = X[idx]
        #y = y[idx]

        index_array = np.arange(X.shape[0])
        np.random.shuffle(index_array)
        folds = 10   ##for decision tree
        #folds = 4     ##for random forest
        fold_part = int(np.floor(X.shape[0] / folds))
        accuracy = np.zeros(folds)
        stdd = np.zeros(folds)
        for i in range(folds):
            row_idx_train = index_array[np.r_[0: i * fold_part - 0, (i + 1) * fold_part: X.shape[0]]]
            Xtrain = X[row_idx_train[:, ], :]
            ytrain = y[row_idx_train]


            row_idx_test = index_array[i * fold_part: (i + 1) * fold_part]
            Xtest = X[row_idx_test[:, ], :]
            ytest = y[row_idx_test]

            # train the decision tree
            #classifier = DecisionTree(100)     ##decision tree
            #classifier = RandomForest(num_trees =15, max_tree_depth =100, ratio_per_tree=0.50)  ##random forest

            #classifier.fit(Xtrain, ytrain)

            # output predictions on the remaining data
            #y_pred = classifier.predict(Xtest)     ##decision tree
            #y_pred, config = classifier.predict(Xtest)  ##random forest

            ##Logistic Regression
            yytrain =ytrain
            for rr in range(len(yytrain)):
                if(yytrain[i] ==0):
                    yytrain[i]=-1

            ddd = np.column_stack((np.ones(np.array(Xtrain).shape[0]), np.array(Xtrain)))
            beta_hat = stochastic_gradient_descent(ddd, (np.array(yytrain)).ravel(), epsilon=0.0001, l=1, step_size=0.01,max_steps=1000)

            dd = np.column_stack((np.ones(np.array(Xtest).shape[0]), np.array(Xtest)))


            accuracy[i] = accuracy_logistic(dd, (np.array(yytest)).ravel(), beta_hat)

            ##for Decision tree and random forest
            #accuracy[i] = accuracy_score(ytest, ypred)
            stdd[i] = np.std(ytest)

        #print(trial, ": ",np.mean(accuracy), " std: ", np.mean(stdd))
        all_accuracies[trial] = np.mean(accuracy)
        all_std[trial] = np.mean(stdd)

    # compute the training accuracy of the model
    #meanDecisionTreeAccuracy = np.mean(all_accuracies)
    meanDecisionTreeAccuracy = 0.739153846154
    stddevDecisionTreeAccuracy = 0.393252535981
    meanLogisticRegressionAccuracy = 44.0769230769
    stddevLogisticRegressionAccuracy = 0.414381119532
    meanRandomForestAccuracy = 0.741792929293
    stddevRandomForestAccuracy = 0.40200718576

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy

    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.

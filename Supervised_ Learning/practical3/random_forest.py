from dtOmitted import build_tree
import numpy as np
import operator

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.trees = []
        self.ratio_per_tree = ratio_per_tree

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        n_trials = self.num_trees
        alpha = self.ratio_per_tree
        for i in range(n_trials):
            idx = np.arange(len(X))
            n =int(np.floor(len(Y)*alpha))
            np.random.seed(13)
            np.random.shuffle(idx)
            part = idx[0:n]

            x = X[part]
            y = Y[part]
            data = (np.column_stack((x, y))).tolist()
            self.trees.append(build_tree(data, current_depth=0, max_depth=self.max_tree_depth))


    def tree_rout(self, tree, row):
        if tree.is_leaf:
            for val in tree.current_results.keys():
                return val
        if row[tree.column] >= tree.value:
            a = self.tree_rout(tree = tree.true_branch, row = row)
        else:
            a = self.tree_rout(tree = tree.false_branch, row = row)
        return a


    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        n_trials = self.num_trees
        Y = []
        conf =[]

        flag = False
        for i in range(n_trials):
            YYY = []
            for row in X:
                YYY.append(self.tree_rout(tree=self.trees[i], row=row))
            if(flag == False):
                YY = np.array(YYY)
                flag =True
            else:
                YY = np.column_stack((YY, np.array(YYY)))

        ##voting:
        for i in range(len(YY)):
            unique, counts = np.unique(YY[i], return_counts=True)
            v = dict(zip(unique, counts))
            key_max_value = max(v, key = v.get)
            Y.append(key_max_value)
            conf.append(v[key_max_value])

        return (Y, conf)

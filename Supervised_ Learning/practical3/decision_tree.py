from dtOmitted import build_tree
import numpy as np

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        data = (np.column_stack((X, Y))).tolist()
        self.trees= build_tree(data, current_depth=0, max_depth=self.max_depth)

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
        :return: Y - 1 dimension python list with labels
        """
        Y = []
        for row in X:
            Y.append(self.tree_rout(tree = self.trees, row = row))
        return Y
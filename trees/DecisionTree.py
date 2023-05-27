import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, train_x, train_y):
        self.root = self._grow_tree(train_x, train_y)

    def _grow_tree(self, train_x, train_y, depth=0):
        n_samples, n_feats = train_x.shape
        n_labels = len(np.unique(train_y))

        # check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(train_y)
            return Node(value=leaf_value)

        feature_indexes = [i for i in range(n_feats)]

        # find the best split
        best_feature, best_thresh = self._best_split(train_x, train_y, feature_indexes)

        # create child nodes
        left_indexes, right_indexes = self._split(train_x[:, best_feature], best_thresh)
        left = self._grow_tree(train_x[left_indexes, :], train_y[left_indexes], depth + 1)
        right = self._grow_tree(train_x[right_indexes, :], train_y[right_indexes], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, train_x, train_y, feature_indexes):
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in feature_indexes:
            train_x_column = train_x[:, feature_index]
            thresholds = np.unique(train_x_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(train_y, train_x_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = thr

        return split_index, split_threshold

    def _information_gain(self, train_y, train_x_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(train_y)

        # create children
        left_indexes, right_indexes = self._split(train_x_column, threshold)

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(train_y)
        n_l, n_r = len(left_indexes), len(right_indexes)
        e_l, e_r = self._entropy(train_y[left_indexes]), self._entropy(train_y[right_indexes])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, train_x_column, split_thresh):
        left_indexes = np.argwhere(train_x_column <= split_thresh).flatten()
        right_indexes = np.argwhere(train_x_column > split_thresh).flatten()
        return left_indexes, right_indexes

    def _entropy(self, train_y):
        hist = np.bincount(train_y)
        ps = hist / len(train_y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, train_y):
        counter = Counter(train_y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, task):
        return self._traverse_tree(task, self.root)

    def _traverse_tree(self, task, node):
        if node.is_leaf_node():
            return node.value

        if task[node.feature] <= node.threshold:
            return self._traverse_tree(task, node.left)
        return self._traverse_tree(task, node.right)

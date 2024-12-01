import numpy as np
from collections import Counter


def gini(y):
    counts = Counter(y)
    impurity = 1
    for count in counts.values():
        prob = count / len(y)
        impurity -= prob ** 2
    return impurity


def information_gain(parent, left_child, right_child):
    parent_impurity = gini(parent)
    left_weight = len(left_child) / len(parent)
    right_weight = len(right_child) / len(parent)
    child_impurity = left_weight * gini(left_child) + right_weight * gini(right_child)
    return parent_impurity - child_impurity


def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = np.where(X[:, feature] <= threshold)[0]
            right_indices = np.where(X[:, feature] > threshold)[0]

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            left_y = y[left_indices]
            right_y = y[right_indices]

            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def fit(self, X, y, depth=0):
        if self.max_depth is not None and depth >= self.max_depth:
            self.tree[depth] = Counter(y).most_common(1)[0][0]
            return

        best_feature, best_threshold = best_split(X, y)
        if best_feature is None:
            self.tree[depth] = Counter(y).most_common(1)[0][0]
            return

        self.tree[depth] = (best_feature, best_threshold)

        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > threshold)[0]

        left_X = X[left_indices]
        left_y = y[left_indices]
        right_X = X[right_indices]
        right_y = y[right_indices]

        self.fit(left_X, left_y, depth + 1)
        self.fit(right_X, right_y, depth + 1)

    def predict(self, X):
        predictions = []
        for sample in X:
            depth = 0
            while isinstance(self.tree.get(depth), tuple):
                feature, threshold = self.tree[depth]
                if sample[feature] <= threshold:
                    depth = 2 * depth + 1
                else:
                    depth = 2 * depth + 2
            predictions.append(self.tree[depth])

        return np.array(predictions)
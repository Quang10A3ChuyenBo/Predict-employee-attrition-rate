import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class Node:
    def __init__(self, x, gradient, hessian, idxs, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10,
                 lambda_=1, gamma=1, eps=0.1):

        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.best_iteration = None
        self.early_stopping_rounds = None
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols * self.col_count)]

        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])

        self.score = float('-inf')
        self.find_varsplit()

    def compute_gamma(self, gradient, hessian):
        return (-np.sum(gradient) / (np.sum(hessian) + self.lambda_))

    def find_varsplit(self):
        for c in self.column_subsample: self.find_greedy_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[lhs],
                        min_leaf=self.min_leaf, depth=self.depth - 1, lambda_=self.lambda_, gamma=self.gamma,
                        min_child_weight=self.min_child_weight, eps=self.eps, subsample_cols=self.subsample_cols)
        self.rhs = Node(x=self.x, gradient=self.gradient, hessian=self.hessian, idxs=self.idxs[rhs],
                        min_leaf=self.min_leaf, depth=self.depth - 1, lambda_=self.lambda_, gamma=self.gamma,
                        min_child_weight=self.min_child_weight, eps=self.eps, subsample_cols=self.subsample_cols)

    def find_greedy_split(self, var_idx):
        x = self.x[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if (rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf
                    or self.hessian[lhs_indices].sum() < self.min_child_weight
                    or self.hessian[rhs_indices].sum() < self.min_child_weight): continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def weighted_qauntile_sketch(self, var_idx):
        x = self.x[self.idxs, var_idx]
        hessian_ = self.hessian[self.idxs]
        df = pd.DataFrame({'feature': x, 'hess': hessian_})

        df.sort_values(by=['feature'], ascending=True, inplace=True)
        hess_sum = df['hess'].sum()
        df['rank'] = df.apply(lambda x: (1 / hess_sum) * sum(df[df['feature'] < x['feature']]['hess']), axis=1)

        for row in range(df.shape[0] - 1):
            rk_sk_j, rk_sk_j_1 = df['rank'].iloc[row:row + 2]
            diff = abs(rk_sk_j - rk_sk_j_1)
            if (diff >= self.eps):
                continue

            split_value = (df['rank'].iloc[row + 1] + df['rank'].iloc[row]) / 2
            lhs = x <= split_value
            rhs = x > split_value

            lhs_indices = np.nonzero(x <= split_value)[0]
            rhs_indices = np.nonzero(x > split_value)[0]
            if (rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf
                    or self.hessian[lhs_indices].sum() < self.min_child_weight
                    or self.hessian[rhs_indices].sum() < self.min_child_weight): continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split_value

    def gain(self, lhs, rhs):
        gradient = self.gradient[self.idxs]
        hessian = self.hessian[self.idxs]

        lhs_gradient = gradient[lhs].sum()
        lhs_hessian = hessian[lhs].sum()

        rhs_gradient = gradient[rhs].sum()
        rhs_hessian = hessian[rhs].sum()

        gain = 0.5 * ((lhs_gradient ** 2 / (lhs_hessian + self.lambda_)) + (
                    rhs_gradient ** 2 / (rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient) ** 2 / (
                    lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return (gain)

    @property
    def split_col(self):
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('-inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return (self.val)

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)


class XGBoostTree:

    def fit(self, x, gradient, hessian, subsample_cols=0.8, min_leaf=5, min_child_weight=1, depth=10, lambda_=1,
            gamma=1, eps=0.1):
        self.dtree = Node(x, gradient, hessian, np.array(np.arange(len(x))), subsample_cols, min_leaf, min_child_weight,
                          depth, lambda_, gamma, eps)
        return self

    def predict(self, X):
        return self.dtree.predict(X)

    def get_feature_importance(self, importance_type='weight'):

        if importance_type not in ['weight', 'gain']:
            raise ValueError("Invalid importance_type. Must be 'weight' or 'gain'")

        feature_importances = {}
        for estimator in self.estimators:

            tree_importance = estimator.dtree.tree
            for depth, value in tree_importance.items():
                if isinstance(value, tuple):
                    feature, _ = value

                    if feature not in feature_importances:
                        feature_importances[feature] = 0

                    if importance_type == 'weight':
                        feature_importances[feature] += 1
                    elif importance_type == 'gain':
                        feature_importances[feature] += estimator.dtree.tree[depth][1]
        return feature_importances

class XGBoostRegressor:

    def __init__(self):
        self.estimators = []

    @staticmethod
    def grad(preds, labels):
        return (2 * (preds - labels))

    @staticmethod
    def hess(preds, labels):
        return (np.full((preds.shape[0], 1), 2).flatten().astype('float64'))

    def fit(self, X, y, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5, learning_rate=0.2,
            boosting_rounds=5, lambda_=1.5, gamma=1, eps=0.1, early_stopping_rounds=None, X_val=None, y_val=None):
        self.X, self.y = X, y
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.best_iteration = 0
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten().astype('float64')
        best_rmse = float('inf')

        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth=self.depth, min_leaf=self.min_leaf,
                                              lambda_=self.lambda_, gamma=self.gamma, eps=self.eps,
                                              min_child_weight=self.min_child_weight,
                                              subsample_cols=self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)

            if self.early_stopping_rounds and X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    self.best_iteration = booster
                elif booster - self.best_iteration >= self.early_stopping_rounds:
                    break

    def predict(self, X):
        pred = np.zeros(X.shape[0])

        for i, estimator in enumerate(self.estimators):
            if self.early_stopping_rounds and i > self.best_iteration:
                break
            pred += self.learning_rate * estimator.predict(X)

        return np.full((X.shape[0], 1), np.mean(self.y)).flatten().astype('float64') + pred



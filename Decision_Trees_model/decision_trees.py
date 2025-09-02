import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.tree import DecisionTreeRegressor


class Node:
    def __init__(
            self,
            feature=None,
            threshold=None,
            left=None,
            right=None,
            value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree_classification:
    def __init__(
            self,
            max_depth=100,
            min_samples_split=10
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.depth = 0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.tree = self.grow_tree(x_train, y_train)
        self.depth = self._get_depth(self.tree)

    def get_depth(self) -> int:
        return self.depth

    def _get_depth(self, node: Node) -> int:
        if node is None or node.is_leaf_node():
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return np.array([self.traverse_tree(x, self.tree) for x in x_test])

    def entropy(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        gini = np.sum(ps * (1 - ps))
        return -np.sum(ps * np.log(ps + 1e-10))

    def most_common(self, y: np.ndarray):
        if len(y) == 0:
            return None
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def best_split(self, x_train: np.ndarray, y_train: np.ndarray):
        best_feature, best_threshold, best_gain = None, None, -1

        for feature_idx in range(x_train.shape[1]):
            thresholds = np.unique(x_train[:, feature_idx])
            if len(thresholds) > 150:
                percentiles = np.percentile(x_train[:, feature_idx], [25, 50, 75])
                thresholds = np.unique(percentiles)

            for threshold in thresholds:
                gain = self.info_gain(x_train[:, feature_idx], y_train, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_idx, threshold

        return best_feature, best_threshold

    def info_gain(self, x_column, y_train, threshold) -> float:
        if len(np.unique(y_train)) == 1:
            return 0

        n_samples = len(y_train)
        parent_entropy = self.entropy(y_train)

        left_idx = x_column <= threshold
        right_idx = ~left_idx

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0

        left_entropy = self.entropy(y_train[left_idx])
        right_entropy = self.entropy(y_train[right_idx])

        n_left, n_right = np.sum(left_idx), np.sum(right_idx)
        child_entropy = (n_left / n_samples) * left_entropy + (n_right / n_samples) * right_entropy

        return parent_entropy - child_entropy

    def grow_tree(self, x_train, y_train, depth=0) -> Node:
        n_samples = len(y_train)
        n_labels = len(np.unique(y_train))

        if n_labels == 1 or depth == self.max_depth or n_samples <= self.min_samples_split:
            return Node(value=self.most_common(y_train))

        best_feature, best_threshold = self.best_split(x_train, y_train)

        left_indexes = np.argwhere(x_train[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(x_train[:, best_feature] > best_threshold).flatten()

        left = self.grow_tree(x_train[left_indexes, :], y_train[left_indexes], depth=depth + 1)
        right = self.grow_tree(x_train[right_indexes, :], y_train[right_indexes], depth=depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def traverse_tree(self, x, tree):
        if tree is None:
            return None

        if getattr(tree, "value", None) is not None and (
                getattr(tree, "left", None) is None and getattr(tree, "right", None) is None):
            return tree.value

        if getattr(tree, "feature", None) is None or getattr(tree, "threshold", None) is None:
            return tree.value

        if x[tree.feature] <= tree.threshold:
            return self.traverse_tree(x, tree.left)
        else:
            return self.traverse_tree(x, tree.right)


class DecisionTree_regression:
    def __init__(
            self,
            max_depth=100,
            min_samples_split=10,
            min_impurity=1e-7,
            lambda_reg=0.1,
            alpha_reg=0.1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.tree = None
        self.depth = 0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.tree = self.grow_tree(x_train, y_train)
        self.depth = self._get_depth(self.tree)

    def _mse(self, y: np.ndarray):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _mean_value(self, y: np.ndarray) -> float:
        n_samples = len(y)
        if n_samples == 0:
            return 0
        sum_y = np.sum(y)

        if self.lambda_reg != 0 and self.alpha_reg != 0:
            if sum_y > (self.alpha_reg / 2):
                w = (sum_y - self.alpha_reg / 2) / (n_samples + self.lambda_reg)
            elif sum_y < -(self.alpha_reg / 2):
                w = (sum_y + self.alpha_reg / 2) / (n_samples + self.lambda_reg)
            else:
                w = 0
        elif self.lambda_reg == 0 and self.alpha_reg != 0:
            if sum_y > (self.alpha_reg / 2):
                w = (sum_y - self.alpha_reg / 2) / n_samples
            elif sum_y < -(self.alpha_reg / 2):
                w = (sum_y + self.alpha_reg / 2) / n_samples
            else:
                w = 0
        elif self.lambda_reg != 0 and self.alpha_reg == 0:
            w = sum_y / (n_samples + self.lambda_reg)
        else:
            w = np.mean(y)
        return w

    def _get_depth(self, node: Node) -> int:
        if node is None or node.is_leaf_node():
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in x_test])

    def _best_split(self, x_train: np.ndarray, y_train: np.ndarray):
        best_feature, best_threshold, best_gain = None, None, -1

        for feature_idx in range(x_train.shape[1]):
            values = x_train[:, feature_idx]
            unique_vals = np.unique(values)

            if len(unique_vals) > 150:
                percentiles = np.linspace(0, 100, 50)
                thresholds = np.unique(np.percentile(values, percentiles))
            else:
                thresholds = unique_vals

            for threshold in thresholds:
                gain = self._information_gain(values, y_train, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _information_gain(self, x_column: np.ndarray, y_train: np.ndarray, threshold: float) -> float:
        parent_mse = self._mse(y_train)

        left_mask = x_column <= threshold
        right_mask = ~left_mask
        y_left = y_train[left_mask]
        y_right = y_train[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        n = len(y_train)
        n_l, n_r = len(y_left), len(y_right)
        child_mse = (n_l/n * self._mse(y_left) + (n_r/n * self._mse(y_right)))

        return parent_mse - child_mse

    def grow_tree(self, x_train: np.ndarray, y_train: np.ndarray, depth=0) -> Node:
        n_samples = len(y_train)

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            self._mse(y_train) < self.min_impurity):
            return Node(value=self._mean_value(y_train))

        feat, thr, gain = self._best_split(x_train, y_train)

        if gain < self.min_impurity:
            return Node(value=self._mean_value(y_train))

        left_mask = x_train[:, feat] <= thr
        right_mask = ~left_mask
        x_left, y_left = x_train[left_mask], y_train[left_mask]
        x_right, y_right = x_train[right_mask], y_train[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return Node(value=self._mean_value(y_train))

        left_subtree = self.grow_tree(x_left, y_left, depth+1)
        right_subtree = self.grow_tree(x_right, y_right, depth+1)

        return Node(feature=feat, threshold=thr,
                   left=left_subtree, right=right_subtree)

    def _traverse_tree(self, x: np.ndarray, node: Node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class RandomForest_classification:
    def __init__(
            self,
            n_trees=100,
            max_depth=100,
            method="sqrt",
            min_samples_split=10
    ):
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.method = method
        self.min_samples_split = min_samples_split
        self.trees = []
        self.features_idx = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = x_train.shape

        if self.method == "log2":
            k = int(np.log2(n_features))
        else:
            k = int(np.sqrt(n_features))
        k = max(1, k)

        print(f"Обучение модели:")
        for _ in tqdm(range(self.n_trees)):
            sample_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            x_bootstrapped = x_train[sample_idxs]
            y_bootstrapped = y_train[sample_idxs]

            feature_idxs = np.random.choice(n_features, size=k, replace=False)
            x_bootstrapped = x_bootstrapped[:, feature_idxs]

            self.features_idx.append(feature_idxs)

            model = DecisionTree_classification(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            model.fit(x_bootstrapped, y_bootstrapped)
            self.trees.append(model)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        all_preds = []
        for i in range(self.n_trees):
            feat_indexes = self.features_idx[i]
            x_sub = x_test[:, feat_indexes]
            preds = self.trees[i].predict(x_sub)
            all_preds.append(preds)

        all_preds = np.array(all_preds).T

        y_pred = []
        for row in all_preds:
            labels, counts = np.unique(row, return_counts=True)
            y_pred.append(labels[np.argmax(counts)])

        return np.array(y_pred)


class RandomForest_regression:
    def __init__(
            self,
            n_trees=100,
            max_depth=100,
            method="sqrt",
            min_samples_split=10
    ):
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.method = method
        self.min_samples_split = min_samples_split
        self.trees = []
        self.features_idx = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        n_samples, n_features = x_train.shape

        if self.method == "log2":
            k = int(np.log2(n_features))
        else:
            k = int(np.sqrt(n_features))
        k = max(1, k)

        print(f"Обучение модели:")
        for _ in tqdm(range(self.n_trees)):
            sample_idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            x_bootstrapped = x_train[sample_idxs]
            y_bootstrapped = y_train[sample_idxs]

            feature_idxs = np.random.choice(n_features, size=k, replace=False)
            x_bootstrapped = x_bootstrapped[:, feature_idxs]

            self.features_idx.append(feature_idxs)

            model = DecisionTree_regression(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            model.fit(x_bootstrapped, y_bootstrapped)
            self.trees.append(model)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        all_preds = []
        for i in range(self.n_trees):
            feat_indexes = self.features_idx[i]
            x_sub = x_test[:, feat_indexes]
            preds = self.trees[i].predict(x_sub)
            all_preds.append(preds)

        all_preds = np.array(all_preds).T
        y_pred = np.array(np.mean(all_preds, axis=0))

        return y_pred


class GBDT_classification:
    def __init__(
            self,
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=10,
            subsample=1.0,
            random_state=None,
            lambda_reg=0.1,
            alpha_reg=0.1
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.trees = []
        self.classes = None
        self.first_probs = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = x_train.shape[0]
        self.classes = np.unique(y_train)
        K = len(self.classes)

        Y = np.zeros((n, K))
        Y[np.arange(n), np.searchsorted(self.classes, y_train)] = 1

        self.first_probs = np.mean(Y, axis=0)
        F = np.full((n, K), np.log(self.first_probs + 1e-8))

        self.trees = []
        for i in tqdm(range(self.n_estimators), desc="Обучение GBDT"):
            probs = softmax(F, axis=1)
            residuals = Y - probs

            if self.subsample < 1.0:
                idx = np.random.choice(n, int(self.subsample * n), replace=False)
            else:
                idx = slice(None)

            X_sub = x_train[idx]
            residuals_sub = residuals[idx]

            trees_iter = []
            for k in range(K):
                tree = DecisionTree_regression(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    alpha_reg=self.alpha_reg,
                    lambda_reg=self.lambda_reg
                )
                tree.fit(X_sub, residuals_sub[:, k])
                trees_iter.append(tree)

                F[:, k] += self.learning_rate * tree.predict(x_train)

            self.trees.append(trees_iter)

    def predict_proba(self, x_train: np.ndarray) -> np.ndarray:
        n = x_train.shape[0]
        K = len(self.classes)
        F = np.full((n, K), np.log(self.first_probs + 1e-8))

        for trees_iter in self.trees:
            for k, tree in enumerate(trees_iter):
                F[:, k] += self.learning_rate * tree.predict(x_train)

        return softmax(F, axis=1)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x_test)
        return self.classes[np.argmax(proba, axis=1)]


class GBDT_with_sklearn_classification:
    def __init__(
            self,
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=10,
            subsample=1.0,
            random_state=None,
            lambda_reg=0.1,
            alpha_reg=0.1
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.trees = []
        self.classes = None
        self.first_probs = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = x_train.shape[0]
        self.classes = np.unique(y_train)
        K = len(self.classes)

        Y = np.zeros((n, K))
        Y[np.arange(n), np.searchsorted(self.classes, y_train)] = 1

        self.first_probs = np.mean(Y, axis=0)
        F = np.full((n, K), np.log(self.first_probs + 1e-8))

        self.trees = []
        for m in tqdm(range(self.n_estimators), desc="Обучение GBDT"):
            probs = softmax(F, axis=1)
            residuals = Y - probs

            if self.subsample < 1.0:
                idx = np.random.choice(n, int(self.subsample * n), replace=False)
            else:
                idx = slice(None)

            X_sub = x_train[idx]
            residuals_sub = residuals[idx]

            trees_iter = []
            for k in range(K):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                tree.fit(X_sub, residuals_sub[:, k])
                trees_iter.append(tree)

                F[:, k] += self.learning_rate * tree.predict(x_train)

            self.trees.append(trees_iter)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        n = x_test.shape[0]
        K = len(self.classes)
        F = np.full((n, K), np.log(self.first_probs + 1e-8))

        for trees_iter in self.trees:
            for k, tree in enumerate(trees_iter):
                F[:, k] += self.learning_rate * tree.predict(x_test)

        return softmax(F, axis=1)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x_test)
        return self.classes[np.argmax(proba, axis=1)]
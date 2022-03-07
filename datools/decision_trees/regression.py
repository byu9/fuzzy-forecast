'''
Decision tree base class
'''


import numpy
from types import SimpleNamespace
from collections import deque
from ..containers.binary_trees import Binary_Tree, Binary_Tree_Node
from ..metrics.regression import sum_of_squared_error
from ..gradients.nonlinearity import Sigmoid


class Decision_Tree_Node(Binary_Tree_Node):
    __slots__ = (
        'feature_col',
        'threshold',
        'gain',
        'impurity',
        'features',
        'target',
        'ybar',
        'r',
    )

    def __init__(self):
        self.gain = None


class Decision_Tree_Regressor:

    __slots__ = (
        '_min_count',
        '_impurity',
        '_tree',
        '_is_fitted',
        '_min_impurity_drop',
    )

    def __init__(self, min_count, min_impurity_drop):
        self._min_count = min_count
        self._impurity = sum_of_squared_error
        self._min_impurity_drop = min_impurity_drop
        self._is_fitted = False

    def _get_candidate_splits(self, feature_vals):
        (sorted_vals, counts) = numpy.unique(feature_vals,
                                             return_counts=True)

        # calculates the midpoint between two values
        candidate_split_points = (sorted_vals[1:] + sorted_vals[:-1]) / 2

        # makes sure the split results in at least the minimal sample size
        # on both sides
        counts_left = numpy.cumsum(counts[:-1])
        counts_right = counts.sum() - counts_left
        mask = (
            (counts_left >= self._min_count) &
            (counts_right >= self._min_count)
        )
        candidate_split_points = candidate_split_points[mask]

        return candidate_split_points

    def _find_best_split(self, features, target):
        best_split = SimpleNamespace()
        best_split.impurity = numpy.inf
        best_split.left = Decision_Tree_Node()
        best_split.right = Decision_Tree_Node()

        feature_cols = range(features.shape[1])

        for feature_col in feature_cols:
            feature_vals = features[:, feature_col]
            candidate_splits = self._get_candidate_splits(feature_vals)

            for threshold in candidate_splits:
                left_mask = (feature_vals <= threshold)
                right_mask = ~left_mask

                left_target = target[left_mask]
                right_target = target[right_mask]

                left_ybar = left_target.mean()
                right_ybar = right_target.mean()
                left_impurity = self._impurity(left_ybar, left_target)
                right_impurity = self._impurity(right_ybar, right_target)
                impurity_after_split = left_impurity + right_impurity

                if impurity_after_split < best_split.impurity:
                    best_split.impurity = impurity_after_split
                    best_split.feature_col = feature_col
                    best_split.threshold = threshold

                    best_split.left.features = features[left_mask]
                    best_split.right.features = features[right_mask]
                    best_split.left.target = target[left_mask]
                    best_split.right.target = target[right_mask]

                    best_split.left.ybar = left_ybar
                    best_split.right.ybar = right_ybar
                    best_split.left.impurity = left_impurity
                    best_split.right.impurity = right_impurity

        return best_split

    def _build_tree(self, features, target):
        self._tree = Binary_Tree()

        root_node = Decision_Tree_Node()

        root_node.features = features
        root_node.target = target
        root_node.ybar = target.mean()
        root_node.impurity = self._impurity(root_node.ybar, target)

        self._tree.add_node(root_node, parent=None)

        list_of_nodes_to_split = deque()
        list_of_nodes_to_split.append(root_node)

        while list_of_nodes_to_split:
            node = list_of_nodes_to_split.popleft()

            best_split = self._find_best_split(node.features, node.target)

            if (node.impurity - best_split.impurity) > self._min_impurity_drop:
                node.feature_col = best_split.feature_col
                node.threshold = best_split.threshold

                left_child = best_split.left
                right_child = best_split.right

                self._tree.add_node(left_child, parent=node, left_side=True)
                self._tree.add_node(right_child, parent=node, left_side=False)

                list_of_nodes_to_split.append(left_child)
                list_of_nodes_to_split.append(right_child)


    def _forward_prop(self, features):
        '''
        Propagates firing strength down to leaves
        Note that all n_samples are propagated at the same time
        '''
        sigmoid = Sigmoid()

        self._tree.root.r = 1
        for node in self._tree.topological_ordering():

            if not node.is_leaf:
                feature_vals = features[:, node.feature_col]

                if node.gain is None:
                    mu = (feature_vals <= node.threshold)
                    node.left_child.r = mu & node.r
                    node.right_child.r = (~mu) & node.r

                else:
                    node.x = feature_vals
                    node.a = -node.gain * (node.x - node.threshold)
                    node.mu = sigmoid.primitive(node.a)
                    node.left_child.r = node.mu * node.r
                    node.right_child.r = (1 - node.mu) * node.r

    def _backward_prop(self, dl_dyhat):
        '''
        Propogates loss gradient to parameters in each node
        '''
        sigmoid = Sigmoid()
        for node in reversed(list(self._tree.topological_ordering())):
            if node.is_leaf:
                dyhat_dybar = node.r
                dyhat_dr = node.ybar

                node.dl_dr = dl_dyhat * dyhat_dr
                node.dl_dybar = dl_dyhat * dyhat_dybar

            else:
                dl_dri_left = node.left_child.dl_dr
                dl_dri_right = node.right_child.dl_dr

                dri_dmup_left = node.r
                dri_dmup_right = -node.r
                dl_dmup = (
                    dl_dri_left * dri_dmup_left +
                    dl_dri_right * dri_dmup_right
                )
                dl_dmu = dl_dmup
                dmu_da = sigmoid.derivative(node.a)
                dl_da = dl_dmu * dmu_da

                da_dg = node.threshold - node.x
                da_dt = node.gain

                node.dl_dg = dl_da * da_dg
                node.dl_dt = dl_da * da_dt

                dri_drp_left = node.mu
                dri_drp_right = 1 - node.mu
                dl_drp = (
                    dl_dri_left * dri_drp_left +
                    dl_dri_right * dri_drp_right
                )
                node.dl_dr = dl_drp


    def fit(self, features, target):
        '''
        Fit features and output, resulting in a crisp tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''
        features = numpy.atleast_2d(features)
        target = numpy.asarray(target).reshape(-1)

        assert features.shape[0] == target.shape[0]

        self._is_fitted = True
        self._build_tree(features, target)

    def predict(self, features):
        '''
        Predict output based on features
        :param features ndarray: array of shape (n_samples, n_features, )
        :returns: array of shape (n_samples, )
        '''
        assert self._is_fitted, 'Model is not fitted'
        features = numpy.atleast_2d(features)

        self._forward_prop(features)
        predictions_per_leaf = numpy.asarray([
            leaf.r * leaf.ybar
            for leaf in self._tree.leaves
        ])
        return predictions_per_leaf.sum(axis=0).reshape(-1)

    def tune(self, features, target, learning_rate=0.01, n_iter=900):
        '''
        Fuzzify a crisp tree and tune the fuzzy tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''
        assert self._is_fitted, 'Model is not fitted.'
        features = numpy.atleast_2d(features)
        target = numpy.asarray(target).reshape(-1)
        assert features.shape[0] == target.shape[0]

        for node in self._tree.nodes:
            if not node.is_leaf:
                node.gain = 1


        from ..metrics.regression import mean_squared_error

        for _ in range(n_iter):
            yhat = self.predict(features)
            dl_dyhat = -2 * (target - yhat)
            self._backward_prop(dl_dyhat)

            for node in self._tree.nodes:
                if node.is_leaf:
                    node.ybar -= learning_rate * node.dl_dybar

                else:
                    node.gain -= learning_rate * node.dl_dg
                    node.threshold -= learning_rate * node.dl_dt

            print(f'loss={mean_squared_error(yhat, target)}')







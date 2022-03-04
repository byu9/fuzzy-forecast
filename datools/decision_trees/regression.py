'''
Decision tree regressor
'''


import numpy
from types import SimpleNamespace
from collections import deque

from ..containers.binary_trees import Binary_Tree, Binary_Tree_Node
from ..metrics.regression import Sum_Of_Squared_Error
from .queries import Crisp_Threshold_Query, fuzzify


__all__ = (
    'Decision_Tree_Regressor',
)



class _Decision_Tree_Base:
    __slots__ = (
        '_tree',
        '_is_fitted',
        '_impurity',
        '_min_impurity_decrease',
        '_min_samples',
    )

    def __init__(self, impurity, min_impurity_decrease, min_samples):
        self._tree = Binary_Tree()
        self._is_fitted = False
        self._impurity = impurity
        self._min_samples = min_samples
        self._min_impurity_decrease = min_impurity_decrease

    def _build_tree(self, features, target, feature_names=None):
        features = numpy.asarray(features)
        target = numpy.asarray(target)

        assert features.ndim == 2
        assert target.ndim == 1
        assert features.shape[0] == target.shape[0]
        assert not self._is_fitted, 'Model is fitted'

        self._is_fitted = True

        feature_cols = range(features.shape[1])
        if feature_names is None:
            feature_names = [None] * features.shape[1]

        root_node = Binary_Tree_Node()
        self._tree.add_node(root_node, parent=None)

        root_node.samples_features = features
        root_node.samples_target = target
        root_node.pred = root_node.samples_target.mean()

        list_of_nodes_to_split = deque()
        list_of_nodes_to_split.append(root_node)

        while list_of_nodes_to_split:
            node = list_of_nodes_to_split.popleft()

            node.impurity = self._impurity(node.pred, node.samples_target)

            if len(node.samples_target) < self._min_samples:
                continue

            # calculates the impurities for all features at all possible splits
            best = SimpleNamespace()
            best.left = SimpleNamespace()
            best.right = SimpleNamespace()
            best.impurity = numpy.inf

            for feature_col in feature_cols:
                feature_vals = node.samples_features[:, feature_col]
                target_vals = node.samples_target

                # excludes the last value so we don't end up with an empty split
                candidate_thresholds = numpy.unique(feature_vals)[:-1]

                for threshold in candidate_thresholds:
                    split_left = (feature_vals <= threshold)
                    split_right = ~split_left

                    left_samples_target = target_vals[split_left]
                    right_samples_target = target_vals[split_right]

                    left_pred = left_samples_target.mean()
                    right_pred = right_samples_target.mean()

                    left_impurity = self._impurity(
                        left_pred, left_samples_target)

                    right_impurity = self._impurity(
                        right_pred, right_samples_target)

                    impurity_after_split = left_impurity + right_impurity

                    if impurity_after_split < best.impurity:
                        best.impurity = impurity_after_split

                        best.left.samples_target = left_samples_target
                        best.right.samples_target = right_samples_target

                        best.left.samples_features = (
                            node.samples_features[split_left])

                        best.right.samples_features = (
                            node.samples_features[split_right])

                        best.left.pred = left_pred
                        best.right.pred = right_pred

                        best.feature_col = feature_col
                        best.threshold = threshold

            if best.impurity < node.impurity - self._min_impurity_decrease:
                left_child = Binary_Tree_Node()
                right_child = Binary_Tree_Node()

                self._tree.add_node(left_child, parent=node, left_side=True)
                self._tree.add_node(right_child, parent=node, left_side=False)

                list_of_nodes_to_split.append(left_child)
                list_of_nodes_to_split.append(right_child)

                left_child.pred = best.left.pred
                left_child.samples_target = best.left.samples_target
                left_child.samples_features = best.left.samples_features

                right_child.pred = best.right.pred
                right_child.samples_target = best.right.samples_target
                right_child.samples_features = best.right.samples_features

                node.query = Crisp_Threshold_Query(
                    col_index=best.feature_col,
                    feature_name=feature_names[best.feature_col],
                    threshold=best.threshold, side='<=')

        # while list_of_nodes_to_split



    def _forward_prop(self, features):
        '''
        Propogates firing strength down to leaves
        Note that all n_samples are propagated at the same time
        '''
        if self._tree.root is not None:
            self._tree.root.firing_strength = 1

        for node in self._tree.topological_ordering():

            degree_of_truth = node.query.degree_of_truth(features)
            degree_of_false = 1 - degree_of_truth

            if node.left_child is not None:
                node.left_child.firing_strength = (
                    node.firing_strength * degree_of_truth)

            if node.right_child is not None:
                node.right_child.firing_strength = (
                    node.firing_strength * degree_of_false)

    def _backward_prop(self):
        pass






class Decision_Tree_Regressor(_Decision_Tree_Base):

    def __init__(self, impurity=Sum_Of_Squared_Error(),
                 min_impurity_decrease=0.0, min_samples=200):
        '''
        '''
        super().__init__(impurity, min_impurity_decrease, min_samples)

    def fit(self, features, target, feature_names=None):
        '''
        Fit features and output, resulting in a crisp tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''
        self._build_tree(features, target, feature_names)


    def tune(self, features, output, loss):
        '''
        Fuzzify a crisp tree and tune the fuzzy tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''
        assert self._is_fitted, "You must fit the tree first."

        for node in self._tree.nodes:
            node.query = fuzzify(node.query)

        pass

    def predict(self, features):
        '''
        Predict output based on features
        :param features ndarray: array of shape (n_samples, n_features, )
        :returns: array of shape (n_samples, )
        '''
        assert self._is_fitted, "You must fit the tree first."

        self._forward_prop(features)

        prediction = numpy.mean((
            leaf.firing_strength * leaf.pred
            for leaf in self._tree.leaves
        ), axis=1)

        return prediction

    def __repr__(self):
        rules = [
            (
                f'Rule {index}:\n' +

                'If\n\t{}\n\n'.format('\t and \n\t'.join(
                     str(ancestor.query)
                     for ancestor in leaf.ancestors)) +

                f'then pred={leaf.pred}\n'
            )

            for index, leaf in enumerate(self._tree.leaves)
        ]

        return '\n'.join(rules)



'''
Decision tree regressor
'''

from ..collections.trees import Tree, Tree_Node
from .queries import Crisp_Threshold_Query, fuzzify

__all__ = (
        'Decision_Tree_Regressor',
)

class Decision_Tree_Regressor(Tree):

    def __init__(self, min_impurity_decrease=0.0):
        '''
        :param min_impurity_decrease float: minimal impurity decrease to result
        in a split
        '''
        super().__init__()
        self._min_impurity_decrease = min_impurity_decrease

    def fit(self, features, output):
        '''
        Fit features and output, resulting in a crisp tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''

        # TODO
        # Build a tree like this

        root_node = Tree_Node()
        root_node.query = Crisp_Threshold_Query(
            threshold=10,
            feature_of_interest='feature_column_name_or_index'
        )

        self._add_node(root_node)

        left_child = Tree_Node()
        left_child.query = Crisp_Threshold_Query(
            threshold=-0.80,
            feature_of_interest='feature_column_name_or_index'
        )

        self._add_node(left_child, parent=root_node, left_side=True)


    def tune(self, features, output):
        '''
        Fuzzify a crisp tree and tune the fuzzy tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''

        for node in self._nodes:
                node.query = fuzzify(node.query)

        pass

    def predict(self, features):
        '''
        Predict output based on features
        :param features ndarray: array of shape (n_samples, n_features, )
        :returns: array of shape (n_samples, )
        '''
        pass

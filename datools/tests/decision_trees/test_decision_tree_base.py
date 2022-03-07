'''
Unit tests for decision tree base
'''

import unittest
import numpy

from datools.decision_trees.decision_tree_base import (
    Decision_Tree_Base,
)



class Test_Decision_Tree_Base(unittest.TestCase):

    def test_candidate_splits(self):
        tree = Decision_Tree_Base(min_count=2, impurity='dummy',
                                  min_impurity_drop='dummy')
        test_input = numpy.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
        test_key = numpy.array([1.5, 2.5])

        to_test = tree._get_candidate_splits(test_input)
        self.assertTrue((to_test == test_key).all())

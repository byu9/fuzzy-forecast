'''
Unit tests for decision tree queries
'''


import unittest
import numpy

from datools.decision_trees.queries import (
    Crisp_Threshold_Query,
    Fuzzy_Threshold_Query,
    fuzzify,
)


class Test_Crisp_Threshold_Query(unittest.TestCase):
    def test_degree_of_truth(self):
        features = numpy.array([
            (0.1, 1.2, 0.6, 3.4),
            (0.2, 1.3, numpy.inf, 3.5),
            (0.3, 0.7, -0.8, 0)
        ])

        query = Crisp_Threshold_Query(threshold=0.6,
                                      feature_index=-2)

        test_keys = [1, 0, 1]
        test_outputs = query.degree_of_truth(features)

        for test_key, test_output in zip(test_keys, test_outputs):
            self.assertAlmostEqual(test_key, test_output)



class Test_Fuzzy_Threshold_Query(unittest.TestCase):
    def test_degree_of_truth(self):
        features = numpy.array([
            (0.1, 1.2, 0.6, 3.4),
            (0.2, 1.3, numpy.inf, 3.5),
            (0.3, 0.7, -numpy.inf, 0)
        ])

        query = Fuzzy_Threshold_Query(threshold=0.6,
                                      feature_index=-2,
                                      gain=0.3)

        test_keys = [0.5, 0, 1]
        test_outputs = query.degree_of_truth(features)

        for test_key, test_output in zip(test_keys, test_outputs):
            self.assertAlmostEqual(test_key, test_output)

    def test_tune(self):
        query = Fuzzy_Threshold_Query(threshold=0.6,
                                      feature_index=0,
                                      gain=0.3)

        query.tune(numpy.array([[0.7], [0.6]]), numpy.array([[1, 2]]),
                   learning_rate=1)

        self.assertAlmostEqual(query.gain, 0.325, places=4)
        self.assertAlmostEqual(query.threshold, 0.375, places=4)


class Test_Fuzzify(unittest.TestCase):
    def test_case1(self):
        crisp = Crisp_Threshold_Query(threshold=0.6,
                                      feature_index=-2)

        fuzzy = fuzzify(crisp, gain=0.3)

        features = numpy.array([
            (0.1, 1.2, 0.6, 3.4),
            (0.2, 1.3, numpy.inf, 3.5),
            (0.3, 0.7, -numpy.inf, 0)
        ])

        test_keys = [0.5, 0, 1]
        test_outputs = fuzzy.degree_of_truth(features)

        for test_key, test_output in zip(test_keys, test_outputs):
            self.assertAlmostEqual(test_key, test_output)


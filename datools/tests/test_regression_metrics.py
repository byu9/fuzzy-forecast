'''
Unit tests for regression metrics
'''


import unittest

from datools.metrics.regression import Mean_Squared_Error


class Test_Mean_Squared_Error(unittest.TestCase):
    def test1(self):
        test_input_actual = [1.0, 2.0, 3.0, 4.0]
        test_input_pred   = [1.0, 1.0, 2.0, 2.0]
        test_key          = 1.5

        mse = Mean_Squared_Error()

        test_output = mse(test_input_pred, test_input_actual)

        self.assertAlmostEqual(test_output, test_key)



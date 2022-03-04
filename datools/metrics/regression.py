'''
Regression Metrics
'''

__all__ = (
    'Mean_Squared_Error',
    'Sum_Of_Squared_Error',
)


import numpy


class Mean_Squared_Error:

    @staticmethod
    def __call__(pred, actual):
        return Mean_Squared_Error.primitive(pred, actual)

    @staticmethod
    def primitive(pred, actual):
        pred = numpy.asarray(pred)
        actual = numpy.asarray(actual)

        return numpy.square(actual - pred).mean()

class Sum_Of_Squared_Error:

    @staticmethod
    def __call__(pred, actual):
        return Sum_Of_Squared_Error.primitive(pred, actual)

    @staticmethod
    def primitive(pred, actual):
        pred = numpy.asarray(pred)
        actual = numpy.asarray(actual)

        return numpy.square(actual - pred).sum()

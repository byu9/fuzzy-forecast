'''
Nonlinear functions
'''

__all__ = [
    'Sigmoid',
]


import numpy


class Sigmoid:
    @staticmethod
    def primitive(arr):
        arr = numpy.asarray(arr)
        return 1 / (1 + numpy.exp(-arr))

    @staticmethod
    def derivative(arr):
        primitive = Sigmoid.primitive(arr)
        return primitive * (1 - primitive)

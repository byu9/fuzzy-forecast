'''
Nonlinear functions
'''

__all__ = [
    'Sigmoid',
]


import numpy


class Sigmoid:
    def primitive(self, arr):
        arr = numpy.asarray(arr)
        arr = numpy.clip(arr, -500, 500)
        return 1 / (1 + numpy.exp(-arr))

    @staticmethod
    def derivative(self, arr):
        primitive = Sigmoid.primitive(arr)
        return primitive * (1 - primitive)

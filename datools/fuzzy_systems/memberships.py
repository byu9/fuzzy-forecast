'''
Membership function classes
'''


import numpy

__all__ = [
    'Sigmoid',
]


class Sigmoid:
    @staticmethod
    def primitive(arr):
        arr = numpy.asarray(arr)
        return 1 / (1 + numpy.exp(-arr))

    @classmethod
    def derivative(cls, arr):
        primitive = cls.primitive(arr)
        return primitive * (1 - primitive)

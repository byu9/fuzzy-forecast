'''
Query classes
'''


import numpy

from abc import ABCMeta, abstractmethod
from functools import singledispatch
from ..fuzzy_systems.memberships import Sigmoid


__all__ = [
    'fuzzify',
    'Crisp_Threshold_Query',
    'Fuzzy_Threshold_Query'
]


class Abstract_Binary_Query(metaclass=ABCMeta):
    __slots__ = (
        'col_index',
        'feature_name',
    )

    def __init__(self, col_index, feature_name=None):
        self.col_index = col_index
        self.feature_name = (
            feature_name if feature_name is not None else
            f'column[{col_index}]'
        )

    @abstractmethod
    def degree_of_truth(self, features):
        raise NotImplementedError()

    def degree_of_false(self, features):
        return 1 - self.degree_of_truth(features)



class Crisp_Threshold_Query(Abstract_Binary_Query):
    __slots__ = (
        'threshold',
    )

    def __init__(self, col_index, threshold=0.0, feature_name=None):
        super().__init__(col_index, feature_name)
        self.threshold = threshold

    def degree_of_truth(self, features):
        features = numpy.asarray(features)
        assert features.ndim == 2

        values_under_query = features[:, self.col_index]

        return values_under_query > self.threshold

    def __repr__(self):
        feature = self.feature_name
        threshold = self.threshold
        return f'{{Is {feature} > {threshold}?}}'



class Fuzzy_Threshold_Query(Abstract_Binary_Query):
    __slots__ = (
        'threshold',
        'gain',
        'membership_func',
    )

    def __init__(self, col_index, feature_name=None, threshold=0.0,
                 membership_func=Sigmoid(), gain=1.0):

        super().__init__(col_index, feature_name)
        self.threshold = threshold
        self.gain = gain
        self.membership_func = membership_func

    def degree_of_truth(self, features):
        features = numpy.asarray(features)
        assert features.ndim == 2

        values_under_query = features[:, self.col_index]

        activation = self.gain * (
            values_under_query - self.threshold)

        return self.membership_func.primitive(activation)

    def __repr__(self):
        feature = self.feature_name
        threshold = self.threshold
        gain = self.gain
        return f'{{Is {feature} roughly > {threshold} with gain {gain}?}}'


@singledispatch
def fuzzify(crisp, *args, **kwargs):
    raise NotImplementedError()


@fuzzify.register
def _(crisp : Crisp_Threshold_Query, *args, **kwargs):

    kwargs['feature_name'] = crisp.feature_name
    kwargs['threshold'] = crisp.threshold
    kwargs['col_index'] = crisp.col_index

    fuzzified = Fuzzy_Threshold_Query(*args, **kwargs)

    return fuzzified

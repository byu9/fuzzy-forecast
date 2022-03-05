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


class Crisp_Threshold_Query(Abstract_Binary_Query):
    __slots__ = (
        'threshold',
        '_operator_func',
        'operator'
    )

    def __init__(self, col_index, threshold=0.0, feature_name=None,
                 operator='>='):

        import operator as operatorlib

        super().__init__(col_index, feature_name)

        self.threshold = threshold
        self.operator = operator
        self._operator_func = {
            '>' : operatorlib.gt,
            '>=': operatorlib.ge,
            '<' : operatorlib.lt,
            '<=': operatorlib.le,
        }[operator]

    def degree_of_truth(self, features):
        features = numpy.asarray(features)
        assert features.ndim == 2

        values_under_query = features[:, self.col_index]

        return self._operator_func(values_under_query, self.threshold)

    def __repr__(self):
        feature = self.feature_name
        threshold = self.threshold
        operator = self.operator
        return f'{{Is {feature} {operator} {threshold}?}}'



class Fuzzy_Threshold_Query(Abstract_Binary_Query):
    __slots__ = (
        'threshold',
        'gain',
        'membership_func',
        'operator',
        '_sign',
    )

    def __init__(self, col_index, feature_name=None, threshold=0.0,
                 membership_func=Sigmoid(), gain=1.0, operator='>='):

        super().__init__(col_index, feature_name)
        assert gain > 0
        self.threshold = threshold
        self.gain = gain
        self.membership_func = membership_func
        self.operator = operator
        self._sign = {
            '>=' :  1,
            '>'  :  1,
            '<=' : -1,
            '<'  : -1
        }[operator]

    def degree_of_truth(self, features):
        features = numpy.asarray(features)
        assert features.ndim == 2

        values_under_query = features[:, self.col_index]

        activation = self._sign * self.gain * (
            values_under_query - self.threshold)

        return self.membership_func.primitive(activation)

    def __repr__(self):
        feature = self.feature_name
        threshold = self.threshold
        gain = self.gain
        operator = self.operator
        return (
            f'{{Is {feature} roughly {operator} {threshold} '
            f'with gain {gain}?}}'
        )


@singledispatch
def fuzzify(crisp, *args, **kwargs):
    raise NotImplementedError()


@fuzzify.register
def _(crisp : Crisp_Threshold_Query, *args, **kwargs):

    kwargs['feature_name'] = crisp.feature_name
    kwargs['threshold'] = crisp.threshold
    kwargs['col_index'] = crisp.col_index
    kwargs['operator'] = crisp.operator

    fuzzified = Fuzzy_Threshold_Query(*args, **kwargs)

    return fuzzified

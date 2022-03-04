'''
Query classes
'''


import numpy
import operator
from abc import ABCMeta, abstractmethod
from functools import singledispatch
from ..fuzzy_systems.memberships import Sigmoid


__all__ = [
    'fuzzify',
    'Crisp_Threshold_Query',
    'Fuzzy_Threshold_Query'
]


_operators = {
    '>' : operator.gt,
    '>=': operator.ge,
    '<' : operator.lt,
    '<=': operator.le,
}


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
        '_operator',
        '_operator_name'
    )

    def __init__(self, col_index, threshold=0.0, feature_name=None,
                 side='>='):

        super().__init__(col_index, feature_name)

        self.threshold = threshold
        self._operator_name = side
        self._operator = _operators[side]

    def degree_of_truth(self, features):
        features = numpy.asarray(features)
        assert features.ndim == 2

        values_under_query = features[:, self.col_index]

        return self._operator(values_under_query, self.threshold)

    def __repr__(self):
        feature = self.feature_name
        threshold = self.threshold
        operator = self._operator_name
        return f'{{Is {feature} {operator} {threshold}?}}'



class Fuzzy_Threshold_Query(Abstract_Binary_Query):
    __slots__ = (
        'threshold',
        'gain',
        'membership_func',
        '_operator_name',
    )

    def __init__(self, col_index, feature_name=None, threshold=0.0,
                 membership_func=Sigmoid(), gain=1.0, side='>='):

        super().__init__(col_index, feature_name)
        assert gain > 0
        self.threshold = threshold
        self.gain = gain
        self.membership_func = membership_func
        self._operator_name = side

    def degree_of_truth(self, features):
        features = numpy.asarray(features)
        assert features.ndim == 2

        values_under_query = features[:, self.col_index]

        sign = {
            '>=' :  1,
            '>'  :  1,
            '<=' : -1,
            '<'  : -1
        }[self._operator_name]

        activation = sign * self.gain * (
            values_under_query - self.threshold)

        return self.membership_func.primitive(activation)

    def __repr__(self):
        feature = self.feature_name
        threshold = self.threshold
        gain = self.gain
        operator = self._operator_name
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
    kwargs['side'] = crisp.side

    fuzzified = Fuzzy_Threshold_Query(*args, **kwargs)

    return fuzzified

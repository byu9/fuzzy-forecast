'''
Query classes
'''


from abc import ABCMeta, abstractmethod
from functools import singledispatch
import numpy
from ..fuzzy_systems.memberships import *


__all__ = [
    'fuzzify',
    'Crisp_Threshold_Query',
    'Fuzzy_Threshold_Query'
]


class Abstract_Query(metaclass=ABCMeta):
    @abstractmethod
    def degree_of_truth(self, features):
        raise NotImplementedError()

    def degree_of_false(self, features):
        return 1 - self.degree_of_truth(features)



class Crisp_Threshold_Query(Abstract_Query):
    __slots__ = (
        'feature_of_interest',
        'threshold',
    )

    def __init__(self, threshold, feature_of_interest):
        self.threshold = threshold
        self.feature_of_interest = feature_of_interest

    def degree_of_truth(self, features):
        value_under_query = features[self.feature_of_interest]

        return (
            1 if value_under_query > self.threshold
            else 0
        )

    def __repr__(self):
        feature = self.feature_of_interest
        threshold = self.threshold
        return f'{{Is X[{feature}] > {threshold}?}}'



class Fuzzy_Threshold_Query(Abstract_Query):
    __slots__ = (
        'feature_of_interest',
        'threshold',
        'fuzziness_coef',
        'membership_func',
    )

    def __init__(self, membership_func=Sigmoid, fuzziness_coef=numpy.inf):
        self.membership_func = membership_func
        self.fuzziness_coef = fuzziness_coef

    def degree_of_truth(self, features):
        value_under_query = features[self.feature_of_interest]

        activation = self.fuzziness_coef * (
            value_under_query - self.threshold)

        return self.membership_func.primitive(activation)

    def __repr__(self):
        feature = self.feature_of_interest
        threshold = self.threshold
        return f'{{Is X[{feature}] roughly > {threshold}?}}'


@singledispatch
def fuzzify(crisp, *args, **kwargs):
    raise NotImplementedError()


@fuzzify.register
def _(crisp : Crisp_Threshold_Query, *args, **kwargs):

    fuzzified = Fuzzy_Threshold_Query(*args, **kwargs)

    fuzzified.feature_of_interest = crisp.feature_of_interest
    fuzzified.threshold = crisp.threshold

    return fuzzified

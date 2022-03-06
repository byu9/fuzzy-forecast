'''
Query classes
'''


from abc import ABCMeta, abstractmethod
from functools import singledispatch
from ..gradients.nonlinearity import Sigmoid


__all__ = [
    'fuzzify',
    'Crisp_Threshold_Query',
    'Fuzzy_Threshold_Query'
]

class Abstract_Threshold_Query(metaclass=ABCMeta):
    __slots__ = (
        'feature_index',
        'threshold'
    )

    @abstractmethod
    def degree_of_truth(self):
        raise NotImplementedError()


    def describe(self, feature_names=None):
        feature_name = (
            f'column[{self.feature_index}]' if feature_names is None else
            feature_names[self.feature_index]
        )

        return f'{{{feature_name} <= {self.threshold}}}'



class Crisp_Threshold_Query(Abstract_Threshold_Query):
    __slots__ = ()

    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold

    def degree_of_truth(self, features):
        values_under_query = features[:, self.feature_index]
        return values_under_query <= self.threshold


class Fuzzy_Threshold_Query(Abstract_Threshold_Query):
    __slots__ = (
        'gain',
        'boundary_func',
        'activation',
        'feature_value',
    )

    def __init__(self, feature_index, threshold, gain, boundary_func=Sigmoid()):
        assert gain >= 0

        self.feature_index = feature_index
        self.threshold = threshold
        self.gain = gain
        self.boundary_func = boundary_func

    def degree_of_truth(self, features):
        self.feature_value = features[:, self.feature_index]
        self.activation = -self.gain * (self.feature_value - self.threshold)
        return self.boundary_func.primitive(self.activation)

    def tune(self, output_gradients):
        output_gradients = output_gradients.reshape(-1)

        # uses the chain rule
        activation_gradient = (
            output_gradients * self.boundary_func.derivative(self.activation)
        )

        gain_gradient = (
            activation_gradient * (self.threshold - self.feature_value)
        ).sum()

        threshold_gradient = (activation_gradient * self.gain).sum()

        self.gain += gain_gradient
        self.threshold += threshold_gradient


@singledispatch
def fuzzify(crisp, *args, **kwargs):
    raise NotImplementedError()


@fuzzify.register
def _(crisp : Crisp_Threshold_Query, *args, **kwargs):
    kwargs['feature_index'] = crisp.feature_index
    kwargs['threshold'] = crisp.threshold

    fuzzified = Fuzzy_Threshold_Query(*args, **kwargs)

    return fuzzified

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
    def degree_of_truth(self, features):
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
    )

    def __init__(self, feature_index, threshold, gain, boundary_func=Sigmoid()):
        self.feature_index = feature_index
        self.threshold = threshold
        self.gain = gain
        self.boundary_func = boundary_func

    def degree_of_truth(self, features):
        a = -self.gain * (features[:, self.feature_index] - self.threshold)
        return self.boundary_func.primitive(a)

    def tune(self, features, dl_dmu, learning_rate):
        a = -self.gain * (features[:, self.feature_index] - self.threshold)

        dmu_da = self.boundary_func.derivative(a)
        dl_da = dl_dmu * dmu_da

        da_dg = self.threshold - features[:, self.feature_index]
        da_dt = self.gain

        dl_dg = dl_da * da_dg
        dl_dt = dl_da * da_dt

        self.gain -= learning_rate * dl_dg.sum()
        self.threshold -= learning_rate * dl_dt.sum()



@singledispatch
def fuzzify(crisp, *args, **kwargs):
    raise NotImplementedError()


@fuzzify.register
def _(crisp : Crisp_Threshold_Query, *args, **kwargs):
    kwargs['feature_index'] = crisp.feature_index
    kwargs['threshold'] = crisp.threshold

    fuzzified = Fuzzy_Threshold_Query(*args, **kwargs)

    return fuzzified

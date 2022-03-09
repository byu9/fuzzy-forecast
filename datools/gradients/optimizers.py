'''
Classes to produce corrective gradients
'''


import numpy


class Constant_Learning_Rate:
    def __init__(self, learning_rate):
        assert learning_rate > 0
        self.epsilon = learning_rate

    def __call__(self, gradient):
        g = numpy.asarray(gradient).reshape(-1)
        return -self.epsilon * g


class RMSProp:
    _delta = 1e-6

    def __init__(self, learning_rate, decay_rate):
        assert learning_rate > 0
        assert 0 <= decay_rate <= 1
        self.epsilon = learning_rate
        self.rho = decay_rate
        self.r = 0

    def __call__(self, gradient):
        g = numpy.asarray(gradient).reshape(-1)
        self.r = self.rho * self.r + (1 - self.rho) * (g * g)
        return -self.epsilon / numpy.sqrt(self._delta + self.r) * g

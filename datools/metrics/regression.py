'''
Regression Metrics
'''

__all__ = (
    'mean_absolute_error',
    'mean_squared_error',
    'root_mean_squared_error',
    'sum_of_squared_error',
    'mean_absolute_percent_error',
    'mean_absolute_percent_full_scale_error',
    'mean_bias_error',
    'coefficient_of_determination',
)


import numpy


def mean_absolute_error(pred, actual):
    pred = numpy.asarray(pred)
    actual = numpy.asarray(actual)

    error = actual - pred

    return abs(error).mean()

def mean_squared_error(pred, actual):
    pred = numpy.asarray(pred)
    actual = numpy.asarray(actual)

    error = actual - pred

    return numpy.square(error).mean()


def root_mean_squared_error(pred, actual):
    return numpy.sqrt(mean_squared_error(pred, actual))


def sum_of_squared_error(pred, actual):
    pred = numpy.asarray(pred)
    actual =  numpy.asarray(actual)

    error = actual - pred

    return numpy.square(error).sum()


def mean_absolute_percent_error(pred, actual):
    pred = numpy.asarray(pred)
    actual = numpy.asarray(actual)

    error = actual - pred
    error_percent = error / actual * 100

    return abs(error_percent).mean()


def mean_absolute_percent_full_scale_error(pred, actual, fullscale=None):
    pred = numpy.asarray(pred)
    actual = numpy.asarray(actual)

    if fullscale is None:
        fullscale = abs(actual).max()

    error = actual - pred
    error_percent = error / fullscale * 100

    return abs(error_percent).mean()


def mean_bias_error(pred, actual):
    pred = numpy.asarray(pred)
    actual = numpy.asarray(actual)

    error = actual - pred

    return error.mean()

def coefficient_of_determination(pred, actual):
    pred = numpy.asarray(pred)
    actual = numpy.asarray(actual)

    error = actual - pred

    residual_sum_of_square = numpy.square(error).sum()
    total_sum_of_square = numpy.square(actual).sum()

    return 1 - residual_sum_of_square / total_sum_of_square



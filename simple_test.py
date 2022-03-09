#!/usr/bin/env python3
import numpy
import matplotlib.pyplot as pyplot
from pandas import DataFrame
from datools.gradients.nonlinearity import Lorentzian
from datools.regression.decision_trees import Decision_Tree_Regressor
from datools.regression.fuzzy_decision_trees import Fuzzy_Decision_Tree_Regressor
from datools.gradients.optimizers import Constant_Learning_Rate, RMSProp, Adam
# from datools.regression.decision_trees import Decision_Tree_Regressor
from datools.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percent_error,
    mean_absolute_percent_full_scale_error,
    mean_bias_error,
    coefficient_of_determination,
)


train_x = numpy.linspace(-1, 1, 500).reshape(-1, 1)
train_y = Lorentzian().primitive(train_x).reshape(-1, 1)
model0 = Decision_Tree_Regressor(min_impurity_drop=0.1, min_count=50)
model = Fuzzy_Decision_Tree_Regressor(min_impurity_drop=0.1, min_count=50)

model0.fit(train_x, train_y)
model.fit(train_x, train_y, n_iter=5000,
          ybar_optimizer=Adam(),
          gain_optimizer=Adam(),
          threshold_optimizer=Adam())

# model.fit(train_x, train_y)
train_yhat = model.predict(train_x)
train_yhat0 = model0.predict(train_x)

df_result = DataFrame({
    'train_y': train_y.reshape(-1),
    'train_yhat': train_yhat.reshape(-1),
    'crisp_yhat': train_yhat0
})

print_metrics = (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percent_error,
    mean_absolute_percent_full_scale_error,
    mean_bias_error,
    coefficient_of_determination,
)

for metric_func in print_metrics:
    print('crisp {}={:.12f}'.format(
        metric_func.__name__,
        metric_func(train_yhat0, train_y)))

    print('train {}={:.12f}'.format(
        metric_func.__name__,
        metric_func(train_yhat, train_y)))



df_result.plot(title='Train')
pyplot.show()


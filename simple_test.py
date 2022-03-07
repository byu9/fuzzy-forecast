#!/usr/bin/env python3
import numpy
import matplotlib.pyplot as pyplot
from pandas import DataFrame
from datools.gradients.nonlinearity import Lorentzian
from datools.decision_trees.regression import Decision_Tree_Regressor
from datools.metrics.regression import mean_absolute_percent_error


train_x = numpy.linspace(-1, 1, 500).reshape(-1, 1)
train_y = Lorentzian().primitive(train_x).reshape(-1, 1)

model = Decision_Tree_Regressor(min_impurity_drop=0.1, min_count=50)

model.fit(train_x, train_y)
train_yhat = model.predict(train_x)
model.tune(train_x, train_y)
tune_yhat = model.predict(train_x)

df_result = DataFrame({
    'train_y': train_y.reshape(-1),
    'train_yhat': train_yhat.reshape(-1),
    'tune_yhat': tune_yhat.reshape(-1)
})

print(f'mape_train={mean_absolute_percent_error(train_yhat, train_y)}')
print(f'mape_tune={mean_absolute_percent_error(tune_yhat, train_y)}')

df_result.plot(title='Train')
pyplot.show()


#!/usr/bin/env python3

def prepare_dataset():
    from pandas import read_csv
    df = read_csv('data.csv').fillna(0)

    df['pastweek'] = df['Demand'].shift(24*7)
    df['pastday'] = df['Demand'].shift(24)


    df.dropna(inplace=True)

    split_at = -(31+30)*24+1

    train = df[:split_at].reset_index(drop=True)
    test  = df[split_at:].reset_index(drop=True)

    y_train = train['Demand']
    y_test  = test['Demand']
    x_train = train.drop(columns=['Demand'])
    x_test  = test.drop(columns=['Demand'])

    return (x_train, y_train, x_test, y_test)


(x_train, y_train, x_test, y_test) = prepare_dataset()


from datools.regression.fuzzy_decision_trees import Fuzzy_Decision_Tree_Regressor
from datools.regression.decision_trees import Decision_Tree_Regressor
from datools.metrics.regression import (
    mean_absolute_percent_error,
    mean_absolute_percent_full_scale_error)
from datools.gradients.optimizers import Constant_Learning_Rate, RMSProp, Adam

model0 = Decision_Tree_Regressor(min_impurity_drop=0, min_count=10)
model = Fuzzy_Decision_Tree_Regressor(min_impurity_drop=0, min_count=10)
model0.fit(x_train, y_train)
loss = model.fit(x_train, y_train, batch_size=256, epochs=30,
          ybar_optimizer=Adam(),
          gain_optimizer=Adam(),
          threshold_optimizer=Adam())

yhat_train = model.predict(x_train)
yhat_test = model.predict(x_test)
yhat0_train = model0.predict(x_train)
yhat0_test = model0.predict(x_test)

mape0 = mean_absolute_percent_full_scale_error(yhat0_test, y_test)
mape = mean_absolute_percent_full_scale_error(yhat_test, y_test)

from pandas import DataFrame
from matplotlib import pyplot as plt

df = DataFrame({'y_test': y_test, 'fuzzy': yhat_test,
                'crisp': yhat0_test})

print(f'test mape crisp={mape0:.10f}%')
print(f'test mape fuzzy={mape:.10f}%')
reduction = (mape0 - mape) / mape0 * 100
print(f'reduction={reduction:.10f}%')
#model.print_tree(feature_names=x_train.columns)

df.plot()
plt.show()
import numpy
DataFrame({'mean': loss.mean(axis=1),
           '90%': numpy.quantile(loss, 0.9, axis=1),
           '10%': numpy.quantile(loss, 0.1, axis=1)
           }).plot(title='Loss')
plt.show()


#
# model.tune(x_train, y_train)
# print(model)

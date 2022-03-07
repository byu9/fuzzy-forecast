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


from datools.decision_trees.regression import Decision_Tree_Regressor
from diskcache import Cache
from datools.metrics.regression import mean_absolute_percent_error

cache = Cache('cache')
cache.evict('fit')
@cache.memoize(tag='fit')
def fit_model():
    model = Decision_Tree_Regressor(min_impurity_drop=500, min_count=300)
    model.fit(x_train, y_train)
    return model

model = fit_model()
yhat_train = model.predict(x_train)
#model.print_tree(feature_names=x_train.columns)
yhat_test = model.predict(x_test)

mape = mean_absolute_percent_error(yhat_test, y_test)

from pandas import DataFrame
from matplotlib import pyplot as plt

df = DataFrame({'y_test': y_test, 'yhat_test': yhat_test})


model.tune(x_train, y_train, learning_rate=0.0001, n_iter=2)

yhat_test_tune = model.predict(x_test)

df['tuned_yhat_test'] = yhat_test_tune

mape_tune = mean_absolute_percent_error(yhat_test_tune, y_test)

print(f'mape={mape}')
print(f'mape_tune={mape_tune}')

#model.print_tree(feature_names=x_train.columns)

df.plot()
plt.show()



#
# model.tune(x_train, y_train)
# print(model)

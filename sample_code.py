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

model = Decision_Tree_Regressor(min_impurity_decrease=1000, min_samples=500)

model.fit(x_train, y_train, feature_names=x_train.columns)

print(model)

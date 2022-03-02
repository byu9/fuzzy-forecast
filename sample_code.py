
from datools.decision_trees.regression import Decision_Tree_Regressor


model = Decision_Tree_Regressor()
model.fit('dummy', 'dummy')

for n in model._nodes:
    print(n.query)

model.tune('dunny', 'dummy')
for n in model._nodes:
    print(n.query)

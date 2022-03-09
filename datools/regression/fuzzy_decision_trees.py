'''
Fuzzy decision tree for regression
'''


import numpy
from .decision_trees import Decision_Tree_Regressor
from ..gradients.nonlinearity import Sigmoid
from ..metrics.regression import mean_squared_error


class Fuzzy_Decision_Tree_Regressor(Decision_Tree_Regressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_prop(self, features):
        sigmoid = Sigmoid()
        self._tree.root.r = 1
        for node in self._tree.topological_ordering():
            if not node.is_leaf:
                feature_vals = features[:, node.feature_col]
                node.x = feature_vals
                node.a = -node.gain * (node.x - node.threshold)
                node.mu = sigmoid.primitive(node.a)
                node.left_child.r = node.mu * node.r
                node.right_child.r = (1 - node.mu) * node.r

    def _backward_prop(self, dl_dyhat):
        sigmoid = Sigmoid()
        for node in reversed(list(self._tree.topological_ordering())):
            if node.is_leaf:
                dyhat_dybar = node.r
                dyhat_dr = node.ybar

                node.dl_dr = dl_dyhat * dyhat_dr
                node.dl_dybar = dl_dyhat * dyhat_dybar

            else:
                dl_dri_left = node.left_child.dl_dr
                dl_dri_right = node.right_child.dl_dr

                dri_dmup_left = node.r
                dri_dmup_right = -node.r
                dl_dmup = (
                    dl_dri_left * dri_dmup_left +
                    dl_dri_right * dri_dmup_right
                )
                dl_dmu = dl_dmup
                dmu_da = sigmoid.derivative(node.a)
                dl_da = dl_dmu * dmu_da

                da_dg = node.threshold - node.x
                da_dt = node.gain

                node.dl_dg = dl_da * da_dg
                node.dl_dt = dl_da * da_dt

                dri_drp_left = node.mu
                dri_drp_right = 1 - node.mu
                dl_drp = (
                    dl_dri_left * dri_drp_left +
                    dl_dri_right * dri_drp_right
                )
                node.dl_dr = dl_drp

    def fit(self, features, target, ybar_optimizer, gain_optimizer,
            threshold_optimizer, n_iter=100):
        '''
        Fit features and output, resulting in a crisp tree
        :param features ndarray: array of shape (n_samples, n_features, )
        :param output ndarray: array of shape (n_samples,)
        '''
        super().fit(features, target)
        features = numpy.atleast_2d(features)
        target = numpy.asarray(target).reshape(-1)

        # calculates initial gain
        for node in self._tree.topological_ordering():
            if not node.is_leaf:
                feature_vals = features[:, node.feature_col]
                a = feature_vals - node.threshold
                f = numpy.sqrt(
                    node.impurity /
                    (node.left_child.impurity + node.right_child.impurity)
                ) - 1
                node.gain = f / (2 * min(a.max(), (-a).max()))

        for _ in range(n_iter):
            yhat = self.predict(features)
            dl_dyhat = -2 * (target - yhat)
            self._backward_prop(dl_dyhat)

            for node in self._tree.nodes:
                if node.is_leaf:
                    node.ybar += ybar_optimizer(node.dl_dybar.mean())

                else:
                    node.gain += gain_optimizer(node.dl_dg.mean())
                    node.threshold += threshold_optimizer(node.dl_dt.mean())

            print(f'loss={mean_squared_error(yhat, target)}')



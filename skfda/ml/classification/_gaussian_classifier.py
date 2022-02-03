from __future__ import annotations

import numpy as np
from GPy.kern import Kern
from GPy.models import GPRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ...representation import FDataGrid


class GaussianClassifier(
    BaseEstimator,  # type: ignore
    ClassifierMixin,  # type: ignore
):

    def __init__(self, kernel: Kern) -> None:
        self._kernel = kernel

    def fit(self, X: FDataGrid, y: np.ndarray):

        grid = X.grid_points[0][:, np.newaxis]
        classes, y_ind = _classifier_get_classes(y)

        self._classes = classes

        for cur_class in range(0, self._classes.size):
            class_n = X[y_ind == cur_class]
            class_n -= class_n.mean()
            data = class_n.data_matrix[:, :, 0]

            reg_n = GPRegression(grid, data.T, kernel=self._kernel)
            reg_n.optimize()
            if cur_class == 0:
                self._covariance_kern_zero = reg_n.kern
                self._mean_zero = X[y_ind == cur_class].gmean()
            else:
                self._covariance_kern_one = reg_n.kern
                self._mean_one = X[y_ind == cur_class].gmean()

        return self

    def predict(self, X: FDataGrid):

        check_is_fitted(self)

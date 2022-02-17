from __future__ import annotations

import numpy as np
from GPy.kern import Kern
from GPy.models import GPRegression
from scipy.linalg import logm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ...representation import FDataGrid


class GaussianClassifier(
    BaseEstimator,  # type: ignore
    ClassifierMixin,  # type: ignore
):

    def __init__(self, kernel: Kern, regularizer: float) -> None:
        self._kernel_ = kernel
        self._regularizer_ = regularizer

    def fit(self, X: FDataGrid, y: np.ndarray) -> GaussianClassifier:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape (n_samples).

        Returns:
            self
        """
        self._classes, self._y_ind = _classifier_get_classes(y)  # noqa:WPS414

        self._cov_kernels_, self._means = self._fit_kernels_and_means(X)

        self._priors = self._calculate_priors(y)
        self._log_priors = np.log(self._priors)  # Calculates prior logartithms
        self._covariances = self._calculate_covariances(X)

        # Calculates logarithmic covariance -> -1/2 * log|sum|
        self._log_cov = self._calculate_log_covariances()

        return self

    def predict(self, X: FDataGrid) -> np.ndarray:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples) with class labels
            for each data sample.
        """
        check_is_fitted(self)
        likelihoods = [
            self._calculate_log_likelihood(curve)
            for curve in X.data_matrix
        ]

        predictions = np.asarray([
            np.where(likelihood == max(likelihood))
            for likelihood in likelihoods
        ])
        return np.reshape(predictions, predictions.size)

    def _calculate_priors(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate the prior probability of each class.

        Args:
            y: ndarray with the labels of the training data.

        Returns:
            Numpy array with the respective prior of each class.
        """
        return np.asarray([
            np.count_nonzero(y == cur_class) / y.size
            for cur_class in range(0, self._classes.size)
        ])

    def _calculate_covariances(
        self,
        X: FDataGrid,
    ) -> np.ndarray:
        """
        Calculate the covariance matrices for each class.

        It bases the calculation on the kernels that where already
        fitted with data of the corresponding classes.

        Args:
            X: FDataGrid with the training data.

        Returns:
            Numpy array with the covariance matrices.
        """
        covariance = []
        for i in range(0, self._classes.size):
            class_data = X[self._y_ind == i].data_matrix
            # Data needs to be two-dimensional
            class_data_r = class_data.reshape(
                class_data.shape[0],
                class_data.shape[1],
            )
            # Caculate covariance matrix
            covariance = covariance + [self._cov_kernels_[i].K(class_data_r.T)]
        return np.asarray(covariance)

    def _calculate_log_covariances(self) -> np.ndarray:
        """
        Calculate the logarithm of the covariance matrices for each class.

        A regularizer parameter has been used to avoid singular matrices.

        Returns:
            Numpy array with the logarithmic computation of the
            covariance matrices.
        """
        return np.asarray([
            -0.5 * np.trace(
                logm(cov + self._regularizer_ * np.eye(cov.shape[0])),
            )
            for cov in self._covariances
        ])

    def _fit_kernels_and_means(
        self,
        X: FDataGrid,
    ) -> np.ndarray:
        """
        Fit the kernel to the data in each class.

        For each class the initial kernel passed as parameter is
        adjusted and the mean is calculated.

        Args:
            X: FDataGrid with the training data.

        Returns:
            Tuple containing a ndarray of fitted kernels and
            another ndarray with the means of each class.
        """
        grid = X.grid_points[0][:, np.newaxis]
        kernels = []
        means = []
        for cur_class in range(0, self._classes.size):
            class_n = X[self._y_ind == cur_class]
            class_n_centered = class_n - class_n.mean()
            data = class_n_centered.data_matrix[:, :, 0]

            reg_n = GPRegression(grid, data.T, kernel=self._kernel_)
            reg_n.optimize()

            kernels = kernels + [reg_n.kern]
            means = means + [class_n.gmean().data_matrix[0]]
        return np.asarray(kernels), np.asarray(means)

    def _calculate_log_likelihood(self, curve: np.ndarray) -> np.ndarray:
        """
        Calculate the log likelihood quadratic discriminant analysis.

        Args:
            curve: sample where we want to calculate the discriminant.

        Returns:
            A ndarray with the log likelihoods corresponding to the
            output classes.
        """
        # Calculates difference wrt. the mean (x - un)
        data_mean = curve - self._means

        # Calculates mahalanobis distance (-1/2*(x - un).T*inv(sum)*(x - un))
        mahalanobis = []
        for j in range(0, self._classes.size):
            mh = -0.5 * data_mean[j].T @ np.linalg.solve(
                self._covariances[j] + self._regularizer_ * np.eye(
                    self._covariances[j].shape[0],
                ),
                data_mean[j],
            )
            mahalanobis = mahalanobis + [mh[0][0]]

        # Calculates the log_likelihood
        return self._log_cov + np.asarray(
            mahalanobis,
        ) + self._log_priors

from __future__ import annotations

from typing import Sequence, TypeVar, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ..._utils._sklearn_adapter import BaseEstimator, ClassifierMixin
from ...exploratory.stats import mean
from ...exploratory.stats.covariance import CovarianceEstimator
from ...representation import FData
from ...typing._numpy import NDArrayFloat, NDArrayInt, NDArrayStr

Input = TypeVar("Input", bound=FData)
Target = TypeVar("Target", bound=Union[NDArrayInt, NDArrayStr])


class LinearDiscriminantAnalysis(
    BaseEstimator,
    ClassifierMixin[Input, Target],
):
    """
    Linear Discriminant Analysis for functional data.

    Linear classifier that assumes the Gaussian distribution for each class
    share the same covariance matrix. The classification consists of computing
    the mahalanobis distance between the mean of each class and assigning the
    class with the minimum distance.
    """

    cov_estimator: CovarianceEstimator[Input]
    y_ind: NDArrayInt

    means_: Sequence[Input]
    priors_: NDArrayFloat

    _log_priors: NDArrayFloat

    def __init__(
        self,
        cov_estimator: CovarianceEstimator[Input],
    ) -> None:
        self.cov_estimator = cov_estimator

    def _calculate_priors(
        self,
        y_ind: NDArrayInt,
    ) -> None:
        """
        Calculate the prior probability of each class.

        Args:
            y_ind: Target classes indexes.

        Returns:
            None.
        """
        self.priors_ = np.bincount(y_ind) / float(len(y_ind))

    def fit(
        self,
        X: Input,
        y: Target,
    ) -> LinearDiscriminantAnalysis[Input, Target]:
        """
        Fit the model according to the given training data.

        This consists in computing the Mahalanobis distance and
        the mean of each class.

        Args:
            X: Training data.
            y: Target values.

        Returns:
            self.
        """
        classes_, y_ind = _classifier_get_classes(y)
        self.classes_ = classes_
        self.y_ind = y_ind

        self._calculate_priors(y_ind)
        self._log_priors = np.log(self.priors_)

        self._fit_gaussian_process(X)
        return self

    def _fit_gaussian_process(
        self,
        X: Input,
    ) -> None:
        """Fit the means and covariances."""
        # Fit covariance estimator with all training data
        self.cov_estimator.fit(X)

        # Compute the mean of each class
        means = []
        for class_index, _ in enumerate(self.classes_):
            X_class = X[self.y_ind == class_index]
            means.append(mean(X_class))

        self.means_ = means

    def _discriminant_function(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """
        Compute the discriminant function of the given samples.

        This is the same as the result obtained from predict_log_proba
        before applying the normalization needed to obtain the
        probabilities.

        Args:
            X: Test samples.

        Returns:
            Discriminant results of the given samples.
        """
        check_is_fitted(self)

        # Xs_centered has shape (n_classes, n_samples, n_features, n_points)
        Xs_centered = [X - class_mean for class_mean in self.means_]

        # discriminants has shape (n_classes, n_samples)
        discriminants: NDArrayFloat = np.array([
            -0.5
            * self.cov_estimator.mahalanobis(
                X_centered,
            )
            + log_prior
            for log_prior, X_centered in zip(
                self._log_priors,
                Xs_centered,
            )
        ])

        return discriminants

    def predict(
        self,
        X: Input,
    ) -> Target:
        """
        Perform classification on an array of test vectors X.

        Args:
            X: Test samples.

        Returns:
            Predicted target values for X.
        """
        probas = self.predict_log_proba(X)

        return self.classes_[  # type: ignore[no-any-return]
            np.argmax(probas, axis=0)
        ]

    def predict_log_proba(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """
        Return log of probability estimates for the test vector X.

        Args:
            X: Test samples.

        Returns:
            Log of probability estimates for the test vector X.
        """
        log_probas = self._discriminant_function(X)

        # Include the normalization constant
        log_probas -= np.logaddexp.reduce(log_probas, axis=0)

        # log_probas has shape: n_classes, n_samples with the log of
        # the probability of each sample of being in each class
        return log_probas

    def predict_proba(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """
        Return probability estimates for the test vector X.

        Args:
            X: Test samples.

        Returns:
            Probability estimates for the test vector X.
        """
        return np.exp(self.predict_log_proba(X))

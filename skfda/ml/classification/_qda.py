from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import logm
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ..._utils._sklearn_adapter import BaseEstimator, ClassifierMixin
from ...representation import FDataGrid
from ...typing._numpy import NDArrayFloat, NDArrayInt


class QuadraticDiscriminantAnalysis(
    BaseEstimator,
    ClassifierMixin[FDataGrid, NDArrayInt],
):
    """
    Functional quadratic discriminant analysis.

    It is based on the assumption that the data is part of a Gaussian process
    and depending on the output label, the covariance and mean parameters are
    different for each class. This means that curves classified with one
    determined label come from a distinct Gaussian process compared with data
    that is classified with a different label.

    The training phase of the classifier will try to approximate the two
    main parameters of a Gaussian process for each class. The covariance
    will be estimated by fitting the initial kernel passed on the creation of
    the ParameterizedFunctionalQDA object.
    The result of the training function will be two arrays, one of means and
    another one of covariances. Both with length ``n_classes``.

    The prediction phase instead uses a quadratic discriminant classifier to
    predict which gaussian process of the fitted ones correspond the most with
    each curve passed.

    Warning:
        This classifier is experimental as it does not come from a
        peer-published paper.

    Parameters:
        kernel: Initial kernel to be fitted with the training data. For now,
            only kernels that belongs to the GPy module are allowed.

        regularizer: Parameter that regularizes the covariance matrices in
            order to avoid Singular matrices. It is multiplied by the identity
            matrix and then added to the covariance one.


    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = fetch_growth(return_X_y=True, as_frame=True)
        >>> X = X.iloc[:, 0].values
        >>> y = y.values.codes
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X,
        ...     y,
        ...     test_size=0.3,
        ...     stratify=y,
        ...     random_state=0,
        ... )

        Then we need to choose and import a kernel so it can be fitted with
        the data in the training phase. We will use a Gaussian kernel. The
        variance and lengthscale parameters will be optimized during the
        training phase. Therefore, the initial values do not matter too much.
        We will use random values such as 1 for the mean and 6 for the
        variance.

        >>> from skfda.exploratory.stats.covariance import (
        ...     ParametricGaussianCovariance
        ... )
        >>> from skfda.ml.classification import QuadraticDiscriminantAnalysis
        >>> from GPy.kern import RBF
        >>> rbf = RBF(input_dim=1, variance=6, lengthscale=1)

        We will fit the ParameterizedFunctionalQDA with training data. We
        use as regularizer parameter a low value such as 0.05.

        >>> qda = QuadraticDiscriminantAnalysis(
        ...     ParametricGaussianCovariance(rbf),
        ...     regularizer=0.05,
        ... )
        >>> qda = qda.fit(X_train, y_train)


        We can predict the class of new samples.

        >>> list(qda.predict(X_test))
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
         1, 0, 1, 0, 1, 0, 1, 1]

        Finally, we calculate the mean accuracy for the test data.

        >>> round(qda.score(X_test, y_test), 2)
        0.96

    """

    def __init__(
        self,
        cov_estimator,
        *,
        regularizer: float = 0,
    ) -> None:
        self.cov_estimator = cov_estimator
        self.regularizer = regularizer

    def fit(
        self,
        X: FDataGrid,
        y: NDArrayInt,
    ) -> QuadraticDiscriminantAnalysis:
        """
        Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape (n_samples).

        Returns:
            self
        """
        classes, y_ind = _classifier_get_classes(y)
        self.classes = classes
        self.y_ind = y_ind

        _, means = self._fit_gaussian_process(X)
        self.means_ = means

        self.priors_ = self._calculate_priors(y)
        self._log_priors = np.log(self.priors_)

        self._regularized_covariances = (
            self._covariances
            + self.regularizer * np.eye(len(X.grid_points[0]))
        )

        self._log_determinant_covariances = np.asarray([
            np.trace(logm(regularized_covariance))
            for regularized_covariance in self._regularized_covariances
        ])

        return self

    def predict(self, X: FDataGrid) -> NDArrayInt:
        """
        Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples) with class labels
            for each data sample.
        """
        check_is_fitted(self)

        return np.argmax(  # type: ignore[no-any-return]
            self._calculate_log_likelihood(X.data_matrix),
            axis=1,
        )

    def _calculate_priors(self, y: NDArrayInt) -> NDArrayFloat:
        """
        Calculate the prior probability of each class.

        Args:
            y: ndarray with the labels of the training data.

        Returns:
            Numpy array with the respective prior of each class.
        """
        _, counts = np.unique(y, return_counts=True)
        return counts / len(y)

    def _fit_gaussian_process(
        self,
        X: FDataGrid,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
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
        cov_estimators = []
        means = []
        covariance = []
        for class_index, _ in enumerate(self.classes):
            X_class = X[self.y_ind == class_index]
            cov_estimator = clone(self.cov_estimator).fit(X_class)

            cov_estimators.append(cov_estimator)
            means.append(cov_estimator.location_.data_matrix[0])
            covariance.append(cov_estimator.covariance_.data_matrix[0, ..., 0])

        self._covariances = np.asarray(covariance)

        return np.asarray(cov_estimators), np.asarray(means)

    def _calculate_log_likelihood(self, X: NDArrayFloat) -> NDArrayFloat:
        """
        Calculate the log likelihood quadratic discriminant analysis.

        Args:
            X: sample where we want to calculate the discriminant.

        Returns:
            A ndarray with the log likelihoods corresponding to the
            output classes.
        """
        # Calculates difference wrt. the mean (x - un)
        X_centered = X[:, np.newaxis, :, :] - self.means_[np.newaxis, :, :, :]

        # Calculates mahalanobis distance (-1/2*(x - un).T*inv(sum)*(x - un))
        mahalanobis_distances = np.reshape(
            np.transpose(X_centered, axes=(0, 1, 3, 2))
            @ np.linalg.solve(
                self._regularized_covariances,
                X_centered,
            ),
            (-1, self.classes.size),
        )

        return np.asarray(
            (
                -0.5 * self._log_determinant_covariances
                - 0.5 * mahalanobis_distances
                + self._log_priors
            ),
        )

from __future__ import annotations

from typing import Sequence, TypeVar, Union

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ._lda import LinearDiscriminantAnalysis
from ...exploratory.stats.covariance import CovarianceEstimator
from ...representation import FData
from ...typing._numpy import NDArrayFloat, NDArrayInt, NDArrayStr

Input = TypeVar("Input", bound=FData)
Target = TypeVar("Target", bound=Union[NDArrayInt, NDArrayStr])


class QuadraticDiscriminantAnalysis(
    LinearDiscriminantAnalysis[Input, Target],
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
        >>> from skfda.misc.covariances import Gaussian
        >>> rbf = Gaussian(variance=6, length_scale=1)

        We will fit the ParameterizedFunctionalQDA with training data. We
        use as regularizer parameter a low value such as 0.05.

        >>> qda = QuadraticDiscriminantAnalysis(
        ...     ParametricGaussianCovariance(
        ...         rbf,
        ...         regularization_parameter=0.05,
        ...     ),
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

    covariances_: Sequence[CovarianceEstimator[Input]]

    def _fit_gaussian_process(
        self,
        X: Input,
    ) -> None:
        """Fit the means and covariances."""
        # Compute the mean of each class and the covariance estimator
        cov_estimators = []
        means = []
        for class_index, _ in enumerate(self.classes_):
            X_class = X[self.y_ind == class_index]
            cov_estimator: CovarianceEstimator[Input] = clone(
                self.cov_estimator,
            ).fit(X_class)

            cov_estimators.append(cov_estimator)
            means.append(cov_estimator.location_)

        self.means_ = means
        self.covariances_ = cov_estimators

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
            cov_estimator.score(
                X_centered,
            )
            + log_prior
            for log_prior, X_centered, cov_estimator in zip(
                self._log_priors,
                Xs_centered,
                self.covariances_,
            )
        ])

        return discriminants

"""Depth-based models for supervised classification."""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.polynomial.polynomial as poly
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ..._utils import _classifier_fit_distributions
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...preprocessing.dim_reduction.feature_extraction import DDGTransformer
from ...representation.grid import FDataGrid

T = TypeVar("T", contravariant=True)


class MaximumDepthClassifier(BaseEstimator, ClassifierMixin):
    """Maximum depth classifier for functional data.

    Test samples are classified to the class where they are deeper.

    Parameters:
        depth_method:
            The depth class to use when calculating the depth of a test
            sample in a class. See the documentation of the depths module
            for a list of available depths. By default it is ModifiedBandDepth.
    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        We will fit a Maximum depth classifier

        >>> from skfda.ml.classification import MaximumDepthClassifier
        >>> clf = MaximumDepthClassifier()
        >>> clf.fit(X_train, y_train)
        MaximumDepthClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test) # Predict labels for test samples
        array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.DDClassifier`
        :class:`~skfda.ml.classification.DDGClassifier`

    References:
        Ghosh, A. K. and Chaudhuri, P. (2005b). On maximum depth and
        related classifiers. Scandinavian Journal of Statistics, 32, 327–350.
    """

    def __init__(self, depth_method: Optional[Depth[T]] = None) -> None:
        self.depth_method = depth_method

    def fit(self, X: FDataGrid, y: ndarray) -> MaximumDepthClassifier:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        if self.depth_method is None:
            self.depth_method = ModifiedBandDepth()

        classes_, distributions_ = _classifier_fit_distributions(
            X, y, [self.depth_method],
        )

        self.classes_ = classes_
        self.distributions_ = distributions_

        return self

    def predict(self, X: FDataGrid) -> ndarray:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            ndarray: array of shape (n_samples) with class labels
                for each data sample.
        """
        sklearn_check_is_fitted(self)

        depths = [
            distribution.predict(X)
            for distribution in self.distributions_
        ]

        return self.classes_[np.argmax(depths, axis=0)]


class DDClassifier(BaseEstimator, ClassifierMixin):
    """Depth-versus-depth (DD) classifer for functional data.

    Transforms the data into a DD-plot and then classifies using a polynomial
    of a chosen degree. The polynomial passes through zero and maximizes the
    accuracy of the classification on the train dataset.

    If a point is below the polynomial in the DD-plot, it is classified to
    the first class. Otherwise, the point is classified to the second class.

    Parameters:
        degree: degree of the polynomial used to classify in the DD-plot
        depth_method:
            The depth class to use when calculating the depth of a test
            sample in a class. See the documentation of the depths module
            for a list of available depths. By default it is ModifiedBandDepth.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        We will fit a DD-classifier

        >>> from skfda.ml.classification import DDClassifier
        >>> clf = DDClassifier(degree=2)
        >>> clf.fit(X_train, y_train)
        DDClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test)
        array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.DDGClassifier`
        :class:`~skfda.ml.classification.MaximumDepthClassifier`
        :class:`~skfda.preprocessing.dim_reduction.feature_extraction._ddg_transformer`

    References:
        Li, J., Cuesta-Albertos, J. A., and Liu, R. Y. (2012). DD-classifier:
        Nonparametric classification procedure based on DD-plot. Journal of
        the American Statistical Association, 107(498):737-753.
    """

    def __init__(
        self,
        degree: int,
        depth_method: Optional[Depth[T]] = None,
    ) -> None:
        self.depth_method = depth_method
        self.degree = degree

    def fit(self, X: FDataGrid, y: ndarray) -> DDClassifier:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        if self.depth_method is None:
            self.depth_method = ModifiedBandDepth()

        classes_, distributions_ = _classifier_fit_distributions(
            X, y, [self.depth_method],
        )

        self.classes_ = classes_
        self.distributions_ = distributions_

        dd_coordinates = [
            distribution.predict(X)
            for distribution in self.distributions_
        ]

        polynomial_elements = combinations(
            range(len(dd_coordinates[0])),
            self.degree,
        )

        accuracy = -1

        for elements in polynomial_elements:
            x_coord = [0] + [dd_coordinates[0][e] for e in elements]
            y_coord = [0] + [dd_coordinates[1][e] for e in elements]

            coefs = poly.polyfit(
                x_coord, y_coord, self.degree,
            )

            polynomial = poly.Polynomial(coefs)

            predicted_values = polynomial(dd_coordinates[0])

            y_pred = [
                self.classes_[0] if z - y > 0 else self.classes_[1]
                for (z, y) in zip(predicted_values, dd_coordinates[1])
            ]

            new_accuracy = accuracy_score(y, y_pred)

            if (new_accuracy > accuracy):
                accuracy = new_accuracy
                self.polynomial = polynomial

        return self

    def predict(self, X: FDataGrid) -> ndarray:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            ndarray: array of shape (n_samples) with class labels
                for each data sample.
        """
        sklearn_check_is_fitted(self)

        dd_coordinates = [
            distribution.predict(X)
            for distribution in self.distributions_
        ]

        predicted_values = self.polynomial(dd_coordinates[0])

        return np.array([
            self.classes_[0] if z - y > 0 else self.classes_[1]
            for (z, y) in zip(predicted_values, dd_coordinates[1])
        ])


class DDGClassifier(BaseEstimator, ClassifierMixin):
    r"""Generalized depth-versus-depth (DD) classifer for functional data.

    This classifier builds an interface around the DDGTransfomer.

    The transformer takes a list of k depths and performs the following map:

    .. math::
        \mathcal{X} &\rightarrow \mathbb{R}^G \\
        x &\rightarrow \textbf{d} = (D_1^1(x), D_1^2(x),...,D_g^k(x))

    Where :math:`D_i^j(x)` is the depth of the point :math:`x` with respect to
    the data in the :math:`i`-th group using the :math:`j`-th depth of the
    provided list.

    Note that :math:`\mathcal{X}` is possibly multivariate, that is,
    :math:`\mathcal{X} = \mathcal{X}_1 \times ... \times \mathcal{X}_p`.

    In the G dimensional space the classification is performed using a
    multivariate classifer.

    Parameters:
        depth_method:
            The depth class or sequence of depths to use when calculating
            the depth of a test sample in a class. See the documentation of
            the depths module for a list of available depths. By default it
            is ModifiedBandDepth.
        multivariate_classifier:
            The multivariate classifier to use in the DDG-plot.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        >>> from sklearn.neighbors import KNeighborsClassifier

        We will fit a DDG-classifier using KNN

        >>> from skfda.ml.classification import DDGClassifier
        >>> clf = DDGClassifier(KNeighborsClassifier())
        >>> clf.fit(X_train, y_train)
        DDGClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test)
        array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1,
               1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.DDClassifier`
        :class:`~skfda.ml.classification.MaximumDepthClassifier`
        :class:`~skfda.preprocessing.dim_reduction.feature_extraction._ddg_transformer`

    References:
        Li, J., Cuesta-Albertos, J. A., and Liu, R. Y. (2012). DD-classifier:
        Nonparametric classification procedure based on DD-plot. Journal of
        the American Statistical Association, 107(498):737-753.

        Cuesta-Albertos, J.A., Febrero-Bande, M. and Oviedo de la Fuente, M.
        (2017) The DDG-classifier in the functional setting. TEST, 26. 119-142.
    """

    def __init__(
        self,
        multivariate_classifier: ClassifierMixin = None,
        depth_method: Optional[Union[Depth[T], Sequence[Depth[T]]]] = None,
    ) -> None:
        self.multivariate_classifier = multivariate_classifier
        self.depth_method = depth_method

    def fit(self, X: FDataGrid, y: ndarray) -> DDGClassifier:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data.
            y: Target values of shape = (n_samples).

        Returns:
            self
        """
        self.pipeline = make_pipeline(
            DDGTransformer(self.depth_method),
            self.multivariate_classifier,
        )

        self.pipeline.fit(X, y)

        return self

    def predict(self, X: FDataGrid) -> ndarray:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            ndarray: array of shape (n_samples) with class labels
                for each data sample.
        """
        return self.pipeline.predict(X)

"""Depth-based models for supervised classification."""

from skfda.exploratory.depth import multivariate
from typing import Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ..._utils import _classifier_fit_distributions
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...preprocessing.dim_reduction.feature_extraction import DDGTransformer


class MaximumDepthClassifier(BaseEstimator, ClassifierMixin):
    """Maximum depth classifier for functional data.

    Test samples are classified to the class where they are deeper.

    Parameters:
        depth_method (Depth, default
            :class:`ModifiedBandDepth <skfda.depth.ModifiedBandDepth>`):
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
        :class:`~skfda.ml.classification.DDGClassifier`

    References:
        Ghosh, A. K. and Chaudhuri, P. (2005b). On maximum depth and
        related classifiers. Scandinavian Journal of Statistics, 32, 327–350.
    """

    def __init__(self, depth_method: Depth = ModifiedBandDepth()):
        self.depth_method = depth_method

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the training data.
            y (array-like): Target values of shape = (n_samples).

        Returns:
            self (object)
        """
        classes_, distributions_ = _classifier_fit_distributions(
            X, y, [self.depth_method],
        )

        self.classes_ = classes_
        self.distributions_ = distributions_

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:
            y (np.array): array of shape (n_samples) with class labels
                for each data sample.
        """
        sklearn_check_is_fitted(self)

        depths = [
            distribution.predict(X)
            for distribution in self.distributions_
        ]

        return self.classes_[np.argmax(depths, axis=0)]


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
        depth_method (default
            :class:`ModifiedBandDepth <skfda.depth.ModifiedBandDepth>`):
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

        We will fit a Maximum depth classifier using KNN

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
        depth_method: Union[Depth, Sequence[Depth]] = ModifiedBandDepth(),
    ):
        self.multivariate_classifier = multivariate_classifier
        self.depth_method = depth_method

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the training data.
            y (array-like): Target values of shape = (n_samples).

        Returns:
            self (object)
        """
        self.pipeline = make_pipeline(
            DDGTransformer(self.depth_method),
            self.multivariate_classifier,
        )

        self.pipeline.fit(X, y)

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:
            y (np.array): array of shape (n_samples) with class labels
                for each data sample.
        """
        return self.pipeline.predict(X)

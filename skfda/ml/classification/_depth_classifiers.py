"""Depth-based models for supervised classification."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ..._utils import _classifier_get_classes
from ...exploratory.depth import Depth, ModifiedBandDepth


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
        :class:`~skfda.ml.classification.DTMClassifier`

    References:
        Ghosh, A. K. and Chaudhuri, P. (2005b). On maximum depth and
        related classifiers. Scandinavian Journal of Statistics, 32, 327–350.
    """

    def __init__(self, depth_method: Depth = None):
        self.depth_method = depth_method

        if depth_method is None:
            self.depth_method = ModifiedBandDepth()
        else:
            self.depth_method = depth_method

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the training data.
            y (array-like): Target values of shape = [n_samples].

        Returns:
            self (object)

        """
        classes_, y_ind = _classifier_get_classes(y)

        self.classes_ = classes_
        self.distributions_ = [
            clone(self.depth_method).fit(X[y_ind == cur_class])
            for cur_class in range(self.classes_.size)
        ]

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:
            y (np.array): array of shape [n_samples] with class labels
                for each data sample.

        """
        sklearn_check_is_fitted(self)

        depths = [
            distribution.predict(X)
            for distribution in self.distributions_
        ]

        return self.classes_[np.argmax(depths, axis=0)]

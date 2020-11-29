"""Distance to trimmed means (DTM) classification."""

from typing import Callable
from sklearn.base import ClassifierMixin, BaseEstimator

from ..._neighbors.classification import NearestCentroid
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...exploratory.stats import trim_mean
from ...misc.metrics import lp_distance


class DTMClassifier(BaseEstimator, ClassifierMixin):
    """Distance to trimmed means (DTM) classification.

    Test samples are classified to the class that minimizes the distance of
    the observation to the trimmed mean of the group.

    Parameters:
        proportiontocut (float): indicates the percentage of functions to
            remove. It is not easy to determine as it varies from dataset to
            dataset.
        depth_method (Depth, default
            :class:`ModifiedBandDepth <skfda.depth.ModifiedBandDepth>`):
            The depth class used to order the data. See the documentation of
            the depths module for a list of available depths. By default it
            is ModifiedBandDepth.
        metric (function, default
            :func:`lp_distance <skfda.misc.metrics.lp_distance>`):
            Distance function between two functional objects. See the
            documentation of the metrics module for a list of available
            metrics.

    Examples:
        Firstly, we will import and split the Berkeley Growth Study dataset

        >>> from skfda.datasets import fetch_growth
        >>> from sklearn.model_selection import train_test_split
        >>> dataset = fetch_growth()
        >>> fd = dataset['data']
        >>> y = dataset['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     fd, y, test_size=0.25, stratify=y, random_state=0)

        We will fit a Distance to trimmed means classifier

        >>> from skfda.ml.classification import DTMClassifier
        >>> clf = DTMClassifier(proportiontocut=0.25)
        >>> clf.fit(X_train, y_train)
        DTMClassifier(...)

        We can predict the class of new samples

        >>> clf.predict(X_test) # Predict labels for test samples
        array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

        Finally, we calculate the mean accuracy for the test data

        >>> clf.score(X_test, y_test)
        0.875

    See also:
        :class:`~skfda.ml.classification.MaximumDepthClassifier

    References:
        Fraiman, R. and Muniz, G. (2001). Trimmed means for functional
        data. Test, 10, 419-440.
    """

    def __init__(self, proportiontocut: float,
                 depth_method: Depth = ModifiedBandDepth(),
                 metric: Callable = lp_distance) -> None:
        """Initialize the classifier."""
        self.proportiontocut = proportiontocut
        self.depth_method = depth_method
        self.metric = metric

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the training data.
            y (array-like): Target values of shape = [n_samples].

        """
        self._clf = NearestCentroid(
                    metric=self.metric,
                    centroid=lambda fdatagrid: trim_mean(fdatagrid,
                                                         self.proportiontocut,
                                                         self.depth_method))
        self._clf.fit(X, y)

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:
            y (np.array): array of shape [n_samples] or
            [n_samples, n_outputs] with class labels for each data sample.
        """
        return self._clf.predict(X)

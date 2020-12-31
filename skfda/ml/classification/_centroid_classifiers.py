"""Centroid-based models for supervised classification."""

from typing import Callable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from ..._utils import _classifier_get_classes
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...exploratory.stats import mean, trim_mean
from ...misc.metrics import l2_distance, lp_distance, pairwise_distance


class NearestCentroid(BaseEstimator, ClassifierMixin):
    """Nearest centroid classifier for functional data.

    Each class is represented by its centroid, with test samples classified to
    the class with the nearest centroid.

    Parameters:
        metric: callable, (default
            :func:`lp_distance <skfda.metrics.lp_distance>`)
            The metric to use when calculating distance between test samples
            and centroids. See the documentation of the metrics module
            for a list of available metrics. Defaults used L2 distance.
        centroid: callable, (default
            :func:`mean <skfda.exploratory.stats.mean>`)
            The centroids for the samples corresponding to each class is the
            point from which the sum of the distances (according to the metric)
            of all samples that belong to that particular class are minimized.
            By default it is used the usual mean, which minimizes the sum of L2
            distances. This parameter allows change the centroid constructor.
            The function must accept a :class:`FData` with the samples of one
            class and return a :class:`FData` object with only one sample
            representing the centroid.
    Examples:
        Firstly, we will create a toy dataset with 2 classes

        >>> from skfda.datasets import make_sinusoidal_process
        >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
        >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
        ...                               phase_std=.25, random_state=0)
        >>> fd = fd1.concatenate(fd2)
        >>> y = 15*[0] + 15*[1]

        We will fit a Nearest centroids classifier

        >>> from skfda.ml.classification import NearestCentroid
        >>> neigh = NearestCentroid()
        >>> neigh.fit(fd, y)
        NearestCentroid(...)

        We can predict the class of new samples

        >>> neigh.predict(fd[::2]) # Predict labels for even samples
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    See also:
        :class:`~skfda.ml.classification.DTMClassifier`
    """

    def __init__(self, metric=l2_distance, centroid=mean):
        self.metric = metric
        self.centroid = centroid

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`, array_matrix): Training data. FDataGrid
                with the training data or array matrix with shape
                [n_samples, n_samples] if metric='precomputed'.
            y (array-like or sparse matrix): Target values of
                shape = [n_samples] or [n_samples, n_outputs].

        Returns:
            self (object)
        """
        classes_, y_ind = _classifier_get_classes(y)

        self.classes_ = classes_
        self.centroids_ = self.centroid(X[y_ind == 0])

        for cur_class in range(1, self.classes_.size):
            centroid = self.centroid(X[y_ind == cur_class])
            self.centroids_ = self.centroids_.concatenate(centroid)

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:
            y (np.array): array of shape [n_samples] or
            [n_samples, n_outputs] with class labels for each data sample.
        """
        sklearn_check_is_fitted(self)

        return self.classes_[pairwise_distance(self.metric)(
            X,
            self.centroids_,
        ).argmin(axis=1)
        ]


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
        metric (Callable, default
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
        :class:`~skfda.ml.classification.MaximumDepthClassifier`

    References:
        Fraiman, R. and Muniz, G. (2001). Trimmed means for functional
        data. Test, 10, 419-440.
    """

    def __init__(
        self,
        proportiontocut: float,
        depth_method: Depth = None,
        metric: Callable = lp_distance,
    ) -> None:
        self.proportiontocut = proportiontocut

        if depth_method is None:
            self.depth_method = ModifiedBandDepth()
        else:
            self.depth_method = depth_method

        self.metric = metric

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the training data.
            y (array-like): Target values of shape = [n_samples].

        Returns:
            self (object)
        """
        self._clf = NearestCentroid(
            metric=self.metric,
            centroid=lambda fdatagrid: trim_mean(
                fdatagrid,
                self.proportiontocut,
                depth_method=self.depth_method,
            ),
        )
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

"""Centroid-based models for supervised classification."""
from __future__ import annotations

from typing import Callable, TypeVar, Union

from sklearn.utils.validation import check_is_fitted

from ..._utils import _classifier_get_classes
from ..._utils._sklearn_adapter import BaseEstimator, ClassifierMixin
from ...exploratory.depth import Depth, ModifiedBandDepth
from ...exploratory.stats import mean, trim_mean
from ...misc.metrics import PairwiseMetric, l2_distance
from ...misc.metrics._utils import _fit_metric
from ...representation import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayInt, NDArrayStr

Input = TypeVar("Input", bound=FData)
Target = TypeVar("Target", bound=Union[NDArrayInt, NDArrayStr])


class NearestCentroid(
    BaseEstimator,
    ClassifierMixin[Input, Target],
):
    """
    Nearest centroid classifier for functional data.

    Each class is represented by its centroid, with test samples classified to
    the class with the nearest centroid.

    Parameters:
        metric:
            The metric to use when calculating distance between test samples
            and centroids. See the documentation of the metrics module
            for a list of available metrics. L2 distance is used by default.
        centroid:
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

    def __init__(
        self,
        metric: Metric[Input] = l2_distance,
        centroid: Callable[[Input], Input] = mean,
    ):
        self.metric = metric
        self.centroid = centroid

    def fit(self, X: Input, y: Target) -> NearestCentroid[Input, Target]:
        """Fit the model using X as training data and y as target values.

        Args:
            X: FDataGrid with the training data or array matrix with shape
                (n_samples, n_samples) if metric='precomputed'.
            y: Target values of
                shape = (n_samples) or (n_samples, n_outputs).

        Returns:
            self
        """
        _fit_metric(self.metric, X)

        classes, y_ind = _classifier_get_classes(y)

        self.classes_ = classes
        self.centroids_ = self.centroid(X[y_ind == 0])

        for cur_class in range(1, self.classes_.size):
            centroid = self.centroid(X[y_ind == cur_class])
            self.centroids_ = self.centroids_.concatenate(centroid)

        return self

    def predict(self, X: Input) -> Target:
        """Predict the class labels for the provided data.

        Args:
            X: FDataGrid with the test samples.

        Returns:
            Array of shape (n_samples) or
                (n_samples, n_outputs) with class labels for each data sample.
        """
        check_is_fitted(self)

        return self.classes_[  # type: ignore[no-any-return]
            PairwiseMetric(self.metric)(
                X,
                self.centroids_,
            ).argmin(axis=1)
        ]


class DTMClassifier(NearestCentroid[Input, Target]):
    """Distance to trimmed means (DTM) classification.

    Test samples are classified to the class that minimizes the distance of
    the observation to the trimmed mean of the group
    :footcite:`fraiman+muniz_2001_trimmed`.

    Parameters:
        proportiontocut:
            Indicates the percentage of functions to remove.
            It is not easy to determine as it varies from dataset to
            dataset.
        depth_method:
            The depth class used to order the data. See the documentation of
            the depths module for a list of available depths. By default it
            is ModifiedBandDepth.
        metric:
            Distance function between two functional objects. See the
            documentation of the metrics module for a list of available
            metrics. L2 distance is used by default.

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
        :class:`~skfda.ml.classification.NearestCentroid`

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        proportiontocut: float,
        depth_method: Depth[Input] | None = None,
        metric: Metric[Input] = l2_distance,
    ) -> None:
        self.proportiontocut = proportiontocut
        self.depth_method = depth_method

        super().__init__(
            metric,
            centroid=self._centroid,
        )

    def _centroid(self, fdatagrid: Input) -> Input:
        if self.depth_method is None:
            self.depth_method = ModifiedBandDepth()

        return trim_mean(
            fdatagrid,
            self.proportiontocut,
            depth_method=self.depth_method,
        )

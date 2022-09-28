"""Neighbor models for supervised classification."""

from __future__ import annotations

from typing import Sequence, TypeVar, Union, overload

from sklearn.neighbors import (
    KNeighborsClassifier as _KNeighborsClassifier,
    RadiusNeighborsClassifier as _RadiusNeighborsClassifier,
)
from typing_extensions import Literal

from ...misc.metrics import l2_distance
from ...representation import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat, NDArrayInt
from .._neighbors_base import (
    AlgorithmType,
    KNeighborsMixin,
    NeighborsClassifierMixin,
    RadiusNeighborsMixin,
    WeightsType,
)

InputBound = Union[NDArrayFloat, FData]
Input = TypeVar("Input", contravariant=True, bound=InputBound)
OutlierLabelType = Union[int, str, Sequence[int], Sequence[str], None]


class KNeighborsClassifier(
    KNeighborsMixin[Input, NDArrayInt],
    NeighborsClassifierMixin[Input, NDArrayInt],
):
    """
    Classifier implementing the k-nearest neighbors vote.

    Parameters:
        n_neighbors: Number of neighbors to use by default for
            :meth:`kneighbors` queries.
        weights: Weight function used in prediction.
            Possible values:

            - 'uniform': uniform weights. All points in each neighborhood
              are weighted equally.
            - 'distance': weight points by the inverse of their distance.
              in this case, closer neighbors of a query point will have a
              greater influence than neighbors which are further away.
            - [callable]: a user-defined function which accepts an
              array of distances, and returns an array of the same shape
              containing the weights.

        algorithm: Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.

        leaf_size: Leaf size passed to BallTree or KDTree. This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.
        metric: The distance metric to use for the tree. The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        n_jobs: The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.
            Doesn't affect :meth:`fit` method.

    Examples:
        Firstly, we will create a toy dataset with 2 classes

        >>> from skfda.datasets import make_sinusoidal_process
        >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
        >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
        ...                               phase_std=.25, random_state=0)
        >>> fd = fd1.concatenate(fd2)
        >>> y = 15*[0] + 15*[1]

        We will fit a K-Nearest Neighbors classifier

        >>> from skfda.ml.classification import KNeighborsClassifier
        >>> neigh = KNeighborsClassifier()
        >>> neigh.fit(fd, y)
        KNeighborsClassifier(...)

        We can predict the class of new samples

        >>> neigh.predict(fd[::2]) # Predict labels for even samples
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        And the estimated probabilities.

        >>> neigh.predict_proba(fd[0]) # Probabilities of sample 0
        array([[ 1.,  0.]])

    See also:
        :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroid`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
        :class:`~skfda.ml.clustering.NearestNeighbors`

    Notes:
        See Nearest Neighbors in the sklearn online documentation for a
        discussion of the choice of ``algorithm`` and ``leaf_size``.

        This class wraps the sklearn classifier
        `sklearn.neighbors.KNeighborsClassifier`.

    Warning:
        Regarding the Nearest Neighbors algorithms, if it is found that two
        neighbors, neighbor `k+1` and `k`, have identical distances
        but different labels, the results will depend on the ordering of the
        training data.

        https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    @overload
    def __init__(
        self: KNeighborsClassifier[NDArrayFloat],
        *,
        n_neighbors: int = 5,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"],
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self: KNeighborsClassifier[InputBound],
        *,
        n_neighbors: int = 5,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        pass

    # Not useless, it restricts parameters
    def __init__(  # noqa: WPS612
        self,
        *,
        n_neighbors: int = 5,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )

    def _init_estimator(self) -> _KNeighborsClassifier:

        return _KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )


class RadiusNeighborsClassifier(
    RadiusNeighborsMixin[Input, NDArrayInt],
    NeighborsClassifierMixin[Input, NDArrayInt],
):
    """
    Classifier implementing a vote among neighbors within a given radius.

    Parameters:
        radius: Range of parameter space to use by default for
            :meth:`radius_neighbors` queries.
        weights: Weight function used in prediction.
            Possible values:

            - 'uniform': uniform weights. All points in each neighborhood
                are weighted equally.
            - 'distance': weight points by the inverse of their distance.
                in this case, closer neighbors of a query point will have a
                greater influence than neighbors which are further away.
            - [callable]: a user-defined function which accepts an
                array of distances, and returns an array of the same shape
                containing the weights.

        algorithm: Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm.
                based on the values passed to :meth:`fit` method.

        leaf_size: Leaf size passed to BallTree or KDTree. This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.
        metric: The distance metric to use for the tree. The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        outlier_label:
            Label, which is given for outlier samples (samples with no
            neighbors on given radius).
            If set to None, ValueError is raised, when outlier is detected.
        n_jobs: The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.

    Examples:
        Firstly, we will create a toy dataset with 2 classes.

        >>> from skfda.datasets import make_sinusoidal_process
        >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
        >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
        ...                               phase_std=.25, random_state=0)
        >>> fd = fd1.concatenate(fd2)
        >>> y = 15*[0] + 15*[1]

        We will fit a Radius Nearest Neighbors classifier.

        >>> from skfda.ml.classification import RadiusNeighborsClassifier
        >>> neigh = RadiusNeighborsClassifier(radius=.3)
        >>> neigh.fit(fd, y)
        RadiusNeighborsClassifier(...radius=0.3...)

        We can predict the class of new samples.

        >>> neigh.predict(fd[::2]) # Predict labels for even samples
        array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    See also:
        :class:`~skfda.ml.classification.KNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroid`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
        :class:`~skfda.ml.clustering.NearestNeighbors`

    Notes:
        See Nearest Neighbors in the sklearn online documentation for a
        discussion of the choice of ``algorithm`` and ``leaf_size``.

        This class wraps the sklearn classifier
        `sklearn.neighbors.RadiusNeighborsClassifier`.

        https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    @overload
    def __init__(
        self: RadiusNeighborsClassifier[NDArrayFloat],
        *,
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"],
        outlier_label: OutlierLabelType = None,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self: RadiusNeighborsClassifier[InputBound],
        *,
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        outlier_label: OutlierLabelType = None,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Metric[Input] = l2_distance,
        outlier_label: OutlierLabelType = None,
        n_jobs: int | None = None,
    ) -> None:
        pass

    def __init__(
        self,
        *,
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        outlier_label: OutlierLabelType = None,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )

        self.outlier_label = outlier_label

    def _init_estimator(self) -> _RadiusNeighborsClassifier:
        return _RadiusNeighborsClassifier(
            radius=self.radius,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            outlier_label=self.outlier_label,
            n_jobs=self.n_jobs,
        )

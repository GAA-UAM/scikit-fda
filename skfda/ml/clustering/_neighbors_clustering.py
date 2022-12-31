"""Unsupervised learner for implementing neighbor searches."""
from __future__ import annotations

from typing import Any, TypeVar, Union, overload

from typing_extensions import Literal

from ...misc.metrics import l2_distance
from ...representation import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat
from .._neighbors_base import (
    AlgorithmType,
    KNeighborsMixin,
    RadiusNeighborsMixin,
)

InputBound = Union[NDArrayFloat, FData]
Input = TypeVar("Input", contravariant=True, bound=InputBound)
SelfType = TypeVar("SelfType", bound="NearestNeighbors[Any]")


class NearestNeighbors(
    KNeighborsMixin[Input, None],
    RadiusNeighborsMixin[Input, None],
):
    """
    Unsupervised learner for implementing neighbor searches.

    Parameters:
        n_neighbors: Number of neighbors to use by default for
            :meth:`kneighbors` queries.
        radius: Range of parameter space to use by default for
            :meth:`radius_neighbors` queries.
        algorithm: Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.

        leaf_size: Leaf size passed to BallTree or KDTree.  This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree.  The optimal value depends on the
            nature of the problem.
        metric: The distance metric to use for the tree.  The default metric is
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

        We will fit a Nearest Neighbors estimator

        >>> from skfda.ml.clustering import NearestNeighbors
        >>> neigh = NearestNeighbors(radius=.3)
        >>> neigh.fit(fd)
        NearestNeighbors(...radius=0.3...)

        Now we can query the k-nearest neighbors.

        >>> distances, index = neigh.kneighbors(fd[:2])
        >>> index # Index of k-neighbors of samples 0 and 1
        array([[ 0,  7,  6, 11,  2],...)

        >>> distances.round(2) # Distances to k-neighbors
        array([[ 0.  ,  0.28,  0.29,  0.29,  0.3 ],
               [ 0.  ,  0.27,  0.28,  0.29,  0.3 ]])

        We can query the neighbors in a given radius too.

        >>> distances, index = neigh.radius_neighbors(fd[:2])
        >>> index[0]
        array([ 0,  2,  6,  7, 11]...)

        >>> distances[0].round(2) # Distances to neighbors of the sample 0
        array([ 0.  ,  0.3 ,  0.29,  0.28,  0.29])

    See also:
        :class:`~skfda.ml.classification.KNeighborsClassifier`
        :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroids`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`


    Notes:
        See Nearest Neighbors in the sklearn online documentation for a
        discussion of the choice of ``algorithm`` and ``leaf_size``.

        This class wraps the sklearn classifier
        `sklearn.neighbors.KNeighborsClassifier`.

        https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    @overload
    def __init__(
        self: NearestNeighbors[NDArrayFloat],
        *,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"],
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self: NearestNeighbors[InputBound],
        *,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        pass

    # Parameters are important
    def __init__(  # noqa: WPS612
        self,
        *,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )

    # There is actually a change here: the default parameter!!
    def fit(  # noqa: WPS612, D102
        self: SelfType,
        X: Input,
        y: None = None,
    ) -> SelfType:
        return super().fit(X, y)

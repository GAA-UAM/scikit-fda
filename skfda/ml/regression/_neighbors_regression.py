"""Neighbor models for regression."""

from __future__ import annotations

from typing import Tuple, TypeVar, Union, overload

from sklearn.neighbors import (
    KNeighborsRegressor as _KNeighborsRegressor,
    RadiusNeighborsRegressor as _RadiusNeighborsRegressor,
)
from typing_extensions import Literal

from ...misc.metrics import l2_distance
from ...representation import FData
from ...typing._metric import Metric
from ...typing._numpy import NDArrayFloat, NDArrayInt
from .._neighbors_base import (
    AlgorithmType,
    KNeighborsMixin,
    NeighborsRegressorMixin,
    RadiusNeighborsMixin,
    WeightsType,
)

InputBound = Union[NDArrayFloat, FData]
Input = TypeVar("Input", contravariant=True, bound=InputBound)
TargetBound = Union[NDArrayFloat, FData]
Target = TypeVar("Target", bound=TargetBound)


class KNeighborsRegressor(
    NeighborsRegressorMixin[Input, Target],
    KNeighborsMixin[Input, Target],
):
    """
    Regression based on k-nearest neighbors.

    Regression with scalar, multivariate or functional response.

    The target is predicted by local interpolation of the targets associated of
    the nearest neighbors in the training set.

    Parameters:
        n_neighbors: Number of neighbors to use by default for
            :meth:`kneighbors` queries.
        weights: Weight function used in prediction.  Possible values:

            - 'uniform' : uniform weights.  All points in each neighborhood
              are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
              in this case, closer neighbors of a query point will have a
              greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an
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
        metric: The distance metric to use for the tree.  The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        n_jobs: The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.
            Doesn't affect :meth:`fit` method.

    Examples:
        Firstly, we will create a toy dataset with gaussian-like samples
        shifted.

        >>> from skfda.ml.regression import KNeighborsRegressor
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.datasets import make_multimodal_landmarks
        >>> y = make_multimodal_landmarks(
        ...     n_samples=30,
        ...     std=0.5,
        ...     random_state=0,
        ... )
        >>> y_train = y.flatten()
        >>> X_train = make_multimodal_samples(
        ...     n_samples=30,
        ...     std=0.5,
        ...     random_state=0,
        ... )
        >>> X_test = make_multimodal_samples(
        ...     n_samples=5,
        ...     std=0.05,
        ...     random_state=0,
        ... )

        We will fit a K-Nearest Neighbors regressor to regress a scalar
        response.

        >>> neigh = KNeighborsRegressor()
        >>> neigh.fit(X_train, y_train)
        KNeighborsRegressor(...)

        We can predict the modes of new samples

        >>> neigh.predict(X_test).round(2) # Predict test data
        array([ 0.38, 0.14, 0.27, 0.52, 0.38])


        Now we will create a functional response to train the model

        >>> y_train = 5 * X_train + 1
        >>> y_train
        FDataGrid(...)

        We train the estimator with the functional response

        >>> neigh.fit(X_train, y_train)
        KNeighborsRegressor(...)

        And predict the responses as in the first case.

        >>> neigh.predict(X_test)
        FDataGrid(...)

    See also:
        :class:`~skfda.ml.classification.KNeighborsClassifier`
        :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroids`
        :class:`~skfda.ml.regression.RadiusNeighborsRegressor`
        :class:`~skfda.ml.clustering.NearestNeighbors`


    Notes:
        See Nearest Neighbors in the sklearn online documentation for a
        discussion of the choice of ``algorithm`` and ``leaf_size``.

        This class wraps the sklearn regressor
        `sklearn.neighbors.KNeighborsRegressor`.

        .. warning::
           Regarding the Nearest Neighbors algorithms, if it is found that two
           neighbors, neighbor `k+1` and `k`, have identical distances
           but different labels, the results will depend on the ordering of the
           training data.

        https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    """

    @overload
    def __init__(
        self: KNeighborsRegressor[NDArrayFloat, Target],
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
        self: KNeighborsRegressor[InputBound, Target],
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

    # Not useless: it restrict the inputs.
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
        """Initialize the regressor."""
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )

    def _init_estimator(self) -> _KNeighborsRegressor:
        return _KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

    def _query(
        self,
        X: Input,
    ) -> Tuple[NDArrayFloat, NDArrayInt]:
        """Return distances and neighbors of given sample."""
        return self.kneighbors(X)


class RadiusNeighborsRegressor(
    NeighborsRegressorMixin[Input, Target],
    RadiusNeighborsMixin[Input, Target],
):
    """
    Regression based on neighbors within a fixed radius.

    Regression with scalar, multivariate or functional response.

    The target is predicted by local interpolation of the targets associated of
    the nearest neighbors in the training set.

    Parameters:
        radius: Range of parameter space to use by default for
            :meth:`radius_neighbors` queries.
        weights: Weight function used in prediction.  Possible values:

            - 'uniform' : uniform weights.  All points in each neighborhood
              are weighted equally.
            - 'distance' : weight points by the inverse of their distance.
              in this case, closer neighbors of a query point will have a
              greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an
              array of distances, and returns an array of the same shape
              containing the weights.

            Uniform weights are used by default.
        algorithm: Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`.
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.

        leaf_size: Leaf size passed to BallTree. This can affect the
            speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.
        metric: The distance metric to use for the tree.  The default metric is
            the L2 distance. See the documentation of the metrics module
            for a list of available metrics.
        n_jobs: The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors.

    Examples:
        Firstly, we will create a toy dataset with gaussian-like samples
        shifted.

        >>> from skfda.ml.regression import RadiusNeighborsRegressor
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.datasets import make_multimodal_landmarks
        >>> y = make_multimodal_landmarks(
        ...     n_samples=30,
        ...     std=0.5,
        ...     random_state=0,
        ... )
        >>> y_train = y.flatten()
        >>> X_train = make_multimodal_samples(
        ...     n_samples=30,
        ...     std=0.5,
        ...     random_state=0,
        ... )
        >>> X_test = make_multimodal_samples(
        ...     n_samples=5,
        ...     std=0.05,
        ...     random_state=0,
        ... )

        We will fit a Radius-Nearest Neighbors regressor to regress a scalar
        response.

        >>> neigh = RadiusNeighborsRegressor(radius=0.2)
        >>> neigh.fit(X_train, y_train)
        RadiusNeighborsRegressor(...radius=0.2...)

        We can predict the modes of new samples

        >>> neigh.predict(X_test).round(2) # Predict test data
        array([ 0.39, 0.07, 0.26, 0.5 , 0.46])


        Now we will create a functional response to train the model

        >>> y_train = 5 * X_train + 1
        >>> y_train
        FDataGrid(...)

        We train the estimator with the functional response

        >>> neigh.fit(X_train, y_train)
        RadiusNeighborsRegressor(...radius=0.2...)

        And predict the responses as in the first case.

        >>> neigh.predict(X_test)
        FDataGrid(...)

    See also:
        :class:`~skfda.ml.classification.KNeighborsClassifier`
        :class:`~skfda.ml.classification.RadiusNeighborsClassifier`
        :class:`~skfda.ml.classification.NearestCentroids`
        :class:`~skfda.ml.regression.KNeighborsRegressor`
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
        self: RadiusNeighborsRegressor[NDArrayFloat, Target],
        *,
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"],
        n_jobs: int | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self: RadiusNeighborsRegressor[InputBound, Target],
        *,
        radius: float = 1.0,
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
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
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
        radius: float = 1.0,
        weights: WeightsType = 'uniform',
        algorithm: AlgorithmType = 'auto',
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ) -> None:
        """Initialize the classifier."""
        super().__init__(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=n_jobs,
        )

    def _init_estimator(self) -> _RadiusNeighborsRegressor:
        return _RadiusNeighborsRegressor(
            radius=self.radius,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

    def _query(
        self,
        X: Input,
    ) -> Tuple[NDArrayFloat, NDArrayInt]:
        return self.radius_neighbors(X)

"""K-Means Algorithms Module."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, Generic, Tuple, TypeVar

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import (
    BaseEstimator,
    ClusterMixin,
    TransformerMixin,
)
from ...misc.metrics import PairwiseMetric, l2_distance
from ...misc.validation import (
    check_fdata_same_dimensions,
    validate_random_state,
)
from ...representation import FDataGrid
from ...typing._base import RandomState, RandomStateLike
from ...typing._metric import Metric
from ...typing._numpy import NDArrayAny, NDArrayFloat, NDArrayInt

SelfType = TypeVar("SelfType", bound="BaseKMeans[Any, Any]")
MembershipType = TypeVar("MembershipType", bound=NDArrayAny)

# TODO: Generalize to FData and NDArray, without losing performance
Input = TypeVar("Input", bound=FDataGrid)


class BaseKMeans(
    BaseEstimator,
    ClusterMixin[Input],
    TransformerMixin[Input, NDArrayFloat, object],
    Generic[Input, MembershipType],
):
    """Base class to implement K-Means clustering algorithms.

    Class from which both :class:`K-Means
    <skfda.ml.clustering.base_kmeans.KMeans>` and
    :class:`Fuzzy K-Means <skfda.ml.clustering.base_kmeans.FuzzyKMeans>`
    classes inherit.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 2,
        init: Input | None = None,
        metric: Metric[Input] = l2_distance,
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: RandomStateLike = 0,
    ):
        """Initialize the BaseKMeans class.

        Args:
            n_clusters: Number of groups into which the samples
                are classified. Defaults to 2.
            init: Contains the initial centers of the
                different clusters the algorithm starts with. Its data_marix
                must be of the shape (n_clusters, fdatagrid.ncol,
                fdatagrid.dim_codomain). Defaults to None, and the centers are
                initialized randomly.
            metric: functional data metric. Defaults to
                *l2_distance*.
            n_init: Number of time the k-means algorithm will
                be run with different centroid seeds. The final results will
                be the best output of n_init consecutive runs in terms of
                inertia.
            max_iter: Maximum number of iterations of the
                clustering algorithm for a single run. Defaults to 100.
            tol: tolerance used to compare the centroids
                calculated with the previous ones in every single run of the
                algorithm.
            random_state:
                Determines random number generation for centroid
                initialization. Use an int to make the randomness
                deterministic. Defaults to 0.
                See :term:`Glossary <random_state>`.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _check_clustering(self, fdata: Input) -> Input:
        """Check the arguments used in fit.

        Args:
            fdata: Object whose samples
                are classified into different groups.

        Returns:
            Validated input.

        """
        if len(fdata) < 2:
            raise ValueError(
                "The number of observations must be greater than 1.",
            )

        if self.n_clusters < 2:
            raise ValueError(
                "The number of clusters must be greater than 1.",
            )

        if self.n_init < 1:
            raise ValueError(
                "The number of iterations must be greater than 0.",
            )

        if self.init is not None and self.n_init != 1:
            self.n_init = 1
            warnings.warn(
                "Warning: The number of iterations is ignored "
                "because the init parameter is set.",
            )

        if (
            self.init is not None
            and self.init.data_matrix.shape != (
                (self.n_clusters,) + fdata.data_matrix.shape[1:]
            )
        ):
            raise ValueError(
                "The init FDataGrid data_matrix should be of "
                "shape (n_clusters, n_features, dim_codomain) "
                "and gives the initial centers.",
            )

        if self.max_iter < 1:
            raise ValueError(
                "The number of maximum iterations must be greater than 0.",
            )

        if self.tol < 0:
            raise ValueError("The tolerance must be positive.")

        return fdata

    def _tolerance(self, fdata: Input) -> float:
        variance = fdata.var()
        mean_variance = np.mean(variance[0].data_matrix)

        return float(mean_variance * self.tol)

    def _init_centroids(
        self,
        fdatagrid: Input,
        random_state: RandomState,
    ) -> Input:
        """
        Compute the initial centroids.

        Args:
            fdatagrid: Object whose samples are classified into different
                groups.
            random_state: Random number generation for centroid initialization.

        Returns:
            Initial centroids.

        """
        if self.init is None:
            _, idx = np.unique(
                fdatagrid.data_matrix,
                axis=0,
                return_index=True,
            )
            unique_data = fdatagrid[np.sort(idx)]

            if len(unique_data) < self.n_clusters:
                raise ValueError(
                    "Not enough unique data points to "
                    "initialize the requested number of "
                    "clusters",
                )

            indices = random_state.permutation(len(unique_data))[
                :self.n_clusters
            ]
            centroids = unique_data[indices]

            return centroids.copy()

        return self.init.copy()

    def _check_params(self) -> None:
        pass

    @abstractmethod
    def _create_membership(self, n_samples: int) -> MembershipType:
        pass

    @abstractmethod
    def _update(
        self,
        fdata: Input,
        membership_matrix: MembershipType,
        distances_to_centroids: NDArrayFloat,
        centroids: Input,
    ) -> None:
        pass

    def _algorithm(
        self,
        fdata: Input,
        random_state: RandomState,
    ) -> Tuple[NDArrayFloat, Input, NDArrayFloat, int]:
        """
        Fuzzy K-Means algorithm.

        Implementation of the Fuzzy K-Means algorithm for FDataGrid objects
        of any dimension.

        Args:
            fdata: Object whose samples are clustered,
                classified into different groups.
            random_state: random number generation for
                centroid initialization.

        Returns:
            Tuple containing:

                membership values:
                    membership value that observation has to each cluster.

                centroids:
                    centroids for each cluster.

                distances_to_centroids: distances of each sample to each
                    cluster.

                repetitions: number of iterations the algorithm was run.

        """
        repetitions = 0
        centroids_old_matrix = np.zeros(
            (self.n_clusters,) + fdata.data_matrix.shape[1:],
        )
        membership_matrix = self._create_membership(fdata.n_samples)

        centroids = self._init_centroids(fdata, random_state)
        centroids_old = centroids.copy(data_matrix=centroids_old_matrix)

        pairwise_metric = PairwiseMetric(self.metric)

        tolerance = self._tolerance(fdata)

        while (
            repetitions == 0
            or (
                not np.all(self.metric(centroids, centroids_old) < tolerance)
                and repetitions < self.max_iter
            )
        ):

            centroids_old.data_matrix[...] = centroids.data_matrix

            distances_to_centroids = pairwise_metric(fdata, centroids)

            self._update(
                fdata=fdata,
                membership_matrix=membership_matrix,
                distances_to_centroids=distances_to_centroids,
                centroids=centroids,
            )

            repetitions += 1

        return (
            membership_matrix,
            centroids,
            distances_to_centroids,
            repetitions,
        )

    @abstractmethod
    def _compute_inertia(
        self,
        membership: MembershipType,
        centroids: Input,
        distances_to_centroids: NDArrayFloat,
    ) -> float:
        pass

    def fit(
        self: SelfType,
        X: Input,
        y: object = None,
        sample_weight: None = None,
    ) -> SelfType:
        """
        Fit the model.

        Args:
            X: Object whose samples are clusered,
                classified into different groups.
            y: present here for API consistency by convention.
            sample_weight: present here for API consistency by
                convention.

        Returns:
            Fitted model.

        """
        fdata = self._check_clustering(X)
        random_state = validate_random_state(self.random_state)

        self._check_params()

        best_inertia = np.inf

        for _ in range(self.n_init):
            (
                membership,
                centroids,
                distances_to_centroids,
                n_iter,
            ) = (
                self._algorithm(
                    fdata=fdata,
                    random_state=random_state,
                )
            )

            inertia = self._compute_inertia(
                membership,
                centroids,
                distances_to_centroids,
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_membership = membership
                best_centroids = centroids
                best_distances_to_centroids = distances_to_centroids
                best_n_iter = n_iter

        self._best_membership = best_membership
        self.labels_ = self._prediction_from_membership(best_membership)
        self.cluster_centers_ = best_centroids
        self._distances_to_centers = best_distances_to_centroids
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def _predict_membership(
        self,
        X: Input,
        sample_weight: None = None,
    ) -> MembershipType:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X: Object whose samples are classified into different groups.
            sample_weight: present here for API consistency by convention.

        Returns:
            Label of each sample.

        """
        check_is_fitted(self)
        check_fdata_same_dimensions(self.cluster_centers_, X)

        membership_matrix = self._create_membership(X.n_samples)
        centroids = self.cluster_centers_.copy()

        pairwise_metric = PairwiseMetric(self.metric)

        distances_to_centroids = pairwise_metric(X, centroids)

        self._update(
            fdata=X,
            membership_matrix=membership_matrix,
            distances_to_centroids=distances_to_centroids,
            centroids=centroids,
        )

        return membership_matrix

    @abstractmethod
    def _prediction_from_membership(
        self,
        membership_matrix: MembershipType,
    ) -> NDArrayInt:
        pass

    def predict(
        self,
        X: Input,
        sample_weight: None = None,
    ) -> NDArrayInt:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X: Object whose samples are classified into different groups.
            sample_weight: present here for API consistency by convention.

        Returns:
            Label of each sample.

        """
        return self._prediction_from_membership(
            self._predict_membership(X, sample_weight),
        )

    def transform(self, X: Input) -> NDArrayFloat:
        """Transform X to a cluster-distance space.

        Args:
            X: Object whose samples are classified into
                different groups.

        Returns:
            distances_to_centers:
                distances of each sample to each cluster.

        """
        check_is_fitted(self)
        check_fdata_same_dimensions(self.cluster_centers_, X)
        return self._distances_to_centers

    def fit_transform(
        self,
        X: Input,
        y: object = None,
        sample_weight: None = None,
    ) -> NDArrayFloat:
        """Compute clustering and transform X to cluster-distance space.

        Args:
            X: Object whose samples are classified into different groups.
            y: present here for API consistency by convention.
            sample_weight: present here for API consistency by convention.

        Returns:
            Distances of each sample to each cluster.

        """
        self.fit(X)
        return self._distances_to_centers

    def score(
        self,
        X: Input,
        y: object = None,
        sample_weight: None = None,
    ) -> float:
        """Opposite of the value of X on the K-means objective.

        Args:
            X: Object whose samples are classified into
                different groups.
            y: present here for API consistency by convention.
            sample_weight: present here for API consistency by
                convention.

        Returns:
            Negative ``inertia_`` attribute.

        """
        check_is_fitted(self)
        check_fdata_same_dimensions(self.cluster_centers_, X)
        return -self.inertia_


class KMeans(BaseKMeans[Input, NDArrayInt]):
    r"""K-Means algorithm for functional data.

    Let :math:`\mathbf{X = \left\{ x_{1}, x_{2}, ..., x_{n}\right\}}` be a
    given dataset to be analyzed, and :math:`\mathbf{V = \left\{ v_{1}, v_{2},
    ..., v_{c}\right\}}` be the set of centers of clusters in
    :math:`\mathbf{X}` dataset in :math:`m` dimensional space :math:`\left(
    \mathbb{R}^m \right)`. Where :math:`n` is the number of objects, :math:`m`
    is the number of features, and :math:`c` is the number of partitions or
    clusters.

    KM iteratively computes cluster centroids in order to minimize the sum with
    respect to the specified measure. KM algorithm aims at minimizing an
    objective function known as the squared error function given as follows:

    .. math::
        J_{KM}\left(\mathbf{X}; \mathbf{V}\right) = \sum_{i=1}^{c}
        \sum_{j=1}^{n}D_{ij}^2

    Where, :math:`D_{ij}^2` is the squared chosen distance measure which can
    be any p-norm: :math:`D_{ij} = \lVert x_{ij} - v_{i} \rVert = \left(
    \int_I \lvert x_{ij} - v_{i}\rvert^p dx \right)^{ \frac{1}{p}}`, being
    :math:`I` the domain where :math:`\mathbf{X}` is defined, :math:`1
    \leqslant i \leqslant c`, :math:`1 \leqslant j\leqslant n_{i}`. Where
    :math:`n_{i}` represents the number of data points in i-th cluster.

    For :math:`c` clusters, KM is based on an iterative algorithm minimizing
    the sum of distances from each observation to its cluster centroid. The
    observations are moved between clusters until the sum cannot be decreased
    any more. KM algorithm involves the following steps:

    1. Centroids of :math:`c` clusters are chosen from :math:`\mathbf{X}`
        randomly or are passed to the function as a parameter.

    2. Distances between data points and cluster centroids are calculated.

    3. Each data point is assigned to the cluster whose centroid is
        closest to it.

    4. Cluster centroids are updated by using the following formula:
        :math:`\mathbf{v_{i}} ={\sum_{i=1}^{n_{i}}x_{ij}}/n_{i}` :math:`1
        \leqslant i \leqslant c`.

    5. Distances from the updated cluster centroids are recalculated.

    6. If no data point is assigned to a new cluster the run of algorithm is
        stopped, otherwise the steps from 3 to 5 are repeated for probable
        movements of data points between the clusters.

    This algorithm is applied for each dimension on the image of the FDataGrid
    object.

    Args:
        n_clusters: Number of groups into which the samples are
            classified. Defaults to 2.
        init: Contains the initial centers of the
            different clusters the algorithm starts with. Its data_marix must
            be of the shape (n_clusters, fdatagrid.ncol,
            fdatagrid.dim_codomain). Defaults to None, and the centers are
            initialized randomly.
        metric: functional data metric. Defaults to
            *l2_distance*.
        n_init: Number of time the k-means algorithm will be
            run with different centroid seeds. The final results will be the
            best output of n_init consecutive runs in terms of inertia.
        max_iter: Maximum number of iterations of the
            clustering algorithm for a single run. Defaults to 100.
        tol: Tolerance used to compare the centroids
            calculated with the previous ones in every single run of the
            algorithm.
        random_state:
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic. Defaults to 0.
            See :term:`Glossary <random_state>`.

    Attributes:
        labels\_: Vector in which each entry contains the cluster each
            observation belongs to.
        cluster_centers\_: data_matrix of shape (n_clusters, ncol,
            dim_codomain) and contains the centroids for each cluster.
        inertia\_: Sum of squared distances of samples to their closest
            cluster center for each dimension.
        n_iter\_: number of iterations the algorithm was run for each
            dimension.

    Example:

        >>> import skfda
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> kmeans = skfda.ml.clustering.KMeans(random_state=0)
        >>> kmeans.fit(fd)
        KMeans(...)
        >>> kmeans.cluster_centers_.data_matrix
        array([[[ 0.16666667],
                [ 0.16666667],
                [ 0.83333333],
                [ 2.        ],
                [ 1.66666667],
                [ 1.16666667]],
               [[-0.5       ],
                [-0.5       ],
                [-0.5       ],
                [-1.        ],
                [-1.        ],
                [-1.        ]]])

    """

    def _compute_inertia(
        self,
        membership: NDArrayInt,
        centroids: Input,
        distances_to_centroids: NDArrayFloat,
    ) -> float:
        distances_to_their_center = np.choose(
            membership,
            distances_to_centroids.T,
        )

        return float(np.sum(distances_to_their_center**2))

    def _create_membership(self, n_samples: int) -> NDArrayInt:
        return np.empty(n_samples, dtype=int)

    def _prediction_from_membership(
        self,
        membership_matrix: NDArrayInt,
    ) -> NDArrayInt:
        return membership_matrix

    def _update(
        self,
        fdata: Input,
        membership_matrix: NDArrayInt,
        distances_to_centroids: NDArrayFloat,
        centroids: Input,
    ) -> None:

        membership_matrix[:] = np.argmin(distances_to_centroids, axis=1)

        for i in range(self.n_clusters):

            indices = np.where(membership_matrix == i)[0]

            if len(indices) != 0:
                centroids.data_matrix[i] = np.average(
                    fdata.data_matrix[indices, ...],
                    axis=0,
                )


class FuzzyCMeans(BaseKMeans[Input, NDArrayFloat]):
    r"""
    Fuzzy c-Means clustering for functional data.

    Let :math:`\mathbf{X = \left\{ x_{1}, x_{2}, ..., x_{n}\right\}}` be a
    given dataset to be analyzed, and :math:`\mathbf{V = \left\{ v_{1}, v_{2},
    ..., v_{c}\right\}}` be the set of centers of clusters in
    :math:`\mathbf{X}` dataset in :math:`m` dimensional space :math:`\left(
    \mathbb{R}^m \right)`. Where :math:`n` is the number of objects, :math:`m`
    is the number of features, and :math:`c` is the number of partitions
    or clusters.

    FCM minimizes the following objective function:

    .. math::
        J_{FCM}\left(\mathbf{X}; \mathbf{U, V}\right) = \sum_{i=1}^{c}
        \sum_{j=1}^{n}u_{ij}^{f}D_{ij}^2.

    This function differs from classical KM with the use of weighted squared
    errors instead of using squared errors only. In the objective function,
    :math:`\mathbf{U}` is a fuzzy partition matrix that is computed from
    dataset :math:`\mathbf{X}`: :math:`\mathbf{U} = [u_{ij}] \in M_{FCM}`.

    The fuzzy clustering of :math:`\mathbf{X}` is represented with
    :math:`\mathbf{U}` membership matrix. The element :math:`u_{ij}` is the
    membership value of j-th object to i-th cluster. In this case, the i-th row
    of :math:`\mathbf{U}` matrix is formed with membership values of :math:`n`
    objects to i-th cluster. :math:`\mathbf{V}` is a prototype vector of
    cluster prototypes (centroids): :math:`\mathbf{V = \left\{ v_{1}, v_{2},
    ..., v_{c}\right\}}`,:math:`\mathbf{v_{i}}\in \mathbb{R}^m`.

    :math:`D_{ij}^2` is the squared chosen distance measure which can be any
    p-norm: :math:`D_{ij} =\lVert x_{ij} - v_{i} \rVert = \left( \int_I \lvert
    x_{ij} - v_{i}\rvert^p dx \right)^{ \frac{1}{p}}`, being :math:`I` the
    domain where :math:`\mathbf{X}` is defined, :math:`1 \leqslant i
    \leqslant c`, :math:`1 \leqslant j\leqslant n_{i}`. Where :math:`n_{i}`
    represents the number of data points in i-th cluster.

    FCM is an iterative process and stops when the number of iterations is
    reached to maximum, or when the centroids of the clusters do not change.
    The steps involved in FCM are:

        1. Centroids of :math:`c` clusters are chosen from :math:`\mathbf{X}`
            randomly or are passed to the function as a parameter.

        2. Membership values of data points to each cluster are calculated
            with: :math:`u_{ij} = \left[ \sum_{k=1}^c\left( D_{ij}/D_{kj}
            \right)^\frac{2}{f-1} \right]^{-1}`.

        3. Cluster centroids are updated by using the following formula:
            :math:`\mathbf{v_{i}} =\frac{\sum_{j=1}^{n}u_{ij}^f x_{j}}{
            \sum_{j=1}^{n} u_{ij}^f}`, :math:`1 \leqslant i \leqslant c`.

        4. If no cluster centroid changes the run of algorithm is stopped,
            otherwise return to step 2.

    This algorithm is applied for each dimension on the image of the FDataGrid
    object.

    Args:
        n_clusters: Number of groups into which the samples are
            classified. Defaults to 2.
        init: Contains the initial centers of the
            different clusters the algorithm starts with. Its data_marix must
            be of the shape (n_clusters, fdatagrid.ncol,
            fdatagrid.dim_codomain). Defaults to None, and the centers are
            initialized randomly.
        metric: functional data metric. Defaults to
            *l2_distance*.
        n_init: Number of time the k-means algorithm will be
            run with different centroid seeds. The final results will be the
            best output of n_init consecutive runs in terms of inertia.
        max_iter: Maximum number of iterations of the
            clustering algorithm for a single run. Defaults to 100.
        tol: tolerance used to compare the centroids
            calculated with the previous ones in every single run of the
            algorithm.
        random_state:
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic. Defaults to 0.
            See :term:`Glossary <random_state>`.
        fuzzifier: Scalar parameter used to specify the
            degree of fuzziness in the fuzzy algorithm. Defaults to 2.

    Attributes:
        membership_degree\_: Matrix in which each entry contains the
            probability of belonging to each group.
        labels\_: Vector in which each entry contains the cluster each
            observation belongs to (the one with the maximum membership
            degree).
        cluster_centers\_: data_matrix of shape
            (n_clusters, ncol, dim_codomain) and contains the centroids for
            each cluster.
        inertia\_: Sum of squared
            distances of samples to their closest cluster center for each
            dimension.
        n_iter\_: number of iterations
            the algorithm was run for each dimension.


    Example:

        >>> import skfda
        >>> data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
        ...                [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
        ...                [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        >>> grid_points = [2, 4, 6, 8]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> fuzzy_kmeans = skfda.ml.clustering.FuzzyCMeans(random_state=0)
        >>> fuzzy_kmeans.fit(fd)
        FuzzyCMeans(...)
        >>> fuzzy_kmeans.cluster_centers_.data_matrix
        array([[[ 2.83994301,  0.24786354],
                [ 3.83994301,  0.34786354],
                [ 4.83994301,  0.44786354],
                [ 5.83994301,  0.53191927]],
               [[ 1.25134384,  0.35023779],
                [ 2.25134384,  0.45023779],
                [ 3.25134384,  0.55023779],
                [ 4.25134384,  0.6251158 ]]])


    """

    def __init__(
        self,
        *,
        n_clusters: int = 2,
        init: Input | None = None,
        metric: Metric[Input] = l2_distance,
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: RandomStateLike = 0,
        fuzzifier: float = 2,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

        self.fuzzifier = fuzzifier

    @property
    def membership_degree_(self) -> NDArrayFloat:
        return self._best_membership

    def _check_params(self) -> None:
        if self.fuzzifier <= 1:
            raise ValueError("The fuzzifier parameter must be greater than 1.")

    def _compute_inertia(
        self,
        membership: NDArrayFloat,
        centroids: Input,
        distances_to_centroids: NDArrayFloat,
    ) -> float:
        return float(
            np.sum(
                membership**self.fuzzifier * distances_to_centroids**2,
            )
        )

    def _create_membership(self, n_samples: int) -> NDArrayFloat:
        return np.empty((n_samples, self.n_clusters))

    def _prediction_from_membership(
        self,
        membership_matrix: NDArrayFloat,
    ) -> NDArrayInt:
        return np.argmax(  # type: ignore[no-any-return]
            membership_matrix,
            axis=1,
        )

    def _update(
        self,
        fdata: Input,
        membership_matrix: NDArrayFloat,
        distances_to_centroids: NDArrayFloat,
        centroids: Input,
    ) -> None:
        # Divisions by zero allowed
        with np.errstate(divide='ignore'):
            distances_to_centers_raised = (
                distances_to_centroids**(2 / (1 - self.fuzzifier))
            )

        # Divisions infinity by infinity allowed
        with np.errstate(invalid='ignore'):
            membership_matrix[:, :] = (
                distances_to_centers_raised
                / np.sum(
                    distances_to_centers_raised,
                    axis=1,
                    keepdims=True,
                )
            )

        # inf / inf divisions should be 1 in this context
        membership_matrix[np.isnan(membership_matrix)] = 1

        membership_matrix_raised = np.power(
            membership_matrix,
            self.fuzzifier,
        )

        slice_denominator = (
            (slice(None),) + (np.newaxis,) * (fdata.data_matrix.ndim - 1)
        )
        centroids.data_matrix[:] = (
            np.einsum(
                'ij,i...->j...',
                membership_matrix_raised,
                fdata.data_matrix,
            )
            / np.sum(membership_matrix_raised, axis=0)[slice_denominator]
        )

    def predict_proba(
        self,
        X: Input,
        sample_weight: None = None,
    ) -> NDArrayFloat:
        """Predict the probability of belonging to each cluster.

        Args:
            X: Object whose samples are classified into different groups.
            sample_weight: present here for API consistency by convention.

        Returns:
            Probability of belonging to each cluster for each sample.

        """
        return self._predict_membership(X, sample_weight)

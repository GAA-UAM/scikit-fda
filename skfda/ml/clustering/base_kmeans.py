"""K-Means Algorithms Module."""

import numpy as np
from ...representation.grid import FDataGrid
from ...misc.metrics import pairwise_distance, lp_distance
from abc import abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
import warnings
from sklearn.utils import check_random_state

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


class BaseKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Base class to implement K-Means clustering algorithms.

    Class from which both :class:`K-Means
    <skfda.ml.clustering.base_kmeans.KMeans>` and
    :class:`Fuzzy K-Means <skfda.ml.clustering.base_kmeans.FuzzyKMeans>`
    classes inherit.
    """

    def __init__(self, n_clusters, init, metric, n_init, max_iter, tol,
                 random_state):
        """Initialization of the BaseKMeans class.

        Args:
            n_clusters (int, optional): Number of groups into which the samples are
                classified. Defaults to 2.
            init (FDataGrid, optional): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, and the centers are initialized randomly.
            metric (optional): metric that acceps two FDataGrid objects and returns
                a matrix with shape (fdatagrid1.nsamples, fdatagrid2.nsamples).
                Defaults to *pairwise_distance(lp_distance)*.
            n_init (int, optional): Number of time the k-means algorithm will be
                run with different centroid seeds. The final results will be the
                best output of n_init consecutive runs in terms of inertia.
            max_iter (int, optional): Maximum number of iterations of the
                clustering algorithm for a single run. Defaults to 100.
            tol (float, optional): tolerance used to compare the centroids
                calculated with the previous ones in every single run of the
                algorithm.
            random_state (int, RandomState instance or None, optional):
                Determines random number generation for centroid initialization. ç
                Use an int to make the randomness deterministic. Defaults to 0.
                See :term:`Glossary <random_state>`.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _generic_clustering_checks(self, fdatagrid):
        """Checks the arguments used in the
        :func:`fit method <skfda.ml.clustering.base_kmeans.fit>`.

        Args:
            fdatagrid (FDataGrid object): Object whose samples
                are classified into different groups.
        """

        if fdatagrid.ndim_domain > 1:
            raise NotImplementedError(
                "Only support 1 dimension on the domain.")

        if fdatagrid.nsamples < 2:
            raise ValueError(
                "The number of observations must be greater than 1.")

        if self.n_clusters < 2:
            raise ValueError(
                "The number of clusters must be greater than 1.")

        if self.n_init < 1:
            raise ValueError(
                "The number of iterations must be greater than 0.")

        if self.init is not None and self.n_init != 1:
            self.n_init = 1
            warnings.warn("Warning: The number of iterations is ignored "
                          "because the init parameter is set.")

        if self.init is not None and self.init.shape != (
                self.n_clusters, fdatagrid.ncol, fdatagrid.ndim_image):
            raise ValueError(
                "The init FDataGrid data_matrix should be of shape (n_clusters, "
                "n_features, ndim_image) and gives the initial centers.")

        if self.max_iter < 1:
            raise ValueError(
                "The number of maximum iterations must be greater than 0.")

        if self.tol < 0:
            raise ValueError("The tolerance must be positive.")

        return fdatagrid

    def _init_centroids(self, fdatagrid, random_state):
        """Compute the initial centroids

        Args:
            data_matrix (ndarray): matrix with the data only of the
                dimension of the image of the fdatagrid the algorithm is
                classifying.
            fdatagrid (FDataGrid object): Object whose samples are
                classified into different groups.
            random_state (RandomState object): random number generation for
                centroid initialization.

        Returns:
            centers (ndarray): initial centers
        """
        comparison = True
        while comparison:
            indices = random_state.permutation(fdatagrid.nsamples)[
                      :self.n_clusters]
            centers = fdatagrid.data_matrix[indices]
            unique_centers = np.unique(centers, axis=0)
            comparison = len(unique_centers) != self.n_clusters

        return centers

    @abstractmethod
    def fit(self, X, y=None, sample_weight=None):
        """ Computes clustering.

        Args:
            X (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
             y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.
        """
        pass

    def _check_is_fitted(self):
        """Perform is_fitted validation for estimator.

        Checks if the estimator is fitted by verifying the presence of
        of the calculated attributes "labels_" and "cluster_centers_", and
        raises a NotFittedError if that is not the case.
        """
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

        if not hasattr(self, "labels_") or \
                not hasattr(self, "cluster_centers_"):
            raise NotFittedError(msg % {'name': type(self).__name__})

    def _check_test_data(self, fdatagrid):
        """Checks that the FDataGrid object and the calculated centroids have
        compatible shapes.
        """
        if fdatagrid.shape[1:3] != self.cluster_centers_.shape[1:3]:
            raise ValueError("The fdatagrid shape is not the one expected for "
                             "the calculated cluster_centers_.")

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X (FDataGrid object): Object whose samples are classified into
                different groups.
            y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.

        Returns:
            labels_
        """
        self._check_is_fitted()
        self._check_test_data(X)
        return self.labels_

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            X (FDataGrid object): Object whose samples are classified into
                different groups.
            y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.

        Returns:
            labels_
        """
        self.fit(X)
        return self.labels_

    def transform(self, X):
        """Transform X to a cluster-distance space.

        Args:
            X (FDataGrid object): Object whose samples are classified into
                different groups.
            y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.

        Returns:
            distances_to_centers (numpy.ndarray: (nsamples, n_clusters)):
                distances of each sample to each cluster.
        """
        self._check_is_fitted()
        self._check_test_data(X)
        return self._distances_to_centers

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.

        Args:
            X (FDataGrid object): Object whose samples are classified into
                different groups.
            y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.

        Returns:
            distances_to_centers (numpy.ndarray: (nsamples, n_clusters)):
                distances of each sample to each cluster.
        """
        self.fit(X)
        return self._distances_to_centers

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Args:
            X (FDataGrid object): Object whose samples are classified into
                different groups.
            y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.

        Returns:
            score (numpy.array: (fdatagrid.ndim_image)): negative *inertia_*
                attribute.

        """
        self._check_is_fitted()
        self._check_test_data(X)
        return -self.inertia_


class KMeans(BaseKMeans):
    r"""Representation and implementation of the K-Means algorithm
    for the FdataGrid object.

    Let :math:`\mathbf{X = \left\{ x_{1}, x_{2}, ..., x_{n}\right\}}` be a given dataset to be
    analyzed, and :math:`\mathbf{V = \left\{ v_{1}, v_{2}, ..., v_{c}\right\}}` be the set of
    centers of clusters in :math:`\mathbf{X}` dataset in :math:`m` dimensional space
    :math:`\left(\mathbb{R}^m \right)`. Where :math:`n` is the number of objects, :math:`m` is the
    number of features, and :math:`c` is the number of partitions or clusters.

    KM iteratively computes cluster centroids in order to minimize the sum with respect to the specified
    measure. KM algorithm aims at minimizing an objective function known as the squared error function given
    as follows:

    .. math::
        J_{KM}\left(\mathbf{X}; \mathbf{V}\right) = \sum_{i=1}^{c}\sum_{j=1}^{n}D_{ij}^2

    Where, :math:`D_{ij}^2` is the squared chosen distance measure which can be any p-norm:
    :math:`D_{ij} = \lVert x_{ij} - v_{i} \rVert = \left( \int_I \lvert x_{ij} - v_{i}\rvert^p dx \right)^{ \frac{1}{p}}`,
    being :math:`I` the domain where :math:`\mathbf{X}` is defined, :math:`1 \leqslant i \leqslant c`,
    :math:`1 \leqslant j\leqslant n_{i}`. Where :math:`n_{i}` represents the number of data points in i-th cluster.

    For :math:`c` clusters, KM is based on an iterative algorithm minimizing the sum of distances from each
    observation to its cluster centroid. The observations are moved between clusters until the sum cannot be decreased
    any more. KM algorithm involves the following steps:

    1. Centroids of :math:`c` clusters are chosen from :math:`\mathbf{X}` randomly or are passed to the
        function as a parameter.

    2. Distances between data points and cluster centroids are calculated.

    3. Each data point is assigned to the cluster whose centroid is closest to it.

    4. Cluster centroids are updated by using the following formula:
        :math:`\mathbf{v_{i}} ={\sum_{i=1}^{n_{i}}x_{ij}}/n_{i}` :math:`1 \leqslant i \leqslant c`.

    5. Distances from the updated cluster centroids are recalculated.

    6. If no data point is assigned to a new cluster the run of algorithm is stopped, otherwise the
        steps from 3 to 5 are repeated for probable movements of data points between the clusters.

    This algorithm is applied for each dimension on the image of the FDataGrid object.

    Args:
        n_clusters (int, optional): Number of groups into which the samples are 
            classified. Defaults to 2.
        init (FDataGrid, optional): Contains the initial centers of the different
            clusters the algorithm starts with. Its data_marix must be of
            the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
            Defaults to None, and the centers are initialized randomly.
        metric (optional): metric that acceps two FDataGrid objects and returns 
            a matrix with shape (fdatagrid1.nsamples, fdatagrid2.nsamples). 
            Defaults to *pairwise_distance(lp_distance)*.
        n_init (int, optional): Number of time the k-means algorithm will be 
            run with different centroid seeds. The final results will be the 
            best output of n_init consecutive runs in terms of inertia.
        max_iter (int, optional): Maximum number of iterations of the 
            clustering algorithm for a single run. Defaults to 100.
        tol (float, optional): tolerance used to compare the centroids 
            calculated with the previous ones in every single run of the 
            algorithm.
        random_state (int, RandomState instance or None, optional): 
            Determines random number generation for centroid initialization. ç
            Use an int to make the randomness deterministic. Defaults to 0.
            See :term:`Glossary <random_state>`.
            
    Attributes:
        labels_ (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix 
            in which each row contains the cluster that observation belongs to.
        cluster_centers_ (FDataGrid object): data_matrix of shape 
            (n_clusters, ncol, ndim_image) and contains the centroids for 
            each cluster.
        inertia_ (numpy.ndarray, (fdatagrid.ndim_image)): Sum of squared 
            distances of samples to their closest cluster center for each 
            dimension.
        n_iter_ (numpy.ndarray, (fdatagrid.ndim_image)): number of iterations the
            algorithm was run for each dimension.

    Example:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> kmeans = KMeans()
        >>> init= np.array([[0, 0, 0, 0, 0, 0], [2, 1, -1, 0.5, 0, -0.5]])
        >>> init_fd = FDataGrid(init, sample_points)
        >>> kmeans.fit(fd, init=init_fd)
        >>> kmeans
        KMeans(max_iter=100,
            metric=<function pairwise_distance.<locals>.pairwise at 0x7faf3aa061e0>, # doctest:+ELLIPSIS
            n_clusters=2, random_state=0, tol=0.0001)
    """.replace('+IGNORE_RESULT', '+ELLIPSIS\n<...>')

    def __init__(self, n_clusters=2, init=None,
                 metric=pairwise_distance(lp_distance),
                 n_init=1, max_iter=100, tol=1e-4, random_state=0):
        """Initialization of the KMeans class.

        Args:
            n_clusters (int, optional): Number of groups into which the samples are
                classified. Defaults to 2.
            init (FDataGrid, optional): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, and the centers are initialized randomly.
            metric (optional): metric that acceps two FDataGrid objects and returns
                a matrix with shape (fdatagrid1.nsamples, fdatagrid2.nsamples).
                Defaults to *pairwise_distance(lp_distance)*.
            n_init (int, optional): Number of time the k-means algorithm will be
                run with different centroid seeds. The final results will be the
                best output of n_init consecutive runs in terms of inertia.
            max_iter (int, optional): Maximum number of iterations of the
                clustering algorithm for a single run. Defaults to 100.
            tol (float, optional): tolerance used to compare the centroids
                calculated with the previous ones in every single run of the
                algorithm.
            random_state (int, RandomState instance or None, optional):
                Determines random number generation for centroid initialization.
                Use an int to make the randomness deterministic. Defaults to 0.
        """
        super().__init__(n_clusters=n_clusters, init=init, metric=metric,
                         n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state)

    def _kmeans_implementation(self, fdatagrid, random_state):
        """ Implementation of the K-Means algorithm for FDataGrid objects
        of any dimension.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
            random_state (RandomState object): random number generation for
                centroid initialization.

        Returns:
            (tuple): tuple containing:

                clustering_values (numpy.ndarray: (nsamples,)): 1-dimensional
                array where each row contains the cluster that observation
                belongs to.

                centers (numpy.ndarray: (n_clusters, ncol, ndim_image)):
                Contains the centroids for each cluster.

                distances_to_centers (numpy.ndarray: (nsamples, n_clusters)):
                distances of each sample to each cluster.

                repetitions(int): number of iterations the algorithm was run.
        """
        repetitions = 0
        centers_old = np.zeros(
            (self.n_clusters, fdatagrid.ncol, fdatagrid.ndim_image))

        if self.init is None:
            centers = self._init_centroids(fdatagrid, random_state)
        else:
            centers = np.copy(self.init.data_matrix)

        while not np.allclose(centers, centers_old, rtol=self.tol,
                              atol=self.tol) and repetitions < self.max_iter:
            centers_old = np.copy(centers)
            centers_fd = FDataGrid(centers, fdatagrid.sample_points)
            distances_to_centers = self.metric(fdata1=fdatagrid,
                                               fdata2=centers_fd)
            clustering_values = np.argmin(distances_to_centers, axis=1)
            for i in range(self.n_clusters):
                indices, = np.where(clustering_values == i)
                if indices.size != 0:
                    centers[i] = np.average(
                        fdatagrid.data_matrix[indices, ...], axis=0)
            repetitions += 1

        return clustering_values, centers, distances_to_centers, repetitions

    def fit(self, X, y=None, sample_weight=None):
        """ Computes K-Means clustering calculating the attributes
        *labels_*, *cluster_centers_*, *inertia_* and *n_iter_*.

        Args:
            X (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
             y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.
        """
        random_state = check_random_state(self.random_state)
        fdatagrid = super()._generic_clustering_checks(fdatagrid=X)

        clustering_values = np.empty(
            (self.n_init, fdatagrid.nsamples)).astype(int)
        centers = np.empty((self.n_init, self.n_clusters,
                            fdatagrid.ncol, fdatagrid.ndim_image))
        distances_to_centers = np.empty(
            (self.n_init, fdatagrid.nsamples, self.n_clusters))
        distances_to_their_center = np.empty((self.n_init, fdatagrid.nsamples))
        n_iter = np.empty((self.n_init))

        for j in range(self.n_init):
            clustering_values[j, :], centers[j, :, :, :], \
            distances_to_centers[j, :, :], n_iter[j] = \
                self._kmeans_implementation(fdatagrid=fdatagrid,
                                            random_state=random_state)
            distances_to_their_center[j, :] = distances_to_centers[
                j, np.arange(fdatagrid.nsamples),
                clustering_values[j, :]]

        inertia = np.sum(distances_to_their_center ** 2, axis=1)
        index_best_iter = np.argmin(inertia)

        self.labels_ = clustering_values[index_best_iter]
        self.cluster_centers_ = FDataGrid(data_matrix=centers[index_best_iter],
                                          sample_points=fdatagrid.sample_points)
        self._distances_to_centers = distances_to_centers[index_best_iter]
        self.inertia_ = inertia[index_best_iter]
        self.n_iter_ = n_iter[index_best_iter]

        return self


class FuzzyKMeans(BaseKMeans):
    r""" Representation and implementation of the Fuzzy K-Means clustering
    algorithm for the FDataGrid object.

    Let :math:`\mathbf{X = \left\{ x_{1}, x_{2}, ..., x_{n}\right\}}` be a given dataset to be
    analyzed, and :math:`\mathbf{V = \left\{ v_{1}, v_{2}, ..., v_{c}\right\}}` be the set of
    centers of clusters in :math:`\mathbf{X}` dataset in :math:`m` dimensional space
    :math:`\left(\mathbb{R}^m \right)`. Where :math:`n` is the number of objects, :math:`m` is the
    number of features, and :math:`c` is the number of partitions or clusters.

    FCM minimizes the following objective function:

    .. math::
        J_{FCM}\left(\mathbf{X}; \mathbf{U, V}\right) = \sum_{i=1}^{c}\sum_{j=1}^{n}u_{ij}^{f}D_{ij}^2.

    This function differs from classical KM with the use of weighted squared errors instead of using squared
    errors only. In the objective function, :math:`\mathbf{U}` is a fuzzy partition matrix that is computed from
    dataset :math:`\mathbf{X}`: :math:`\mathbf{U} = [u_{ij}] \in M_{FCM}`.

    The fuzzy clustering of :math:`\mathbf{X}` is represented with :math:`\mathbf{U}` membership matrix. The element
    :math:`u_{ij}` is the membership value of j-th object to i-th cluster. In this case, the i-th row of :math:`\mathbf{U}`
    matrix is formed with membership values of :math:`n` objects to i-th cluster. :math:`\mathbf{V}` is a prototype vector
    of cluster prototypes (centroids): :math:`\mathbf{V = \left\{ v_{1}, v_{2}, ..., v_{c}\right\}}`,
    :math:`\mathbf{v_{i}}\in \mathbb{R}^m`.

    :math:`D_{ij}^2` is the squared chosen distance measure which can be any p-norm:
    :math:`D_{ij} =\lVert x_{ij} - v_{i} \rVert = \left( \int_I \lvert x_{ij} - v_{i}\rvert^p dx \right)^{ \frac{1}{p}}`,
    being :math:`I` the domain where :math:`\mathbf{X}` is defined, :math:`1 \leqslant i \leqslant c`,
    :math:`1 \leqslant j\leqslant n_{i}`. Where :math:`n_{i}` represents the number of data points in i-th cluster.

    FCM is an iterative process and stops when the number of iterations is reached to maximum, or when
    the centroids of the clusters do not change. The steps involved in FCM are:

        1. Centroids of :math:`c` clusters are chosen from :math:`\mathbf{X}` randomly or are passed to the
           function as a parameter.

        2. Membership values of data points to each cluster are calculated with:
           :math:`u_{ij} = \left[ \sum_{k=1}^c\left( D_{ij}/D_{kj}\right)^\frac{2}{f-1} \right]^{-1}`.

        3. Cluster centroids are updated by using the following formula:
           :math:`\mathbf{v_{i}} =\frac{\sum_{j=1}^{n}u_{ij}^f x_{j}}{\sum_{j=1}^{n} u_{ij}^f}`,
           :math:`1 \leqslant i \leqslant c`.

        4. If no cluster centroid changes the run of algorithm is stopped, otherwise return
           to step 2.

    This algorithm is applied for each dimension on the image of the FDataGrid object.

    Args:          
        n_clusters (int, optional): Number of groups into which the samples are 
            classified. Defaults to 2.
        init (FDataGrid, optional): Contains the initial centers of the different
            clusters the algorithm starts with. Its data_marix must be of
            the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
            Defaults to None, and the centers are initialized randomly.
        metric (optional): metric that acceps two FDataGrid objects and returns 
            a matrix with shape (fdatagrid1.nsamples, fdatagrid2.nsamples). 
            Defaults to *pairwise_distance(lp_distance)*.
        n_init (int, optional): Number of time the k-means algorithm will be 
            run with different centroid seeds. The final results will be the 
            best output of n_init consecutive runs in terms of inertia.
        max_iter (int, optional): Maximum number of iterations of the 
            clustering algorithm for a single run. Defaults to 100.
        tol (float, optional): tolerance used to compare the centroids 
            calculated with the previous ones in every single run of the 
            algorithm.
        random_state (int, RandomState instance or None, optional): 
            Determines random number generation for centroid initialization. ç
            Use an int to make the randomness deterministic. Defaults to 0.
            See :term:`Glossary <random_state>`.
        fuzzifier (int, optional): Scalar parameter used to specify the
            degree of fuzziness in the fuzzy algorithm. Defaults to 2.
        n_dec (int, optional): designates the number of decimals of the labels
            returned in the fuzzy algorithm. Defaults to 3.
            
    Attributes:
        labels_ (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix 
            in which each row contains the cluster that observation belongs to.
        cluster_centers_ (FDataGrid object): data_matrix of shape 
            (n_clusters, ncol, ndim_image) and contains the centroids for 
            each cluster.
        inertia_ (numpy.ndarray, (fdatagrid.ndim_image)): Sum of squared 
            distances of samples to their closest cluster center for each 
            dimension.
        n_iter_ (numpy.ndarray, (fdatagrid.ndim_image)): number of iterations the
            algorithm was run for each dimension.


    Example:

        >>> data_matrix = [[[1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
        ...                [[2, 0.5], [3, 0.6], [4, 0.7], [5, 0.7]],
        ...                [[3, 0.2], [4, 0.3], [5, 0.4], [6, 0.5]]]
        >>> sample_points = [2, 4, 6, 8]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fuzzy_kmeans = FuzzyKMeans()
        >>> init=np.array([[[3, 0], [5, 0], [2, 0], [4, 0]],
        ...                [[0, 0], [0, 1], [0, 0], [0, 1]]])
        >>> init_fd = FDataGrid(init, sample_points)
        >>> fuzzy_kmeans.fit(fd, init=init_fd)
        >>> fuzzy_kmeans
        FuzzyKMeans(fuzzifier=2, max_iter=100,
              metric=<function pairwise_distance.<locals>.pairwise at 0x7faf3aa06488>,  # doctest:+ELLIPSIS
              n_clusters=2, n_dec=3, random_state=0, tol=0.0001)
    """.replace('+IGNORE_RESULT', '+ELLIPSIS\n<...>')

    def __init__(self, n_clusters=2, init=None,
                 metric=pairwise_distance(lp_distance), n_init=1, max_iter=100,
                 tol=1e-4, random_state=0, fuzzifier=2, n_dec=3):
        """Initialization of the FuzzyKMeans class.

        Args:
            n_clusters (int, optional): Number of groups into which the samples are
                classified. Defaults to 2.
            init (FDataGrid, optional): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, and the centers are initialized randomly.
            metric (optional): metric that acceps two FDataGrid objects and returns
                a matrix with shape (fdatagrid1.nsamples, fdatagrid2.nsamples).
                Defaults to *pairwise_distance(lp_distance)*.
            n_init (int, optional): Number of time the k-means algorithm will be
                run with different centroid seeds. The final results will be the
                best output of n_init consecutive runs in terms of inertia.
            max_iter (int, optional): Maximum number of iterations of the
                clustering algorithm for a single run. Defaults to 100.
            tol (float, optional): tolerance used to compare the centroids
                calculated with the previous ones in every single run of the
                algorithm.
            random_state (int, RandomState instance or None, optional):
                Determines random number generation for centroid initialization.
                Use an int to make the randomness deterministic. Defaults to 0.
            fuzzifier (int, optional): Scalar parameter used to specify the
                degree of fuzziness in the fuzzy algorithm. Defaults to 2.
            n_dec (int, optional): designates the number of decimals of the labels
                returned in the fuzzy algorithm. Defaults to 3.
        """
        super().__init__(n_clusters=n_clusters, init=init, metric=metric,
                         n_init=n_init,
                         max_iter=max_iter, tol=tol, random_state=random_state)

        self.fuzzifier = fuzzifier
        self.n_dec = n_dec

    def _fuzzy_kmeans_implementation(self, fdatagrid, random_state):
        """ Implementation of the Fuzzy K-Means algorithm for FDataGrid objects
        of any dimension.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
            random_state (RandomState object): random number generation for
                centroid initialization.

        Returns:
            (tuple): tuple containing:

                membership values (numpy.ndarray: (nsamples, n_clusters)):
                2-dimensional matrix where each row contains the membership
                value that observation has to each cluster.

                centers (numpy.ndarray: (n_clusters, ncol, ndim_image)):
                Contains the centroids for each cluster.

                distances_to_centers (numpy.ndarray: (nsamples, n_clusters)):
                distances of each sample to each cluster.

                repetitions(int): number of iterations the algorithm was run.

        """
        repetitions = 0
        centers_old = np.zeros(
            (self.n_clusters, fdatagrid.ncol, fdatagrid.ndim_image))
        U = np.empty((fdatagrid.nsamples, self.n_clusters))
        distances_to_centers = np.empty((fdatagrid.nsamples, self.n_clusters))

        if self.init is None:
            centers = self._init_centroids(fdatagrid, random_state)
        else:
            centers = np.copy(self.init.data_matrix)

        while not np.allclose(centers, centers_old, rtol=self.tol,
                              atol=self.tol) and repetitions < self.max_iter:

            centers_old = np.copy(centers)
            centers_fd = FDataGrid(centers, fdatagrid.sample_points)
            distances_to_centers = self.metric(
                fdata1=fdatagrid,
                fdata2=centers_fd)
            distances_to_centers_raised = distances_to_centers ** (
                                2 / (self.fuzzifier - 1))

            for i in range(fdatagrid.nsamples):
                comparison = (fdatagrid.data_matrix[i] == centers).all(
                    axis=tuple(np.arange(fdatagrid.ndim)[1:]))
                if comparison.sum() >= 1:
                    U[i, np.where(comparison == True)] = 1
                    U[i, np.where(comparison == False)] = 0
                else:
                    for j in range(self.n_clusters):
                        U[i, j] = 1 / np.sum(
                            distances_to_centers_raised[i, j] /
                            distances_to_centers_raised[i])

            U = np.power(U, self.fuzzifier)
            for i in range(self.n_clusters):
                centers[i] = np.sum((U[:, i] * fdatagrid.data_matrix.T).T,
                                    axis=0) / np.sum(U[:, i])
            repetitions += 1

        return np.round(np.power(U, 1 / self.fuzzifier), self.n_dec), centers, \
               distances_to_centers, repetitions

    def fit(self, X, y=None, sample_weight=None):
        """ Computes Fuzzy K-Means clustering calculating the attributes
        *labels_*, *cluster_centers_*, *inertia_* and *n_iter_*.

        Args:
            X (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
             y (Ignored): present here for API consistency by convention.
            sample_weight (Ignored): present here for API consistency by
                convention.
        """
        fdatagrid = super()._generic_clustering_checks(fdatagrid=X)
        random_state = check_random_state(self.random_state)

        if self.fuzzifier < 2:
            raise ValueError("The fuzzifier parameter must be greater than 1.")

        if self.n_dec < 1:
            raise ValueError(
                "The number of decimals should be greater than 0 in order to "
                "obtain a rational result.")

        membership_values = np.empty(
            (self.n_init, fdatagrid.nsamples, self.n_clusters))
        centers = np.empty(
            (self.n_init, self.n_clusters, fdatagrid.ncol,
             fdatagrid.ndim_image))
        distances_to_centers = np.empty(
            (self.n_init, fdatagrid.nsamples, self.n_clusters))
        distances_to_their_center = np.empty((self.n_init, fdatagrid.nsamples))
        n_iter = np.empty((self.n_init))

        for j in range(self.n_init):
            membership_values[j, :, :], centers[j, :, :, :], \
            distances_to_centers[j, :, :], n_iter[j] = \
                self._fuzzy_kmeans_implementation(fdatagrid=fdatagrid,
                                                  random_state=random_state)
            distances_to_their_center[j, :] = distances_to_centers[
                j, np.arange(fdatagrid.nsamples),
                np.argmax(membership_values[j, :, :], axis=-1)]

        inertia = np.sum(distances_to_their_center ** 2, axis=1)
        index_best_iter = np.argmin(inertia)

        self.labels_ = membership_values[index_best_iter]
        self.cluster_centers_ = FDataGrid(data_matrix=centers[index_best_iter],
                                          sample_points=fdatagrid.sample_points)
        self._distances_to_centers = distances_to_centers[index_best_iter]
        self.inertia_ = inertia[index_best_iter]
        self.n_iter_ = n_iter[index_best_iter]

        return self

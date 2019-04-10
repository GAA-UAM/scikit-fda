"""Clustering Module."""

import numpy as np
from .grid import FDataGrid
from .math import metric, norm_lp
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import random
from abc import ABC, abstractmethod

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"

class ClusteringData(ABC):

    def __init__(self, fdatagrid, n_clusters=2, init=None, max_iter=100, seed=None):
        """Checks the arguments passed to both :func:`K-Means <fda.clustering.kmeans>` and
        :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` functions.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are classified into different groups.
            n_clusters (int): Number of groups into which the samples are classified.
            init (ndarray): Contains the initial centers of the different clusters the algorithm starts with or None.
            max_iter (int): Maximum number of iterations of the clustering algorithm.
            fuzzifier (int, optional): Scalar parameter used to specify the degree of fuzziness in the fuzzy algorithm.
                Defaults to 2.
            n_dec (int, optional): designates the number of decimals of the labels returned in the fuzzy algorithm.
                Defaults to 3.

        Returns:
            init (ndarray): In case all checks have passed, the init parameter.
        """

        if fdatagrid.ndim_domain > 1:
            raise NotImplementedError(
                "Only support 1 dimension on the domain.")

        if fdatagrid.nsamples < 2:
            raise ValueError(
                "The number of observations must be greater than 1.")

        if n_clusters < 2:
            raise ValueError("The number of clusters must be greater than 1.")

        if init is not None and init.shape != (
                fdatagrid.ndim_image, n_clusters, fdatagrid.ncol):
            raise ValueError(
                "The init ndarray should be of shape (ndim_image, n_clusters, n_features) "
                "and gives the initial centers.")
        else:
            init = np.array([None] * fdatagrid.ndim_image)

        if max_iter < 1:
            raise ValueError(
                "The number of iterations must be greater than 0.")

        if fuzzifier < 2:
            raise ValueError("The fuzzifier parameter must be greater than 1.")

        if n_dec < 1:
            raise ValueError(
                "The number of decimals should be greater than 0 in order to "
                "obatain a rational result.")

        self._fdatagrid = fdatagrid
        self._n_clusters = n_clusters
        self._init = init
        self._max_iter = max_iter
        self._seed = seed
        # self._fuzzifier = fuzzifier
        # self._n_dec = n_dec


        @property
        def fdatagrid(self):
            return self._fdatagrid

        @property
        def n_clusters(self):
            return self._n_clusters

        @property
        def init(self):
            return self._init

        @property
        def max_iter(self):
            return self._max_iter

        @property
        def seed(self):
            return self._seed

        @property
        def centers(self):
            return self._centers

        # @property
        # def fuzzifier(self):
        #     return self._fuzzifier
        #
        # @property
        # def n_dec(self):
        #     return self._n_dec

        def _random_initialization_centers(data_matrix):
            """Random initialization of the centroids in both :func:`K-Means <fda.clustering.clustering>`
            and :func:`Fuzzy K-Means <fda.clustering.fuzzy_clustering>` functions if they are not passed
            as a parameter.

            Args:
                fdatagrid (FDataGrid object): Object whose samples are classified into different groups.
                data_matrix (ndarray): matrix with the data only of the dimension of the image of the
                    fdatagrid the algorithm is classifying.
                n_clusters (int): Number of groups into which the samples are classified.
                seed (int): Seed to initialize the random state to choose the initial centroids.

            Returns:
                centers (ndarray): initial centers
            """
            comparison = True
            if seed is None:
                seed_aux = 0
            else:
                seed_aux = seed
            while comparison:
                random.seed(seed_aux)
                indices = random.sample(range(fdatagrid.nsamples), n_clusters)
                centers = data_matrix[indices]
                comparison = np.asarray([(centers[i] == centers).all()
                                         for i in range(n_clusters)]).sum()
                seed_aux += 1

        self._centers = centers




        # @show_full_outliers.setter
        # def show_full_outliers(self, boolean):
        #     if not isinstance(boolean, bool):
        #         raise ValueError("show_full_outliers must be boolean type")
        #     self._show_full_outliers = boolean

class KMeans(ClusteringData):

    def __init__(self, fdatagrid, n_clusters=2, init=None, max_iter=100,
                 seed=None, p=2):
        super().__init__(fdatagrid, n_clusters, init, max_iter, seed)
        self._p = p
        self._clustering_values = np.empty(
            (self.fdatagrid.nsamples, self.fdatagrid.ndim_image))
        self._centers = np.empty((self.fdatagrid.ndim_image, self.n_clusters, self.fdatagrid.ncol))

    @property
    def p(self):
        return self._p

    def _kmeans_1Dimage(num_dim):
        """ Implementation of the K-Means algorithm for each dimension on the image of the FDataGrid object.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
            num_dim (int): Scalar indicating the dimension on the image of the FdataGrid object the algorithm is
                being applied..
            n_clusters (int): Number of groups into which the samples are classified.
            centers (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
                Defaults to None, ans the centers are initialized randomly.
            seed (int): Seed to initialize the random state to choose the initial centroids.
            max_iter (int): Maximum number of iterations of the clustering algorithm.
            p (int): Identifies the p-norm used to calculate the distance between functions.

        Returns:
            (tuple): tuple containing:

                clustering_values (numpy.ndarray: (nsamples,)): 1-dimensional array where each row
                contains the cluster that observation belongs to.

                centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids for each cluster.

        """

        data_matrix = np.copy(self.fdatagrid.data_matrix[:, :, num_dim])
        fdatagrid_1dim = FDataGrid(data_matrix=data_matrix,
                                   sample_points=self.fdatagrid.sample_points[0])
        repetitions = 0
        centers_old = np.empty((self.n_clusters, self.fdatagrid.ncol))

        if centers is None:
            centers = super()._random_initialization_centers(data_matrix)

        while not np.array_equal(centers,
                                 centers_old) and repetitions < max_iter:
            centers_old = np.copy(centers)
            centers_fd = FDataGrid(centers, fdatagrid.sample_points)
            distances_to_centers = metric(fdatagrid=fdatagrid_1dim,
                                          fdatagrid2=centers_fd, p=p)
            clustering_values = np.argmin(distances_to_centers, axis=1)
            for i in range(n_clusters):
                indices = np.where(clustering_values == i)
                centers[i] = np.average(data_matrix[indices, :], axis=1)
            repetitions += 1

        return clustering_values, centers

    def fit(self):
        r"""Implementation of the K-Means algorithm for the FdataGrid object.

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
                fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
                n_clusters (int, optional): Number of groups into which the samples are classified. Defaults to 2.
                init (ndarray, optional): Contains the initial centers of the different clusters the algorithm starts with.
                    Defaults to None, ans the centers are initialized randomly.
                seed (int, optional): Seed to initialize the random state to choose the initial centroids.
                    Defaults to None.
                max_iter (int, optional): Maximum number of iterations of the clustering algorithm. Defaults to 100.
                p (int, optional): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.

            Returns:
                (tuple): tuple containing:

                    clustering_values (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix where each row
                    contains the cluster that observation belongs to.

                    centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.

            """
        # init = _generic_clustering_checks(fdatagrid, n_clusters, init,
        #                                   max_iter)
        clustering_values = np.empty(
            (self.fdatagrid.nsamples, self.fdatagrid.ndim_image))
        centers = np.empty((self.fdatagrid.ndim_image, self.n_clusters, self.fdatagrid.ncol))
        for i in range(self.fdatagrid.ndim_image):
            clustering_values[:, i], centers[i, :, :] = _kmeans_1Dimage(num_dim=i)

        self._clustering_values = clustering_values
        self._centers = centers





#
# def _generic_clustering_checks(fdatagrid, n_clusters, init, max_iter,
#                                fuzzifier=2, n_dec=3):
#     """Checks the arguments passed to both :func:`K-Means <fda.clustering.kmeans>` and
#     :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` functions.
#
#     Args:
#         fdatagrid (FDataGrid object): Object whose samples are classified into different groups.
#         n_clusters (int): Number of groups into which the samples are classified.
#         init (ndarray): Contains the initial centers of the different clusters the algorithm starts with or None.
#         max_iter (int): Maximum number of iterations of the clustering algorithm.
#         fuzzifier (int, optional): Scalar parameter used to specify the degree of fuzziness in the fuzzy algorithm.
#             Defaults to 2.
#         n_dec (int, optional): designates the number of decimals of the labels returned in the fuzzy algorithm.
#             Defaults to 3.
#
#     Returns:
#         init (ndarray): In case all checks have passed, the init parameter.
#     """
#
#     if fdatagrid.ndim_domain > 1:
#         raise NotImplementedError("Only support 1 dimension on the domain.")
#
#     if fdatagrid.nsamples < 2:
#         raise ValueError("The number of observations must be greater than 1.")
#
#     if n_clusters < 2:
#         raise ValueError("The number of clusters must be greater than 1.")
#
#     if init is not None and init.shape != (
#             fdatagrid.ndim_image, n_clusters, fdatagrid.ncol):
#         raise ValueError(
#             "The init ndarray should be of shape (ndim_image, n_clusters, n_features) "
#             "and gives the initial centers.")
#     else:
#         init = np.array([None] * fdatagrid.ndim_image)
#
#     if max_iter < 1:
#         raise ValueError("The number of iterations must be greater than 0.")
#
#     if fuzzifier < 2:
#         raise ValueError("The fuzzifier parameter must be greater than 1.")
#
#     if n_dec < 1:
#         raise ValueError(
#             "The number of decimals should be greater than 0 in order to obatain a rational result.")
#
#     return init





def _kmeans_1Dimage(fdatagrid, num_dim, n_clusters, centers, seed, max_iter,
                    p):
    """ Implementation of the K-Means algorithm for each dimension on the image of the FDataGrid object.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        num_dim (int): Scalar indicating the dimension on the image of the FdataGrid object the algorithm is
            being applied..
        n_clusters (int): Number of groups into which the samples are classified.
        centers (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        seed (int): Seed to initialize the random state to choose the initial centroids.
        max_iter (int): Maximum number of iterations of the clustering algorithm.
        p (int): Identifies the p-norm used to calculate the distance between functions.

    Returns:
        (tuple): tuple containing:

            clustering_values (numpy.ndarray: (nsamples,)): 1-dimensional array where each row
            contains the cluster that observation belongs to.

            centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids for each cluster.

    """

    data_matrix = np.copy(fdatagrid.data_matrix[:, :, num_dim])
    fdatagrid_1dim = FDataGrid(data_matrix=data_matrix,
                               sample_points=fdatagrid.sample_points[0])
    repetitions = 0
    centers_old = np.empty((n_clusters, fdatagrid.ncol))

    if centers is None:
        centers = _random_initialization_centers(fdatagrid, data_matrix,
                                                 n_clusters,
                                                 seed)

    while not np.array_equal(centers, centers_old) and repetitions < max_iter:
        centers_old = np.copy(centers)
        centers_fd = FDataGrid(centers, fdatagrid.sample_points)
        distances_to_centers = metric(fdatagrid=fdatagrid_1dim,
                                      fdatagrid2=centers_fd, p=p)
        clustering_values = np.argmin(distances_to_centers, axis=1)
        for i in range(n_clusters):
            indices = np.where(clustering_values == i)
            centers[i] = np.average(data_matrix[indices, :], axis=1)
        repetitions += 1

    return clustering_values, centers


def kmeans(fdatagrid, n_clusters=2, init=None, seed=None, max_iter=100, p=2):
    r"""Implementation of the K-Means algorithm for the FdataGrid object.

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
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        n_clusters (int, optional): Number of groups into which the samples are classified. Defaults to 2.
        init (ndarray, optional): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        seed (int, optional): Seed to initialize the random state to choose the initial centroids.
            Defaults to None.
        max_iter (int, optional): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        p (int, optional): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.

    Returns:
        (tuple): tuple containing:

            clustering_values (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix where each row
            contains the cluster that observation belongs to.

            centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.

    """
    init = _generic_clustering_checks(fdatagrid, n_clusters, init, max_iter)
    clustering_values = np.empty((fdatagrid.nsamples, fdatagrid.ndim_image))
    centers = np.empty((fdatagrid.ndim_image, n_clusters, fdatagrid.ncol))
    for i in range(fdatagrid.ndim_image):
        clustering_values[:, i], centers[i, :, :] = _kmeans_1Dimage(fdatagrid,
                                                                    num_dim=i,
                                                                    n_clusters=n_clusters,
                                                                    centers=
                                                                    init[i],
                                                                    seed=seed,
                                                                    max_iter=max_iter,
                                                                    p=p)

    return clustering_values, centers


def _fuzzy_kmeans_1Dimage(fdatagrid, num_dim, n_clusters, fuzzifier, centers,
                          seed, max_iter, p, n_dec):
    """ Implementation of the Fuzzy C-Means algorithm for each dimension on the image of the FDataGrid object.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        num_dim (int): Scalar indicating the dimension on the image of the FdataGrid object the algorithm is
            being applied..
        n_clusters (int): Number of groups into which the samples are classified.
        fuzzifier (int): Scalar parameter used to specify the degree of fuzziness.
        centers (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        seed (int): Seed to initialize the random state to choose the initial centroids.
        max_iter (int): Maximum number of iterations of the clustering algorithm.
        p (int): Identifies the p-norm used to calculate the distance between functions.
        n_dec (int): designates the number of decimals of the labels returned.

    Returns:
        (tuple): tuple containing:

            membership values (numpy.ndarray: (n_clusters, nsamples)): 2-dimensional matrix where each row
            contains the membership value that observation has to each cluster.

            centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids for each cluster.

    """

    data_matrix = np.copy(fdatagrid.data_matrix[:, :, num_dim])
    repetitions = 0
    centers_old = np.empty((n_clusters, fdatagrid.ncol))
    U = np.empty((n_clusters, fdatagrid.nsamples))

    if centers is None:
        centers = _random_initialization_centers(fdatagrid, data_matrix,
                                                 n_clusters,
                                                 seed)

    while not np.array_equal(centers, centers_old) and repetitions < max_iter:
        centers_old = np.copy(centers)
        for i in range(fdatagrid.nsamples):
            comparison = (data_matrix[i] == centers).all(-1)
            if comparison.sum() >= 1:
                U[np.where(comparison == True), i] = 1
                U[np.where(comparison == False), i] = 0
            else:
                diff = data_matrix[i] - centers
                diff_fd = FDataGrid(diff, fdatagrid.sample_points)
                distances_to_centers = np.power(norm_lp(diff_fd, p),
                                                2 / (fuzzifier - 1))
                for j in range(n_clusters):
                    U[j, i] = 1 / np.sum(
                        distances_to_centers[j] / distances_to_centers)
        U = np.power(U, fuzzifier)

        for i in range(n_clusters):
            centers[i] = np.sum((U[i] * data_matrix.T).T, axis=0) / np.sum(
                U[i])
        repetitions += 1

    return np.round(np.power(U, 1 / fuzzifier), n_dec), centers


def fuzzy_kmeans(fdatagrid, n_clusters=2, init=None, seed=None, max_iter=100,
                 fuzzifier=2, n_dec=3, p=2):
    r""" Implementation of the Fuzzy C-Means algorithm for the FDataGrid object.

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
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        num_dim (int): Scalar indicating the dimension on the image of the FdataGrid object the algorithm is
            being applied..
        n_clusters (int): Number of groups into which the samples are classified.
        fuzzifier (int): Scalar parameter used to specify the degree of fuzziness.
        init (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        seed (int, optional): Seed to initialize the random state to choose the initial centroids.
            Defaults to None.
        max_iter (int): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        p (int): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
        n_dec (int): designates the number of decimals of the labels returned.

    Returns:
        (tuple): tuple containing:

            membership values (numpy.ndarray: (n_clusters, nsamples)): 2-dimensional matrix where each row
            contains the membership value that observation has to each cluster.

            centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids for each cluster.

    """

    init = _generic_clustering_checks(fdatagrid, n_clusters, init, max_iter,
                                      fuzzifier, n_dec)
    membership_values = np.empty(
        (fdatagrid.nsamples, fdatagrid.ndim_image, n_clusters))
    centers = np.empty((fdatagrid.ndim_image, n_clusters, fdatagrid.ncol))
    for i in range(fdatagrid.ndim_image):
        U, centers[i, :, :] = _fuzzy_kmeans_1Dimage(fdatagrid, num_dim=i,
                                                    n_clusters=n_clusters,
                                                    fuzzifier=fuzzifier,
                                                    centers=init[i], seed=seed,
                                                    max_iter=max_iter, p=p,
                                                    n_dec=n_dec)
        membership_values[:, i, :] = U.T

    return membership_values, centers


def _labels_checks(fdatagrid, xlabels, ylabels, title, xlabel_str):
    """Checks the arguments xlabels, ylabels, title passed to both
    :func:`plot_fuzzy_kmeans_lines <fda.clustering.plot_fuzzy_kmeans_lines>` and
    :func:`plot_fuzzy_kmeans_bars <fda.clustering.plot_fuzzy_kmeans_bars>`
    functions. In case they are not set yet, hey are given a value.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are classified into different groups.
        xlabels (list of str): Labels for the x-axes.
        ylabels (list of str): Labels for the y-axes.
        title (str): Title for the figure where the clustering results are ploted.
        xlabel_str (str): In case xlabels is None, string to use fro the labels
            in the x-axes.

    Returns:
        xlabels (list of str): Labels for the x-axes.
        ylabels (list of str): Labels for the y-axes.
        title (str): Title for the figure where the clustering results are ploted.
    """

    if xlabels is not None and len(xlabels) != fdatagrid.ndim_image:
        raise ValueError(
            "xlabels must contain a label for each dimension on the domain.")

    if ylabels is not None and len(ylabels) != fdatagrid.ndim_image:
        raise ValueError(
            "xlabels must contain a label for each dimension on the domain.")

    if xlabels is None:
        xlabels = [xlabel_str] * fdatagrid.ndim_image

    if ylabels is None:
        ylabels = ["Membership grade"] * fdatagrid.ndim_image

    if title is None:
        title = "Membership grades of the samples to each cluster"

    return xlabels, ylabels, title


def _plot_clustering_checks(fdatagrid, n_clusters, sample_colors,
                            sample_labels, cluster_colors, cluster_labels,
                            center_colors, center_labels):
    """Checks the arguments sample_colors, sample_labels, cluster_colors,
    cluster_labels, center_colors, center_labels passed to functions
    :func:`plot_clustering <fda.clustering.plot_clustering>`,
    :func:`plot_fuzzy_kmeans_lines <fda.clustering.plot_fuzzy_kmeans_lines>` and
    :func:`plot_fuzzy_kmeans_bars <fda.clustering.plot_fuzzy_kmeans_bars>` have
    the correct dimensions.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are classified into different groups.
        n_clusters (int): Number of groups into which the samples are classified.
        sample_colors (list of colors): contains in order the colors of each
            sample of the fdatagrid.
        sample_labels (list of str): contains in order the labels of each sample
            of the fdatagrid.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        center_labels list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are classified into.

    """

    if sample_colors is not None and len(sample_colors) != fdatagrid.nsamples:
        raise ValueError("sample_colors must contain a color for each sample.")

    if sample_labels is not None and len(sample_labels) != fdatagrid.nsamples:
        raise ValueError("sample_labels must contain a label for each sample.")

    if cluster_colors is not None and len(cluster_colors) != n_clusters:
        raise ValueError(
            "cluster_colors must contain a color for each cluster.")

    if cluster_labels is not None and len(cluster_labels) != n_clusters:
        raise ValueError(
            "cluster_labels must contain a label for each cluster.")

    if center_colors is not None and len(center_colors) != n_clusters:
        raise ValueError("center_colors must contain a color for each center.")

    if center_labels is not None and len(center_labels) != n_clusters:
        raise ValueError(
            "centers_labels must contain a label for each center.")


def plot_clustering(fdatagrid, n_clusters=2, method=kmeans, init=None,
                    seed=None, max_iter=100, fuzzifier=2, n_dec=3, p=2,
                    fig=None, ax=None, nrows=None, ncols=None,
                    sample_labels=None, cluster_colors=None,
                    cluster_labels=None, center_colors=None,
                    center_labels=None,
                    colormap=plt.cm.get_cmap('rainbow')):
    """Implementation of the plotting of the samples of the FDataGrid divided
    by clusters.

    A clustering function, passed as argument, is called obtaining the classification
    of the samples into the desired number of clusters. In case the method used is
    :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>`, the cluster with the maximum
    membership value is chosen. According to the grouping,
    each sample is assigned a color, then, the plotting is done, adding a legend.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        n_clusters (int): Number of groups into which the samples are classified.
        method (clustering method, optional): clustering function to classify the data.
            Defaults to :func:`K-Means <fda.clustering.kmeans>`.
        init (ndarray, optional): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, and the centers are initialized randomly.
        seed (int, optional): Seed to initialize the random state to choose the initial centroids.
            Defaults to None.
        max_iter (int, optional): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        fuzzifier (int, optional): Scalar parameter used to specify the degree of fuzziness.
        n_dec (int, optional): designates the number of decimals of the labels returned.
        p (int, optional): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
        fig (figure object, optional): figure over which the graphs are plotted in case ax is not specified.
            If None and ax is also None, the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are plotted. If None, see param fig.
        nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        cluster_colors (list of colors, optional): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str, optional): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors, optional): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        center_labels list of colors, optional): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the plot are taken.

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

            clustering_values (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix where each row
            contains the cluster that observation belongs to.

            centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.


    """
    if method == kmeans:
        labels, centers = kmeans(fdatagrid, n_clusters=n_clusters, init=init,
                                 seed=seed, max_iter=max_iter)
        labels = labels.astype(int)
    elif method == fuzzy_kmeans:
        labels, centers = fuzzy_kmeans(fdatagrid, n_clusters=n_clusters,
                                       init=init, seed=seed, max_iter=max_iter,
                                       fuzzifier=fuzzifier, n_dec=n_dec, p=p)
        labels = np.argmax(labels, axis=-1)
    else:
        raise ValueError("method must be clustering or fuzzy_clustering")

    return _plot_clustering_implementation(fdatagrid, labels, centers,
                                          n_clusters=n_clusters, fig=fig,
                                          ax=ax, nrows=nrows, ncols=ncols,
                                          sample_labels=sample_labels,
                                          cluster_colors=cluster_colors,
                                          cluster_labels=cluster_labels,
                                          center_colors=center_colors,
                                          center_labels=center_labels,
                                          colormap=colormap)


def _plot_clustering_implementation(fdatagrid, labels, centers, n_clusters, fig,
                                   ax, nrows, ncols, sample_labels,
                                   cluster_colors, cluster_labels,
                                   center_colors, center_labels, colormap):

    """Internal implementation of the :func:`plot_clustering method <fda.clustering.plot_clustering>`.

    Once each sample is assigned a label, which are passed as argument to this
    function, the plotting is implemented here. Including the generation of a
    legend.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        labels (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix where each row
            contains the cluster that observation belongs to.
        n_clusters (int): Number of groups into which the samples are classified.
        fig (figure object): figure over which the graphs are plotted in case ax is not specified.
            If None and ax is also None, the figure is initialized.
        ax (list of axis objects): axis over where the graphs are plotted. If None, see param fig.
        nrows(int): designates the number of rows of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        ncols(int): designates the number of columns of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        cluster_colors (list of colors): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        center_colors (list of colors): contains in order the colors of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        center_labels list of colors): contains in order the labels of each
            centroid of the clusters the samples of the fdatagrid are classified into.
        colormap(colormap): colormap from which the colors of the plot are taken.

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

            clustering_values (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix where each row
            contains the cluster that observation belongs to.

            centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.


    """
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    _plot_clustering_checks(fdatagrid, n_clusters, None, sample_labels,
                            cluster_colors, cluster_labels,
                            center_colors, center_labels)

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                         range(fdatagrid.nsamples)]

    if cluster_colors is None:
        cluster_colors = colormap(np.arange(n_clusters) / (n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = ['$CLUSTER: {}$'.format(i) for i in range(n_clusters)]

    if center_colors is None:
        center_colors = ["black"] * n_clusters

    if center_labels is None:
        center_labels = ['$CENTER: {}$'.format(i) for i in range(n_clusters)]

    colors_by_cluster = cluster_colors[labels]

    patches = []
    for i in range(n_clusters):
        patches.append(
            mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]))

    for j in range(fdatagrid.ndim_image):
        for i in range(fdatagrid.nsamples):
            ax[j].plot(fdatagrid.sample_points[0],
                       fdatagrid.data_matrix[i, :, j],
                       c=colors_by_cluster[i, j],
                       label=sample_labels[i])
        for i in range(n_clusters):
            ax[j].plot(fdatagrid.sample_points[0], centers[j, i, :],
                       c=center_colors[i], label=center_labels[i])
        ax[j].legend(handles=patches)
        datacursor(formatter='{label}'.format)

    fdatagrid.set_labels(fig, ax)

    return fig, ax, labels, centers


def plot_fuzzy_kmeans_lines(fdatagrid, n_clusters=2, init=None, seed=None,
                            max_iter=100, fuzzifier=2, n_dec=3, p=2,
                            fig=None, ax=None, nrows=None, ncols=None,
                            sample_colors=None, sample_labels=None,
                            cluster_labels=None,
                            colormap=plt.cm.get_cmap('rainbow'),
                            xlabels=None, ylabels=None, title=None):
    """Implementation of the plotting of the results of the
    :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


    A kind of Parallel Coordinates plot is generated in this function with the
    membership values obtained from the algorithm. A line is plotted for each
    sample with the values for each cluster. See `Clustering Example
    <../auto_examples/plot_clustering.html>`_.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        n_clusters (int): Number of groups into which the samples are classified.
        init (ndarray, optional): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, and the centers are initialized randomly.
        seed (int, optional): Seed to initialize the random state to choose the initial centroids.
            Defaults to None.
        max_iter (int, optional): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        fuzzifier (int, optional): Scalar parameter used to specify the degree of fuzziness.
        n_dec (int, optional): designates the number of decimals of the labels returned.
        p (int, optional): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
        fig (figure object, optional): figure over which the graphs are plotted in case ax is not specified.
            If None and ax is also None, the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are plotted. If None, see param fig.
        nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        sample_colors (list of colors, optional): contains in order the colors of each
            sample of the fdatagrid.
        sample_labels (list of str, optional): contains in order the labels of each sample
            of the fdatagrid.
        cluster_labels (list of str, optional): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the plot are taken.
        xlabels (list of str, optional): Labels for the x-axes. Defaults to
            ["Cluster"] * fdatagrid.ndim_image.
        ylabels (list of str, optional): Labels for the y-axes. Defaults to
            ["Membership grade"] * fdatagrid.ndim_image.
        title (str, optional): Title for the figure where the clustering results are ploted.
            Defaults to "Membership grades of the samples to each cluster".

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

            labels (numpy.ndarray: (n_clusters, nsamples)): 2-dimensional matrix where each row
            contains the membership value that observation has to each cluster.

            centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.

    """
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    _plot_clustering_checks(fdatagrid, n_clusters, sample_colors,
                            sample_labels, None, cluster_labels, None, None)

    xlabels, ylabels, title = _labels_checks(fdatagrid, xlabels, ylabels,
                                             title, "Cluster")

    labels, _ = fuzzy_kmeans(fdatagrid, n_clusters=n_clusters, init=init,
                             seed=seed, max_iter=max_iter,
                             fuzzifier=fuzzifier, n_dec=n_dec, p=p)

    if sample_colors is None:
        cluster_colors = colormap(np.arange(n_clusters) / (n_clusters - 1))
        labels_by_cluster = np.argmax(labels, axis=-1)
        sample_colors = cluster_colors[labels_by_cluster]

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                         range(fdatagrid.nsamples)]

    if cluster_labels is None:
        cluster_labels = ['${}$'.format(i) for i in range(n_clusters)]

    for j in range(fdatagrid.ndim_image):
        ax[j].get_xaxis().set_major_locator(MaxNLocator(integer=True))
        for i in range(fdatagrid.nsamples):
            ax[j].plot(np.arange(n_clusters), labels[i, j, :],
                       label=sample_labels[i], color=sample_colors[i, j])
        ax[j].set_xticks(np.arange(n_clusters))
        ax[j].set_xticklabels(cluster_labels)
        ax[j].set_xlabel(xlabels[j])
        ax[j].set_ylabel(ylabels[j])
        datacursor(formatter='{label}'.format)

    fig.suptitle(title)

    return fig, ax, labels


def plot_fuzzy_kmeans_bars(fdatagrid, n_clusters=2, init=None, seed=None,
                           max_iter=100,  fuzzifier=2, n_dec=3, p=2,
                           fig=None, ax=None, nrows=None, ncols=None, sort=-1,
                           sample_labels=None,
                           cluster_colors=None, cluster_labels=None,
                           colormap=plt.cm.get_cmap('rainbow'),
                           xlabels=None, ylabels=None, title=None):
    """Implementation of the plotting of the results of the
    :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


    A kind of barplot is generated in this function with the
    membership values obtained from the algorithm. There is a bar for each sample
    whose height is 1 (the sum of the membership values of a sample add to 1), and
    the part proportional to each cluster is coloured with the corresponding color.
    See `Clustering Example <../auto_examples/plot_clustering.html>`_.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        n_clusters (int): Number of groups into which the samples are classified.
        init (ndarray, optional): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, and the centers are initialized randomly.
        seed (int, optional): Seed to initialize the random state to choose the initial centroids.
            Defaults to None.
        max_iter (int, optional): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        fuzzifier (int, optional): Scalar parameter used to specify the degree of fuzziness.
        n_dec (int, optional): designates the number of decimals of the labels returned.
        p (int, optional): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
        fig (figure object, optional): figure over which the graphs are plotted in case ax is not specified.
            If None and ax is also None, the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are plotted. If None, see param fig.
        nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        sort(int, optional): Number in the range [-1, n_clusters) designating the cluster whose
            labels are sorted in a decrementing order. Defaults to -1, in this case, no sorting is done.
        sample_labels (list of str, optional): contains in order the labels of each sample
            of the fdatagrid.
        cluster_colors (list of colors, optional): contains in order the colors of each
            cluster the samples of the fdatagrid are classified into.
        cluster_labels (list of str, optional): contains in order the names of each
            cluster the samples of the fdatagrid are classified into.
        colormap(colormap, optional): colormap from which the colors of the plot are taken.
        xlabels (list of str): Labels for the x-axes. Defaults to
            ["Sample"] * fdatagrid.ndim_image.
        ylabels (list of str): Labels for the y-axes. Defaults to
            ["Membership grade"] * fdatagrid.ndim_image.
        title (str): Title for the figure where the clustering results are ploted.
            Defaults to "Membership grades of the samples to each cluster".

    Returns:
        (tuple): tuple containing:

            fig (figure object): figure object in which the graphs are plotted in case ax is None.

            ax (axes object): axes in which the graphs are plotted.

            labels (numpy.ndarray: (n_clusters, nsamples)): 2-dimensional matrix where each row
            contains the membership value that observation has to each cluster.

            centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.

    """
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    if sort < -1 or sort >= n_clusters:
        raise ValueError(
            "The sorting number must belong to the interval [-1, n_clusters)")

    _plot_clustering_checks(fdatagrid, n_clusters, None, sample_labels,
                            cluster_colors, cluster_labels, None, None)

    xlabels, ylabels, title = _labels_checks(fdatagrid, xlabels, ylabels,
                                             title, "Sample")

    labels, _ = fuzzy_kmeans(fdatagrid, n_clusters=n_clusters, init=init,
                             seed=seed, max_iter=max_iter,
                             fuzzifier=fuzzifier, n_dec=n_dec, p=p)

    if sample_labels is None:
        sample_labels = np.arange(fdatagrid.nsamples)

    if cluster_colors is None:
        cluster_colors = colormap(np.arange(n_clusters) / (n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = ['$CLUSTER: {}$'.format(i) for i in range(n_clusters)]

    patches = []
    for i in range(n_clusters):
        patches.append(
            mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]))

    for j in range(fdatagrid.ndim_image):
        sample_labels_dim = np.copy(sample_labels)
        cluster_colors_dim = np.copy(cluster_colors)
        if sort != -1:
            sample_indices = np.argsort(-labels[:, j, sort])
            sample_labels_dim = np.copy(sample_labels[sample_indices])
            labels_dim = np.copy(labels[sample_indices, j])

            temp_labels = np.copy(labels_dim[:, 0])
            labels_dim[:, 0] = labels_dim[:, sort]
            labels_dim[:, sort] = temp_labels

            temp_color = np.copy(cluster_colors_dim[0])
            cluster_colors_dim[0] = cluster_colors_dim[sort]
            cluster_colors_dim[sort] = temp_color
        else:
            labels_dim = np.squeeze(labels[:, j])

        conc = np.zeros((fdatagrid.nsamples, 1))
        labels_dim = np.concatenate((conc, labels_dim), axis=-1)
        for i in range(n_clusters):
            ax[j].bar(np.arange(fdatagrid.nsamples), labels_dim[:, i + 1],
                      bottom=np.sum(labels_dim[:, :(i + 1)], axis=1),
                      color=cluster_colors_dim[i])
        ax[j].set_xticks(np.arange(fdatagrid.nsamples))
        ax[j].set_xticklabels(sample_labels_dim)
        ax[j].set_xlabel(xlabels[j])
        ax[j].set_ylabel(ylabels[j])
        ax[j].legend(handles=patches)

        fig.suptitle(title)

    return fig, ax, labels

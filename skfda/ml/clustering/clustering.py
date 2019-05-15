"""Clustering Module."""

import numpy as np
from ...representation.grid import FDataGrid
from ...misc.metrics import pairwise_distance, lp_distance
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from abc import abstractmethod
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
import warnings
from sklearn.utils import check_random_state

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


class BaseKMeansData(BaseEstimator, ClusterMixin, TransformerMixin):
    """Base class to implement clustering algorithms.

    Class from which both :class:`K-Means <fda.clustering.KMeans>` and
    :class:`Fuzzy K-Means <fda.clustering.FuzzyKMeans>` classes inherit."""

    def __init__(self, n_clusters, init, metric, n_init, max_iter, tol,
                 random_state):
        """Sets the arguments *max_iter* and *random_state* and *p*.

        Args:
            max_iter (int): Maximum number of iterations of the clustering
                algorithm.
            random_state (int): Seed to initialize the random state to choose
                the initial centroids.
            p (int): Identifies the p-norm used to calculate the distance
                between functions.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _generic_clustering_checks(self, fdatagrid):
        """Checks the arguments *fdatagrid*, *n_clusters* and *init*
        used in the :func:`fit <fda.clustering.BaseKMeansData.fit>` function.

        Args:
            fdatagrid (FDataGrid object): Object whose samples
                are classified into different groups.
            n_clusters (int): Number of groups into which the samples are
                classified.
            init (FDataGrid): Contains the initial centers of the different
                clusters the algorithm starts with or None. Its data_marix must
                be of the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).

        Returns:
            fdatagrid (FDataGrid object): Object whose samples
                are classified into different groups.
            n_clusters (int): Number of groups into which the samples are
                classified.
            init (FDataGrid): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                If None, default value, the centers are generated randomly.
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

    def _init_centroids(self, data_matrix, fdatagrid, random_state):
        """Random initialization of the centroids used in the
        :func:`fit function <fda.clustering.BaseKMeansData.fit>` if *init*
        is None.

        Args:
            data_matrix (ndarray): matrix with the data only of the
                dimension of the image of the fdatagrid the algorithm is
                classifying.
            fdatagrid (FDataGrid object): Object whose samples are
                classified into different groups.
            n_clusters (int): Number of groups into which the samples are
                classified.

        Returns:
            centers (ndarray): initial centers
        """

        comparison = True
        while comparison:
            indices = random_state.permutation(fdatagrid.nsamples)[
                      :self.n_clusters]
            centers = data_matrix[indices]
            unique_centers = np.unique(centers, axis=0)
            comparison = len(unique_centers) != self.n_clusters

        return centers

    @abstractmethod
    def fit(self, X, y=None, sample_weight=None):
        pass

    def _check_is_fitted_and_data(self, fdatagrid):

        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

        if not hasattr(self, "labels_") or not hasattr(self,
                                                       "cluster_centers_"):
            raise NotFittedError(msg % {'name': type(self).__name__})

        if fdatagrid.shape[1:3] != self.cluster_centers_.shape[1:3]:
            raise ValueError("The fdatagrid shape is not the one expected for "
                             "the calculated cluster_centers_.")

    def predict(self, X, sample_weight=None):
        self._check_is_fitted_and_data(X)
        return self.labels_

    def fit_predict(self, X, y=None, sample_weight=None, init=None):
        self.fit(X, init)
        return self.labels_

    def transform(self, X):
        self._check_is_fitted_and_data(X)
        return self._distances_to_centers

    def fit_transform(self, X, y=None, sample_weight=None, init=None):
        self.fit(X, init)
        return self._distances_to_centers

    def score(self, X, y=None, sample_weight=None):
        self._check_is_fitted_and_data(X)
        return -self.inertia_

    def _plot_clustering_checks(self, sample_colors, sample_labels,
                                cluster_colors, cluster_labels,
                                center_colors, center_labels):
        """Checks the arguments *sample_colors*, *sample_labels*, *cluster_colors*,
        *cluster_labels*, *center_colors*, *center_labels* passed to the plot
        functions, such as :func:`plot_clustering <fda.clustering.BaseKMeansData.plot>`,
        have the correct dimensions.

        Args:
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

        if sample_colors is not None and len(
                sample_colors) != self.fdatagrid.nsamples:
            raise ValueError(
                "sample_colors must contain a color for each sample.")

        if sample_labels is not None and len(
                sample_labels) != self.fdatagrid.nsamples:
            raise ValueError(
                "sample_labels must contain a label for each sample.")

        if cluster_colors is not None and len(
                cluster_colors) != self.n_clusters:
            raise ValueError(
                "cluster_colors must contain a color for each cluster.")

        if cluster_labels is not None and len(
                cluster_labels) != self.n_clusters:
            raise ValueError(
                "cluster_labels must contain a label for each cluster.")

        if center_colors is not None and len(center_colors) != self.n_clusters:
            raise ValueError(
                "center_colors must contain a color for each center.")

        if center_labels is not None and len(center_labels) != self.n_clusters:
            raise ValueError(
                "centers_labels must contain a label for each center.")

    @abstractmethod
    def plot(self, fig, ax, nrows, ncols, labels, sample_labels,
             cluster_colors, cluster_labels, center_colors, center_labels,
             colormap):
        """Plot of the FDataGrid samples by clusters.

        Once each sample is assigned a label, which are passed as argument to this
        function, the plotting is implemented here. Each group is assigned a color
        described in a legend.

        Args:
            fig (figure object): figure over which the graphs are plotted in
                case ax is not specified. If None and ax is also None, the figure
                is initialized.
            ax (list of axis objects): axis over where the graphs are plotted.
                If None, see param fig.
            nrows(int): designates the number of rows of the figure to plot the
                different dimensions of the image. Only specified if fig and
                ax are None.
            ncols(int): designates the number of columns of the figure to plot
                the different dimensions of the image. Only specified if fig
                and ax are None.
            labels (numpy.ndarray, int: (nsamples, ndim_image)): 2-dimensional
                matrix where each row contains the number of cluster cluster
                that observation belongs to.
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
            colormap(colormap): colormap from which the colors of the plot are taken.

        Returns:
            (tuple): tuple containing:

                fig (figure object): figure object in which the graphs are plotted in case ax is None.

                ax (axes object): axes in which the graphs are plotted.
        """
        fig, ax = self.fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

        self._plot_clustering_checks(None, sample_labels, cluster_colors,
                                     cluster_labels, center_colors,
                                     center_labels)

        if sample_labels is None:
            sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                             range(self.fdatagrid.nsamples)]

        if cluster_colors is None:
            cluster_colors = colormap(
                np.arange(self.n_clusters) / (self.n_clusters - 1))

        if cluster_labels is None:
            cluster_labels = ['$CLUSTER: {}$'.format(i) for i in
                              range(self.n_clusters)]

        if center_colors is None:
            center_colors = ["black"] * self.n_clusters

        if center_labels is None:
            center_labels = ['$CENTER: {}$'.format(i) for i in
                             range(self.n_clusters)]

        colors_by_cluster = cluster_colors[labels]

        patches = []
        for i in range(self.n_clusters):
            patches.append(
                mpatches.Patch(color=cluster_colors[i],
                               label=cluster_labels[i]))

        for j in range(self.fdatagrid.ndim_image):
            for i in range(self.fdatagrid.nsamples):
                ax[j].plot(self.fdatagrid.sample_points[0],
                           self.fdatagrid.data_matrix[i, :, j],
                           c=colors_by_cluster[i, j],
                           label=sample_labels[i])
            for i in range(self.n_clusters):
                ax[j].plot(self.fdatagrid.sample_points[0],
                           self.cluster_centers_.data_matrix[i, :, j],
                           c=center_colors[i], label=center_labels[i])
            ax[j].legend(handles=patches)
            datacursor(formatter='{label}'.format)

        self.fdatagrid.set_labels(fig, ax)

        return fig, ax


class KMeans(BaseKMeansData):
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
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        n_clusters (int, optional): Number of groups into which the samples are classified. Defaults to 2.
        init (FDataGrid, optional): Contains the initial centers of the different
            clusters the algorithm starts with. Its data_marix must be of
            the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
            Defaults to None, and the centers are initialized randomly.
        max_iter (int, optional): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        random_state (int, optional): Seed to initialize the random state to choose the initial centroids.
            Defaults to None.
        p (int, optional): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
        n_iter (numpy.array, (fdatagrid.ndim_image)): number of iterations the
            algorithm was run for each dimension.
        clustering_values (numpy.ndarray: (nsamples, ndim_image)): 2-dimensional matrix where each row
            contains the cluster that observation belongs to.
        centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the centroids for each cluster.

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
            max_iter (int): Maximum number of iterations of the clustering
                algorithm.
            random_state (int): Defaults to 0. Determines random number generation
                for centroid initialization. Use an int to make the randomness
                deterministic.
            p (int): Identifies the p-norm used to calculate the distance
                between functions.
        """
        super().__init__(n_clusters=n_clusters, init=init, metric=metric,
                         n_init=n_init, max_iter=max_iter, tol=tol,
                         random_state=random_state)

    def _kmeans_1Dimage(self, num_dim, fdatagrid, random_state):
        """ Implementation of the K-Means algorithm for each dimension on the
            image of the FDataGrid object.

        Args:
             num_dim (int): Scalar indicating the dimension on the image of the
                FdataGrid object the algorithm is being applied.
            fdatagrid (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
            n_clusters (int): Number of groups into which the samples are classified.
            init (FDataGrid): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, and the centers are initialized randomly.

        Returns:
            (tuple): tuple containing:

                clustering_values (numpy.ndarray: (nsamples,)): 1-dimensional
                array where each row contains the cluster that observation
                   belongs to.

                centers (numpy.ndarray: (n_clusters, ncol)): Contains the
                centroids for each cluster.

                repetitions(int): number of iterations the algorithm was run
                for this dimension og the image.

        """

        data_matrix = fdatagrid.data_matrix[..., num_dim]
        fdatagrid_1dim = fdatagrid.copy(data_matrix=data_matrix)
        repetitions = 0
        centers_old = np.zeros((self.n_clusters, fdatagrid.ncol))

        if self.init is None:
            centers = super()._init_centroids(data_matrix, fdatagrid,
                                              random_state)
        else:
            centers = np.copy(self.init.data_matrix[..., num_dim])

        while not np.allclose(centers, centers_old, rtol=self.tol,
                              atol=self.tol) and repetitions < self.max_iter:
            centers_old = np.copy(centers)
            centers_fd = FDataGrid(centers, fdatagrid.sample_points)
            distances_to_centers = self.metric(fdata1=fdatagrid_1dim,
                                               fdata2=centers_fd)
            clustering_values = np.argmin(distances_to_centers, axis=1)
            for i in range(self.n_clusters):
                indices = np.where(clustering_values == i)
                if indices[0].size != 0:
                    centers[i] = np.average(data_matrix[indices, :], axis=1)
            repetitions += 1

        return clustering_values, centers, distances_to_centers, repetitions

    def fit(self, X, y=None, sample_weight=None):
        """ Computes K-Means clustering calculating the *clustering_values*
        and *centers* arguments.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
            n_clusters (int): Number of groups into which the samples are classified.
            init (FDataGrid): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, and the centers are initialized randomly.
        """
        random_state = check_random_state(self.random_state)
        fdatagrid = super()._generic_clustering_checks(fdatagrid=X)

        clustering_values = np.empty((fdatagrid.nsamples,
                                      fdatagrid.ndim_image)).astype(int)
        centers = np.empty(
            (self.n_clusters, fdatagrid.ncol, fdatagrid.ndim_image))
        distances_to_centers = np.empty(
            (fdatagrid.nsamples, self.n_clusters, fdatagrid.ndim_image))
        n_iter = np.empty((fdatagrid.ndim_image))
        inertia = np.empty((fdatagrid.ndim_image))

        for i in range(fdatagrid.ndim_image):
            clustering_values_1D = np.empty((self.n_init,
                                             fdatagrid.nsamples)).astype(int)
            centers_1D = np.empty(
                (self.n_init, self.n_clusters, fdatagrid.ncol))
            distances_to_centers_1D = np.empty(
                (self.n_init, fdatagrid.nsamples, self.n_clusters))
            distances_to_their_center_1D = np.empty(
                (self.n_init, fdatagrid.nsamples))
            n_iter_1D = np.empty((self.n_init))

            for j in range(self.n_init):
                clustering_values_1D[j, :], centers_1D[j, :, :], \
                distances_to_centers_1D[j, :, :], n_iter_1D[j] = \
                    self._kmeans_1Dimage(num_dim=i, fdatagrid=fdatagrid,
                                         random_state=random_state)

                distances_to_their_center_1D[j, :] = distances_to_centers_1D[
                    j, np.arange(len(distances_to_centers_1D[j])),
                    clustering_values_1D[j, :]]

            inertia_1D = np.sum(distances_to_their_center_1D ** 2, axis=1)
            index_best_iter = np.argmin(inertia_1D)
            inertia[i] = inertia_1D[index_best_iter]
            clustering_values[:, i] = clustering_values_1D[index_best_iter]
            centers[:, :, i] = centers_1D[index_best_iter]
            distances_to_centers[:, :, i] = distances_to_centers_1D[
                index_best_iter]
            n_iter[i] = n_iter_1D[index_best_iter]

        self.fdatagrid = fdatagrid  # TODO: quitar
        self.labels_ = clustering_values
        self.cluster_centers_ = FDataGrid(data_matrix=centers,
                                          sample_points=fdatagrid.sample_points)
        self._distances_to_centers = distances_to_centers
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    def plot(self, fig=None, ax=None, nrows=None, ncols=None,
             sample_labels=None, cluster_colors=None,
             cluster_labels=None, center_colors=None, center_labels=None,
             colormap=plt.cm.get_cmap('rainbow')):
        """Plot of the FDataGrid samples by the clusters calculated with the
        K-Means algorithm.


        Args:
            fig (figure object): figure over which the graphs are plotted in
                case ax is not specified. If None and ax is also None, the figure
                is initialized.
            ax (list of axis objects): axis over where the graphs are plotted.
                If None, see param fig.
            nrows(int): designates the number of rows of the figure to plot the
                different dimensions of the image. Only specified if fig and
                ax are None.
            ncols(int): designates the number of columns of the figure to plot
                the different dimensions of the image. Only specified if fig
                and ax are None.
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
            colormap(colormap): colormap from which the colors of the plot are
                taken. Defaults to `rainbow`.

        Returns:
            (tuple): tuple containing:

                fig (figure object): figure object in which the graphs are plotted in case ax is None.

                ax (axes object): axes in which the graphs are plotted.
        """

        return super().plot(fig=fig, ax=ax, nrows=nrows, ncols=ncols,
                            labels=self.labels_,
                            sample_labels=sample_labels,
                            cluster_colors=cluster_colors,
                            cluster_labels=cluster_labels,
                            center_colors=center_colors,
                            center_labels=center_labels, colormap=colormap)


class FuzzyKMeans(BaseKMeansData):
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
        fdatagrid (FDataGrid object): Object whose samples are clusered,
            classified into different groups.
        n_clusters (int, optional): Number of groups into which the samples are
            classified. Defaults to 2.
        init (FDataGrid, optional): Contains the initial centers of the different
            clusters the algorithm starts with. Its data_marix must be of
            the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
            Defaults to None, ans the centers are initialized randomly.
        max_iter (int, optional): Maximum number of iterations of the clustering
            algorithm. Defaults to 100.
        random_state (int, optional): Seed to initialize the random state to choose
            the initial centroids. Defaults to 0.
        p (int, optional): Identifies the p-norm used to calculate the distance
            between functions. Defaults to 2.
        n_iter (numpy.array, (fdatagrid.ndim_image)): number of iterations the
            algorithm was run for each dimension.
        membership_values (numpy.ndarray: ((nsamples, ndim_image, n_clusters)):
            contains the degree of membership each observation has to each
            cluster.
        centers (numpy.ndarray: (ndim_image, n_clusters, ncol)): Contains the
            centroids for each cluster.
        fuzzifier (int, optional): Scalar parameter used to specify the
            degree of fuzziness in the fuzzy algorithm. Defaults to 2.
        n_dec (int, optional): designates the number of decimals of the labels
            returned in the fuzzy algorithm. Defaults to 3.

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
            max_iter (int): Maximum number of iterations of the clustering
                algorithm.
            random_state (int, optional): Seed to initialize the random state
                to choose the initial centroids. Defaults to 0.
            p (int): Identifies the p-norm used to calculate the distance
                between functions.
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

    def _fuzzy_kmeans_1Dimage(self, num_dim, fdatagrid, random_state):
        """ Implementation of the Fuzzy C-Means algorithm for each dimension
        on the image of the FDataGrid object.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
            num_dim (int): Scalar indicating the dimension on the image of
                the FdataGrid object the algorithm is being applied.
            n_clusters (int): Number of groups into which the samples are classified.
            fuzzifier (int): Scalar parameter used to specify the degree of fuzziness.
            init (FDataGrid): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, ans the centers are initialized randomly.
            random_state (int): Seed to initialize the random state to choose
                the initial centroids.
            max_iter (int): Maximum number of iterations of the clustering algorithm.
            p (int): Identifies the p-norm used to calculate the distance between functions.
            n_dec (int): designates the number of decimals of the labels returned.

        Returns:
            (tuple): tuple containing:

                membership values (numpy.ndarray: (n_clusters, nsamples)):
                2-dimensional matrix where each row contains the membership value
                that observation has to each cluster.

                centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids
                for each cluster.

                repetitions(int): number of iterations the algorithm was run
                for this dimension og the image.

        """

        data_matrix = np.copy(fdatagrid.data_matrix[..., num_dim])
        repetitions = 0
        centers_old = np.zeros((self.n_clusters, fdatagrid.ncol))
        U = np.empty((self.n_clusters, fdatagrid.nsamples))

        if self.init is None:
            centers = super()._init_centroids(data_matrix,
                                              fdatagrid, random_state)
        else:
            centers = np.copy(self.init.data_matrix[..., num_dim])

        distances_to_centers = np.empty((fdatagrid.nsamples, self.n_clusters))

        while not np.array_equal(centers, centers_old) and \
                repetitions < self.max_iter:
            centers_old = np.copy(centers)
            for i in range(fdatagrid.nsamples):
                comparison = (data_matrix[i] == centers).all(-1)
                if comparison.sum() >= 1:
                    U[np.where(comparison == True), i] = 1
                    U[np.where(comparison == False), i] = 0
                else:
                    centers_fd = FDataGrid(centers, fdatagrid.sample_points)
                    fd_single_sample = FDataGrid(data_matrix[i],
                                                 fdatagrid.sample_points)
                    distances_to_centers_single_sample = self.metric(
                        fdata1=fd_single_sample,
                        fdata2=centers_fd) ** (2 / (self.fuzzifier - 1))
                    for j in range(self.n_clusters):
                        U[j, i] = 1 / np.sum(
                            distances_to_centers_single_sample[0, j] /
                            distances_to_centers_single_sample[0])
                distances_to_centers[i] = distances_to_centers_single_sample[0]

            U = np.power(U, self.fuzzifier)

            for i in range(self.n_clusters):
                centers[i] = np.sum((U[i] * data_matrix.T).T, axis=0) / \
                             np.sum(U[i])
            repetitions += 1

        return np.round(np.power(U, 1 / self.fuzzifier), self.n_dec), centers, \
               distances_to_centers, repetitions

    def fit(self, X, y=None, sample_weight=None):
        """ Computes Fuzzy K-Means clustering calculating the *labels_*
        and *centers* arguments.

        Args:
            fdatagrid (FDataGrid object): Object whose samples are clusered,
                classified into different groups.
            n_clusters (int): Number of groups into which the samples are classified.
            init (FDataGrid): Contains the initial centers of the different
                clusters the algorithm starts with. Its data_marix must be of
                the shape (n_clusters, fdatagrid.ncol, fdatagrid.ndim_image).
                Defaults to None, and the centers are initialized randomly.
        """

        fdatagrid = super()._generic_clustering_checks(fdatagrid=X)
        random_state = check_random_state(self.random_state)

        if self.fuzzifier < 2:
            raise ValueError("The fuzzifier parameter must be greater than 1.")

        if self.n_dec < 1:
            raise ValueError(
                "The number of decimals should be greater than 0 in order to "
                "obatain a rational result.")

        membership_values = np.empty(
            (fdatagrid.nsamples, fdatagrid.ndim_image, self.n_clusters))
        centers = np.empty(
            (self.n_clusters, fdatagrid.ncol, fdatagrid.ndim_image))
        distances_to_centers = np.empty(
            (fdatagrid.nsamples, self.n_clusters, fdatagrid.ndim_image))
        n_iter = np.empty((fdatagrid.ndim_image))
        inertia = np.empty((fdatagrid.ndim_image))

        for i in range(fdatagrid.ndim_image):
            membership_values_1D = np.empty((self.n_init, fdatagrid.nsamples,
                                             self.n_clusters))
            centers_1D = np.empty(
                (self.n_init, self.n_clusters, fdatagrid.ncol))
            distances_to_centers_1D = np.empty(
                (self.n_init, fdatagrid.nsamples, self.n_clusters))
            distances_to_their_center_1D = np.empty(
                (self.n_init, fdatagrid.nsamples))
            n_iter_1D = np.empty((self.n_init))

            for j in range(self.n_init):
                U_1D, centers_1D[j, :, :], \
                distances_to_centers_1D[j, :, :], n_iter_1D[j] = \
                    self._fuzzy_kmeans_1Dimage(num_dim=i, fdatagrid=fdatagrid,
                                               random_state=random_state)
                membership_values_1D[j, :, :] = U_1D.T
                distances_to_their_center_1D[j, :] = distances_to_centers_1D[
                    j, np.arange(len(distances_to_centers_1D[j])),
                    np.argmax(membership_values_1D[j, :, :], axis=-1)]

            inertia_1D = np.sum(distances_to_their_center_1D ** 2, axis=1)
            index_best_iter = np.argmin(inertia_1D)
            inertia[i] = inertia_1D[index_best_iter]
            membership_values[:, i, :] = membership_values_1D[index_best_iter]
            centers[:, :, i] = centers_1D[index_best_iter]
            distances_to_centers[:, :, i] = distances_to_centers_1D[
                index_best_iter]
            n_iter[i] = n_iter_1D[index_best_iter]

        self.fdatagrid = fdatagrid  # TODO: quitar
        self.labels_ = membership_values
        self.cluster_centers_ = FDataGrid(data_matrix=centers,
                                          sample_points=fdatagrid.sample_points)
        self._distances_to_centers = distances_to_centers
        self.inertia_ = inertia
        self.n_iter_ = n_iter

        return self

    def plot(self, fig=None, ax=None, nrows=None, ncols=None,
             sample_labels=None, cluster_colors=None,
             cluster_labels=None, center_colors=None, center_labels=None,
             colormap=plt.cm.get_cmap('rainbow')):
        """Plot of the FDataGrid samples by the clusters calculated with the
        Fuzzy K-Means algorithm..

        Args:
            fig (figure object): figure over which the graphs are plotted in
                case ax is not specified. If None and ax is also None, the figure
                is initialized.
            ax (list of axis objects): axis over where the graphs are plotted.
                If None, see param fig.
            nrows(int): designates the number of rows of the figure to plot the
                different dimensions of the image. Only specified if fig and
                ax are None.
            ncols(int): designates the number of columns of the figure to plot
                the different dimensions of the image. Only specified if fig
                and ax are None.
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
            colormap(colormap): colormap from which the colors of the plot are
                taken. Defaults to `rainbow`.

        Returns:
            (tuple): tuple containing:

                fig (figure object): figure object in which the graphs are plotted in case ax is None.

                ax (axes object): axes in which the graphs are plotted.
        """

        return super().plot(fig=fig, ax=ax, nrows=nrows, ncols=ncols,
                            labels=np.argmax(self.labels_, axis=-1),
                            sample_labels=sample_labels,
                            cluster_colors=cluster_colors,
                            cluster_labels=cluster_labels,
                            center_colors=center_colors,
                            center_labels=center_labels, colormap=colormap)

    def _labels_checks(self, xlabels, ylabels, title, xlabel_str):
        """Checks the arguments *xlabels*, *ylabels*, *title* passed to both
        :func:`plot_fuzzy_kmeans_lines <fda.clustering.plot_fuzzy_kmeans_lines>` and
        :func:`plot_fuzzy_kmeans_bars <fda.clustering.plot_fuzzy_kmeans_bars>`
        functions. In case they are not set yet, hey are given a value.

        Args:
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

        if xlabels is not None and len(xlabels) != self.fdatagrid.ndim_image:
            raise ValueError(
                "xlabels must contain a label for each dimension on the domain.")

        if ylabels is not None and len(ylabels) != self.fdatagrid.ndim_image:
            raise ValueError(
                "xlabels must contain a label for each dimension on the domain.")

        if xlabels is None:
            xlabels = [xlabel_str] * self.fdatagrid.ndim_image

        if ylabels is None:
            ylabels = ["Membership grade"] * self.fdatagrid.ndim_image

        if title is None:
            title = "Membership grades of the samples to each cluster"

        return xlabels, ylabels, title

    def plot_lines(self, fig=None, ax=None, nrows=None, ncols=None,
                   sample_colors=None, sample_labels=None, cluster_labels=None,
                   colormap=plt.cm.get_cmap('rainbow'), xlabels=None,
                   ylabels=None, title=None):
        """Implementation of the plotting of the results of the
        :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


        A kind of Parallel Coordinates plot is generated in this function with the
        membership values obtained from the algorithm. A line is plotted for each
        sample with the values for each cluster. See `Clustering Example
        <../auto_examples/plot_clustering.html>`_.

        Args:
            fig (figure object, optional): figure over which the graphs are
                plotted in case ax is not specified. If None and ax is also None,
                the figure is initialized.
            ax (list of axis objects, optional): axis over where the graphs are
                plotted. If None, see param fig.
            nrows(int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            ncols(int, optional): designates the number of columns of the figure
                to plot the different dimensions of the image. Only specified if
                fig and ax are None.
            sample_colors (list of colors, optional): contains in order the colors of each
                sample of the fdatagrid.
            sample_labels (list of str, optional): contains in order the labels
                of each sample  of the fdatagrid.
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

        """
        fig, ax = self.fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

        super()._plot_clustering_checks(sample_colors, sample_labels, None,
                                        cluster_labels, None, None)

        xlabels, ylabels, title = self._labels_checks(xlabels, ylabels,
                                                      title, "Cluster")

        if sample_colors is None:
            cluster_colors = colormap(np.arange(self.n_clusters) /
                                      (self.n_clusters - 1))
            labels_by_cluster = np.argmax(self.labels_, axis=-1)
            sample_colors = cluster_colors[labels_by_cluster]

        if sample_labels is None:
            sample_labels = ['$SAMPLE: {}$'.format(i) for i in
                             range(self.fdatagrid.nsamples)]

        if cluster_labels is None:
            cluster_labels = ['${}$'.format(i) for i in range(self.n_clusters)]

        for j in range(self.fdatagrid.ndim_image):
            ax[j].get_xaxis().set_major_locator(MaxNLocator(integer=True))
            for i in range(self.fdatagrid.nsamples):
                ax[j].plot(np.arange(self.n_clusters),
                           self.labels_[i, j, :],
                           label=sample_labels[i], color=sample_colors[i, j])
            ax[j].set_xticks(np.arange(self.n_clusters))
            ax[j].set_xticklabels(cluster_labels)
            ax[j].set_xlabel(xlabels[j])
            ax[j].set_ylabel(ylabels[j])
            datacursor(formatter='{label}'.format)

        fig.suptitle(title)
        return fig, ax

    def plot_bars(self, fig=None, ax=None, nrows=None, ncols=None, sort=-1,
                  sample_labels=None, cluster_colors=None, cluster_labels=None,
                  colormap=plt.cm.get_cmap('rainbow'), xlabels=None,
                  ylabels=None, title=None):
        """Implementation of the plotting of the results of the
        :func:`Fuzzy K-Means <fda.clustering.fuzzy_kmeans>` method.


        A kind of barplot is generated in this function with the
        membership values obtained from the algorithm. There is a bar for each sample
        whose height is 1 (the sum of the membership values of a sample add to 1), and
        the part proportional to each cluster is coloured with the corresponding color.
        See `Clustering Example <../auto_examples/plot_clustering.html>`_.

        Args:
            fig (figure object, optional): figure over which the graphs are
                plotted in case ax is not specified. If None and ax is also None,
                the figure is initialized.
            ax (list of axis objects, optional): axis over where the graphs are
                plotted. If None, see param fig.
            nrows(int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            ncols(int, optional): designates the number of columns of the figure
                to plot the different dimensions of the image. Only specified if
                fig and ax are None.
            sort(int, optional): Number in the range [-1, n_clusters) designating
                the cluster whose labels are sorted in a decrementing order.
                Defaults to -1, in this case, no sorting is done.
            sample_colors (list of colors, optional): contains in order the colors of each
                sample of the fdatagrid.
            sample_labels (list of str, optional): contains in order the labels
                of each sample  of the fdatagrid.
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

        """
        fig, ax = self.fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

        if sort < -1 or sort >= self.n_clusters:
            raise ValueError(
                "The sorting number must belong to the interval [-1, n_clusters)")

        super()._plot_clustering_checks(None, sample_labels, cluster_colors,
                                        cluster_labels, None, None)

        xlabels, ylabels, title = self._labels_checks(xlabels, ylabels,
                                                      title, "Sample")

        if sample_labels is None:
            sample_labels = np.arange(self.fdatagrid.nsamples)

        if cluster_colors is None:
            cluster_colors = colormap(
                np.arange(self.n_clusters) / (self.n_clusters - 1))

        if cluster_labels is None:
            cluster_labels = ['$CLUSTER: {}$'.format(i) for i in
                              range(self.n_clusters)]

        patches = []
        for i in range(self.n_clusters):
            patches.append(
                mpatches.Patch(color=cluster_colors[i],
                               label=cluster_labels[i]))

        for j in range(self.fdatagrid.ndim_image):
            sample_labels_dim = np.copy(sample_labels)
            cluster_colors_dim = np.copy(cluster_colors)
            if sort != -1:
                sample_indices = np.argsort(
                    -self.labels_[:, j, sort])
                sample_labels_dim = np.copy(sample_labels[sample_indices])
                labels_dim = np.copy(self.labels_[sample_indices, j])

                temp_labels = np.copy(labels_dim[:, 0])
                labels_dim[:, 0] = labels_dim[:, sort]
                labels_dim[:, sort] = temp_labels

                temp_color = np.copy(cluster_colors_dim[0])
                cluster_colors_dim[0] = cluster_colors_dim[sort]
                cluster_colors_dim[sort] = temp_color
            else:
                labels_dim = np.squeeze(self.labels_[:, j])

            conc = np.zeros((self.fdatagrid.nsamples, 1))
            labels_dim = np.concatenate((conc, labels_dim), axis=-1)
            for i in range(self.n_clusters):
                ax[j].bar(np.arange(self.fdatagrid.nsamples),
                          labels_dim[:, i + 1],
                          bottom=np.sum(labels_dim[:, :(i + 1)], axis=1),
                          color=cluster_colors_dim[i])
            ax[j].set_xticks(np.arange(self.fdatagrid.nsamples))
            ax[j].set_xticklabels(sample_labels_dim)
            ax[j].set_xlabel(xlabels[j])
            ax[j].set_ylabel(ylabels[j])
            ax[j].legend(handles=patches)

        fig.suptitle(title)
        return fig, ax

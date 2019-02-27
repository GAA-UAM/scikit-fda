"""Clustering Module."""

import math
import numpy as np
from .grid import FDataGrid
from .math import metric, norm_lp
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def _generic_clustering_checks(fdatagrid, n_clusters, init, max_iter, fuzzifier=2, n_dec=3):
    r"""Checks the arguments passed to both :func:`clustering <fda.clustering.clustering>` and
    :func:`fuzzy clustering <fda.clustering.fuzzy_clustering>` functions.

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
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if fdatagrid.nsamples < 2:
        raise ValueError("The number of observations must be greater than 1.")

    if n_clusters < 2:
        raise ValueError("The number of clusters must be greater than 1.")

    if init is not None and init.shape != (fdatagrid.ndim_image, n_clusters, fdatagrid.ncol):
        raise ValueError("The init ndarray should be of shape (ndim_image, n_clusters, n_features) "
                         "and gives the initial centers.")
    else:
        init = np.array([None] * fdatagrid.ndim_image)

    if max_iter < 1:
        raise ValueError("The number of iterations must be greater than 0.")

    if fuzzifier < 2:
        raise ValueError("The fuzzifier parameter must be greater than 1.")

    if n_dec < 1:
        raise ValueError("The number of decimals should be greater than 0 in order to obatain a rational result.")

    return init


def _clustering_1Dimage(fdatagrid, num_dim, n_clusters, centers, max_iter, p):
    r""" Implementation of the K-Means algorithm for each dimension on the image of the FDataGrid object.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        num_dim (int): Scalar indicating the dimension on the image of the FdataGrid object the algorithm is
            being applied..
        n_clusters (int): Number of groups into which the samples are classified.
        centers (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        max_iter (int): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        p (int): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.

    Returns:
        (tuple): tuple containing:

            clustering_values (numpy.ndarray: (nsamples,)): 1-dimensional array where each row
            contains the cluster that observation belongs to.

            centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids for each cluster.

    """

    data_matrix = np.copy(fdatagrid.data_matrix[:, :, num_dim])
    fdatagrid_1dim = FDataGrid(data_matrix=data_matrix, sample_points=fdatagrid.sample_points[0])
    repetitions = 0
    centers_old = np.empty((n_clusters, fdatagrid.ncol))

    # Method for initialization: choose k observations (rows) at random from data for the initial centroids.
    if centers is None:
        centers = np.empty((n_clusters, fdatagrid.ncol))
        for i in range(n_clusters):
            centers[i] = data_matrix[math.floor(i * fdatagrid.nsamples / n_clusters)].flatten()

    while not np.array_equal(centers, centers_old) and repetitions < max_iter:
        centers_old = np.copy(centers)
        centers_fd = FDataGrid(centers, fdatagrid.sample_points)
        distances_to_centers = metric(fdatagrid=fdatagrid_1dim, fdatagrid2=centers_fd, p=p)
        clustering_values = np.argmin(distances_to_centers, axis=1)
        for i in range(n_clusters):
            indices = np.where(clustering_values == i)
            centers[i] = np.average(data_matrix[indices, :], axis=1)
        repetitions += 1

    return clustering_values, centers


def clustering(fdatagrid, n_clusters=2, init=None, max_iter=100, p=2):
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
        clustering_values[:, i], centers[i, :, :] = _clustering_1Dimage(fdatagrid, num_dim=i,
                                                                        n_clusters=n_clusters,
                                                                        centers=init[i],
                                                                        max_iter=max_iter, p=p)

    return clustering_values, centers


def _fuzzy_clustering_1Dimage(fdatagrid, num_dim, n_clusters, fuzzifier, centers, max_iter, p, n_dec):
    r""" Implementation of the Fuzzy C-Means algorithm for each dimension on the image of the FDataGrid object.

    Args:
        fdatagrid (FDataGrid object): Object whose samples are clusered, classified into different groups.
        num_dim (int): Scalar indicating the dimension on the image of the FdataGrid object the algorithm is
            being applied..
        n_clusters (int): Number of groups into which the samples are classified.
        fuzzifier (int): Scalar parameter used to specify the degree of fuzziness.
        centers (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        max_iter (int): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        p (int): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
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

    # Method for initialization: choose k observations (rows) at random from data for the initial centroids.
    if centers is None:
        centers = np.empty((n_clusters, fdatagrid.ncol))
        quotient, reminder = divmod(fdatagrid.nsamples, n_clusters)
        cluster = np.concatenate((np.tile(np.arange(n_clusters), quotient), np.arange(reminder)))
        for i in range(n_clusters):
            centers[i] = np.average(data_matrix[cluster == i], axis=0)

    while not np.array_equal(centers, centers_old) and repetitions < max_iter:
        centers_old = np.copy(centers)
        for i in range(fdatagrid.nsamples):
            comparison = (data_matrix[i] == centers).all(-1)
            if comparison.sum() == 1:
                U[np.where(comparison == True), i] = 1
                U[np.where(comparison == False), i] = 0
            else:
                diff = data_matrix[i] - centers
                diff_fd = FDataGrid(diff, fdatagrid.sample_points)
                distances_to_centers = np.power(norm_lp(diff_fd, p), 2 / (fuzzifier - 1))
                for j in range(n_clusters):
                    U[j, i] = 1 / np.sum(distances_to_centers[j] / distances_to_centers)
        U = np.power(U, fuzzifier)
        for i in range(n_clusters):
            centers[i] = np.sum((U[i] * data_matrix.T).T, axis=0) / np.sum(U[i])
        repetitions += 1

    return np.round(np.power(U, 1 / fuzzifier), n_dec), centers


def fuzzy_clustering(fdatagrid, n_clusters=2, init=None, max_iter=100, fuzzifier=2, n_dec=3, p=2):
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
        centers (ndarray): Contains the initial centers of the different clusters the algorithm starts with.
            Defaults to None, ans the centers are initialized randomly.
        max_iter (int): Maximum number of iterations of the clustering algorithm. Defaults to 100.
        p (int): Identifies the p-norm used to calculate the distance between functions. Defaults to 2.
        n_dec (int): designates the number of decimals of the labels returned.

    Returns:
        (tuple): tuple containing:

            membership values (numpy.ndarray: (n_clusters, nsamples)): 2-dimensional matrix where each row
            contains the membership value that observation has to each cluster.

            centers (numpy.ndarray: (n_clusters, ncol)): Contains the centroids for each cluster.

    """

    init = _generic_clustering_checks(fdatagrid, n_clusters, init, max_iter, fuzzifier, n_dec)
    membership_values = np.empty((fdatagrid.nsamples, fdatagrid.ndim_image, n_clusters))
    centers = np.empty((fdatagrid.ndim_image, n_clusters, fdatagrid.ncol))
    for i in range(fdatagrid.ndim_image):
        U, centers[i, :, :] = _fuzzy_clustering_1Dimage(fdatagrid, num_dim=i,
                                                        n_clusters=n_clusters,
                                                        fuzzifier=fuzzifier,
                                                        centers=init[i],
                                                        max_iter=max_iter, p=p, n_dec=n_dec)
        membership_values[:, i, :] = U.T

    return membership_values, centers


def _labels_checks(fdatagrid, xlabels, ylabels, title, xlabel_str):
    if xlabels is not None and len(xlabels) != fdatagrid.ndim_image:
        raise ValueError("xlabels must contain a label for each dimension on the domain.")

    if ylabels is not None and len(ylabels) != fdatagrid.ndim_image:
        raise ValueError("xlabels must contain a label for each dimension on the domain.")

    if xlabels is None:
        xlabels = [xlabel_str] * fdatagrid.ndim_image

    if ylabels is None:
        ylabels = ["Membership grade"] * fdatagrid.ndim_image

    if title is None:
        title = "Membership grades of the samples to each cluster"

    return xlabels, ylabels, title


def _plot_clustering_checks(fdatagrid, n_clusters, sample_colors, sample_labels, cluster_colors, cluster_labels,
                            center_colors, center_labels):
    if sample_colors is not None and len(sample_colors) != fdatagrid.nsamples:
        raise ValueError("sample_colors must contain a color for each sample.")

    if sample_labels is not None and len(sample_labels) != fdatagrid.nsamples:
        raise ValueError("sample_labels must contain a label for each sample.")

    if cluster_colors is not None and len(cluster_colors) != n_clusters:
        raise ValueError("cluster_colors must contain a color for each cluster.")

    if cluster_labels is not None and len(cluster_labels) != n_clusters:
        raise ValueError("cluster_labels must contain a label for each cluster.")

    if center_colors is not None and len(center_colors) != n_clusters:
        raise ValueError("center_colors must contain a color for each center.")

    if center_labels is not None and len(center_labels) != n_clusters:
        raise ValueError("centers_labels must contain a label for each center.")


def plot_clustering(fdatagrid, n_clusters=2, method=clustering, init=None, max_iter=100, fuzzifier=2, n_dec=3, p=2,
                    fig=None, ax=None, nrows=None, ncols=None, sample_labels=None, cluster_colors=None,
                    cluster_labels=None, center_colors=None, center_labels=None,
                    colormap=plt.cm.get_cmap('rainbow')):
    if method == clustering:
        labels, centers = clustering(fdatagrid, n_clusters=n_clusters, init=init, max_iter=max_iter)
        labels = labels.astype(int)
    elif method == fuzzy_clustering:
        labels, centers = fuzzy_clustering(fdatagrid, n_clusters=n_clusters, init=init, max_iter=max_iter,
                                           fuzzifier=fuzzifier, n_dec=n_dec, p=p)
        labels = np.argmax(labels, axis=-1)
    else:
        raise ValueError("method must be clustering or fuzzy_clustering")

    return plot_clustering_implementation(fdatagrid, labels, centers, n_clusters=n_clusters, fig=fig, ax=ax,
                                          nrows=nrows,
                                          ncols=ncols, sample_labels=sample_labels, cluster_colors=cluster_colors,
                                          cluster_labels=cluster_labels, center_colors=center_colors,
                                          center_labels=center_labels, colormap=colormap)


def plot_clustering_implementation(fdatagrid, labels, centers, n_clusters, fig, ax, nrows, ncols, sample_labels,
                                   cluster_colors, cluster_labels, center_colors, center_labels, colormap):
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    _plot_clustering_checks(fdatagrid, n_clusters, None, sample_labels, cluster_colors, cluster_labels,
                            center_colors, center_labels)

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in range(fdatagrid.nsamples)]

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
        patches.append(mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]))

    for j in range(fdatagrid.ndim_image):
        for i in range(fdatagrid.nsamples):
            ax[j].plot(fdatagrid.sample_points[0], fdatagrid.data_matrix[i, :, j], c=colors_by_cluster[i, j],
                       label=sample_labels[i])
        for i in range(n_clusters):
            ax[j].plot(fdatagrid.sample_points[0], centers[j, i, :], c=center_colors[i], label=center_labels[i])
        ax[j].legend(handles=patches)
        datacursor(formatter='{label}'.format)

    fdatagrid.set_labels(fig, ax)

    return fig, ax, labels, centers


def plot_fuzzy_clustering_values(fdatagrid, n_clusters=2, init=None, max_iter=100, fuzzifier=2, n_dec=3, p=2,
                                 fig=None, ax=None, nrows=None, ncols=None, sample_colors=None, sample_labels=None,
                                 cluster_labels=None, colormap=plt.cm.get_cmap('rainbow'), xlabels=None,
                                 ylabels=None, title=None):
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    _plot_clustering_checks(fdatagrid, n_clusters, sample_colors, sample_labels, None, cluster_labels, None, None)

    xlabels, ylabels, title = _labels_checks(fdatagrid, xlabels, ylabels, title, "Cluster")

    labels, _ = fuzzy_clustering(fdatagrid, n_clusters=n_clusters, init=init, max_iter=max_iter,
                                 fuzzifier=fuzzifier, n_dec=n_dec, p=p)

    if sample_colors is None:
        cluster_colors = colormap(np.arange(n_clusters) / (n_clusters - 1))
        labels_by_cluster = np.argmax(labels, axis=-1)
        sample_colors = cluster_colors[labels_by_cluster]

    if sample_labels is None:
        sample_labels = ['$SAMPLE: {}$'.format(i) for i in range(fdatagrid.nsamples)]

    if cluster_labels is None:
        cluster_labels = ['${}$'.format(i) for i in range(n_clusters)]

    for j in range(fdatagrid.ndim_image):
        ax[j].get_xaxis().set_major_locator(MaxNLocator(integer=True))
        for i in range(fdatagrid.nsamples):
            ax[j].plot(np.arange(n_clusters), labels[i, j, :], label=sample_labels[i], color=sample_colors[i, j])
        ax[j].set_xticks(np.arange(n_clusters))
        ax[j].set_xticklabels(cluster_labels)
        ax[j].set_xlabel(xlabels[j])
        ax[j].set_ylabel(ylabels[j])
        datacursor(formatter='{label}'.format)

    fig.suptitle(title)

    return fig, ax, labels


def plot_fuzzy_clustering_bars(fdatagrid, n_clusters=2, init=None, max_iter=100, fuzzifier=2, n_dec=3, p=2,
                               fig=None, ax=None, nrows=None, ncols=None, sort=-1, sample_labels=None,
                               cluster_colors=None, cluster_labels=None, colormap=plt.cm.get_cmap('rainbow'),
                               xlabels=None, ylabels=None, title=None):
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    if sort < -1 or sort >= n_clusters:
        raise ValueError("The sorting number must belong to the interval [-1, n_clusters)")

    _plot_clustering_checks(fdatagrid, n_clusters, None, sample_labels, cluster_colors, cluster_labels, None, None)

    xlabels, ylabels, title = _labels_checks(fdatagrid, xlabels, ylabels, title, "Sample")

    labels, _ = fuzzy_clustering(fdatagrid, n_clusters=n_clusters, init=init, max_iter=max_iter,
                                 fuzzifier=fuzzifier, n_dec=n_dec, p=p)

    if sample_labels is None:
        sample_labels = np.arange(fdatagrid.nsamples)

    if cluster_colors is None:
        cluster_colors = colormap(np.arange(n_clusters) / (n_clusters - 1))

    if cluster_labels is None:
        cluster_labels = ['$CLUSTER: {}$'.format(i) for i in range(n_clusters)]

    patches = []
    for i in range(n_clusters):
        patches.append(mpatches.Patch(color=cluster_colors[i], label=cluster_labels[i]))

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

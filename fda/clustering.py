"""Clustering Module."""

import scipy
import math
import numpy as np
from .math import norm_lp

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"

def _clustering_1Dimage(fdatagrid, num_dim, n_clusters, centers, max_iter, metric, *args, **kwargs):

    data_matrix = np.copy(fdatagrid.data_matrix[:, :, num_dim])
    repetitions = 0
    centers_old = np.empty((n_clusters, fdatagrid.ncol))

    # Method for initialization: choose k observations (rows) at random from data for the initial centroids.
    if centers is  None:
        centers = np.empty((n_clusters, fdatagrid.ncol))
        for i in range(n_clusters):
            centers[i] = data_matrix[math.floor(i * fdatagrid.nsamples / n_clusters)].flatten()

    while not np.array_equal(centers, centers_old) and repetitions < max_iter:
        centers_old = np.copy(centers)
        distances_to_centers = scipy.spatial.distance.cdist(data_matrix, centers, metric, *args, **kwargs)
        clustering_values = np.argmin(distances_to_centers, axis=1)
        for i in range(n_clusters):
            indices = np.where(clustering_values == i)
            centers[i] = np.average(data_matrix[indices, :], axis=1)
        repetitions += 1

    return clustering_values, centers


def clustering(fdatagrid, n_clusters=2, init=None, max_iter=100, metric='euclidean', *args, **kwargs):
    if fdatagrid.ndim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if fdatagrid.nsamples < 2:
        raise ValueError("The number of observations must be greater than 1.")

    if n_clusters < 2:
        raise ValueError("The number of clusters must be greater than 1.")

    if max_iter < 1:
        raise ValueError("The number of iterations must be greater than 0.")

    if init is not None and init.shape != (fdatagrid.ndim_image, n_clusters, fdatagrid.ncol):
        raise ValueError("The init ndarray should be of shape (ndim_image, n_clusters, n_features) "
                         "and gives the initial centers.")
    else:
        init = np.array([None] * fdatagrid.ndim_image)

    possible_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                        "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski", "mahalanobis",
                        "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener",
                        "sokalsneath", "sqeuclidean", "wminkowski", "yule"]

    if metric not in possible_metrics:
        raise ValueError("Metric must be one of the specified in the documentation.")

    clustering_values = np.empty((fdatagrid.nsamples, fdatagrid.ndim_image))
    centers = np.empty((fdatagrid.ndim_image, n_clusters, fdatagrid.ncol))
    for i in range(fdatagrid.ndim_image):
        clustering_values[:, i], centers[i, :, :] = _clustering_1Dimage(fdatagrid, num_dim=i,
                                                                       n_clusters=n_clusters, centers=init[i],
                                                                       max_iter=max_iter, metric=metric, *args,
                                                                       **kwargs)

    return clustering_values, centers

def _fuzzy_clustering_1Dimage(fdatagrid, num_dim, n_clusters, fuzzifier, centers, max_iter, norm_matrix):

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

    if norm_matrix is None:
        norm_matrix = np.eye(fdatagrid.ncol)

    while not np.array_equal(centers, centers_old) and repetitions < max_iter:
        centers_old = np.copy(centers)
        for i in range(fdatagrid.nsamples):
            comparison = (data_matrix[i] == centers).all(-1)
            if comparison.sum() == 1:
                U[np.where(comparison == True), i] = 1
                U[np.where(comparison == False), i] = 0
            else:
                diff = data_matrix[i] - centers
                distances_to_centers = np.power(np.diag(np.dot(np.dot(diff, norm_matrix), diff.T)),
                                                2 / (fuzzifier - 1))
                for j in range(n_clusters):
                    U[j, i] = 1 / np.sum(distances_to_centers[j] / distances_to_centers)
        U = np.power(U, fuzzifier)
        for i in range(n_clusters):
            centers[i] = np.sum((U[i] * data_matrix.T).T, axis=0) / np.sum(U[i])
        repetitions += 1

    return U, centers


def fuzzy_clustering(fdatagrid, n_clusters=2, init=None, fuzzifier=2, max_iter=100, norm_matrix=None):
    if fdatagrid.ndim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if fdatagrid.nsamples < 2:
        raise ValueError("The number of observations must be greater than 1.")

    if n_clusters < 2:
        raise ValueError("The number of clusters must be greater than 1.")

    if fuzzifier < 1:
        raise ValueError("The fuzzifier parameter must be greater than 0.")

    if max_iter < 1:
        raise ValueError("The number of iterations must be greater than 0.")

    if init is not None and init.shape != (fdatagrid.ndim_image, n_clusters, fdatagrid.ncol):
        raise ValueError("The init ndarray should be of shape (ndim_image, n_clusters, n_features) "
                         "and gives the initial centers.")
    else:
        init = np.array([None] * fdatagrid.ndim_image)

    if norm_matrix is not None and norm_matrix.shape != (fdatagrid.ndim_image, fdatagrid.ncol, fdatagrid.ncol):
        raise ValueError("The norm_matrix should be of shape (ndim_image, n_features, n_features).")
    elif norm_matrix is not None:
        for i in range(fdatagrid.ndim_image):
            if not np.all(np.linalg.eigvals(norm_matrix[i]) > 0):
                raise ValueError("The norm_matrix should be positive definite.")
            if not np.array_equal(norm_matrix[i], norm_matrix[i].T):
                raise ValueError("The norm_matrix should be symmetric.")
    else:
        norm_matrix = np.array([None] * fdatagrid.ndim_image)

    membership_values = np.empty((fdatagrid.nsamples, fdatagrid.ndim_image, n_clusters))
    centers = np.empty((fdatagrid.ndim_image, n_clusters, fdatagrid.ncol))
    for i in range(fdatagrid.ndim_image):
        U, centers[i, :, :] = _fuzzy_clustering_1Dimage(fdatagrid, num_dim=i, n_clusters=n_clusters,
                                                       fuzzifier=fuzzifier, centers=init[i],
                                                       max_iter=max_iter, norm_matrix=norm_matrix[i])
        membership_values[:, i, :] = U.T

    return membership_values, centers




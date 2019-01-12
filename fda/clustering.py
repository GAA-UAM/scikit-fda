"""Graphics Module to visualize FDataGrid."""

import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import variation
from scipy import stats
import scipy
import math


from .grid import FDataGrid

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"

from fda.depth_measures import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import variation
from scipy import stats
import scipy
import math




def clustering(fdatagrid, ax=None, num_clusters=2):
    if fdatagrid.ndim_image > 1 and fdatagrid.ndim_domain:
        raise NotImplementedError("Only support 1 dimension on the image and on the domain.")

    if num_clusters < 2 or num_clusters > 10:
        raise ValueError("The number of clusters must be between 2 and 10, both included.")

    repetitions = 0
    distances_to_centers = np.empty((fdatagrid.nsamples, num_clusters))
    centers = np.empty((num_clusters, len(fdatagrid.sample_points[0])))
    centers_aux = np.empty((num_clusters, len(fdatagrid.sample_points[0])))

    for i in range(num_clusters):
        centers[i] = fdatagrid.data_matrix[math.floor(i * fdatagrid.nsamples / num_clusters)].flatten()

    while not np.array_equal(centers, centers_aux) and repetitions < 100:
        centers_aux = centers
        for i in range(fdatagrid.nsamples):
            for j in range(num_clusters):
                #pairwise distance
                distances_to_centers[i, j] = scipy.spatial.distance.euclidean(fdatagrid.data_matrix[i], centers[j])
        clustering_values = np.argmin(distances_to_centers, axis=1)
        for i in range(num_clusters):
            indices = np.where(clustering_values == i)
            centers[i] = np.average(fdatagrid.data_matrix[indices, :, :].flatten()
                                    .reshape((len(indices[0]), len(fdatagrid.sample_points[0]))), axis=0)
        repetitions += 1

    colors_samples = np.empty(fdatagrid.nsamples).astype(str)
    for i in range(num_clusters):
        colors_samples[np.where(clustering_values == i)] = "C{}".format(i)

    if ax is None:
        ax = matplotlib.pyplot.gca()

    for i in range(fdatagrid.nsamples):
        ax.plot(fdatagrid.sample_points[0], fdatagrid.data_matrix[i, :, 0].T, color=colors_samples[i])

    return clustering_values
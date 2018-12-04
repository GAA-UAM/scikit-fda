"""Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones)."""

import numpy as np
from scipy.stats import rankdata
from numpy import linalg as LA
from .grid import FDataGrid

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def rank_samples(fdgrid):
    """Ranks the samples in the FDataGrid.

    Args:
    fdgrid (FDataGrid): Object whose samples are ranked.

    Returns:
    numpy.darray: Array containing the rankes of the samples.
    """
    ranks = np.zeros(fdgrid.shape)

    if fdgrid.ndim_domain == 1:
        for i in range(fdgrid.ndim_image):
            for j in range(fdgrid.ncol):
                ranks[:, j, i] = rankdata(fdgrid.data_matrix[:, j, i], method='max')
    else:
        for i in range(fdgrid.ndim_image):
            for j in range(len(fdgrid.sample_points[1])):
                for k in range(len(fdgrid.sample_points[0])):
                    ranks[:, k, j, i] = rankdata(fdgrid.data_matrix[:, k, j, i], method='max')

    return ranks


def band_depth(fdgrid):
    """Implementation of Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of the bands determined by two sample curves containing the whole
    graph of the first one.

    Args:
    fdgrid (FDataGrid): Object over whose samples the band depth is going to be calculated.

    Returns:
    numpy.darray: Array containing the band depth of the samples.

    """
    if fdgrid.ndim_domain > 2:
        raise NotImplementedError("Only support 1 or 2 dimensions on the domain.")

    if fdgrid.ndim_domain == 1:
        axis = 1
    else:
        axis = (1, 2)

    ranks = rank_samples(fdgrid)
    nsamples_above = fdgrid.nsamples - np.amax(ranks, axis=axis)
    nsamples_below = np.amin(ranks, axis = axis) - 1
    n = fdgrid.nsamples
    nchoose2 = n * (n - 1) / 2
    depth = (nsamples_below * nsamples_above + fdgrid.nsamples - 1) / nchoose2

    return depth


def modified_band_depth(fdgrid, pointwise = False):
    """Implementation of Modified Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of time its graph is contained in the bands determined by two
    sample curves.

    Args:
    fdgrid (FDataGrid): Object over whose samples the modified band depth is going to be calculated.
    pointwise (boolean): Indicates if the pointwise depth is also returned.
    
    Returns:
    numpy.darray: Array containing the modified band depth of the samples.

    """
    if fdgrid.ndim_domain > 2:
        raise NotImplementedError("Only support 1 or 2 dimensions on the domain.")
    
    if fdgrid.ndim_domain == 1:
        axis = 1
        npoints_sample = fdgrid.ncol
    else:
        axis = (1, 2)
        npoints_sample = len(fdgrid.sample_points[0]) * len(fdgrid.sample_points[1])

    n = fdgrid.nsamples
    nchoose2 = n * (n - 1) / 2
    
    ranks = rank_samples(fdgrid)
    nsamples_above = fdgrid.nsamples - ranks
    nsamples_below = ranks - 1
    match = nsamples_above * nsamples_below
    proportion = match.sum(axis = axis) / npoints_sample 
    depth = (proportion + fdgrid.nsamples - 1) / nchoose2
    
    if pointwise:
        depth_pointwise = (match + fdgrid.nsamples - 1) / nchoose2
        return [depth_pointwise, depth]
    else:
        return depth

def cumulative_distribution(column):
    """Calculates the cumulative distribution function of the values passed to the function and evaluates it at each point.

    Args:
    column (numpy.darray): Array containing the values over which the distribution function is calculated.

    Returns:
    numpy.darray: Array containing the evaluation at each point of the distribution function.

    """
    cumulative_distribution = np.zeros(column.shape)
    # Obtaining the frequency of each of the occurences.
    unique, counts = np.unique(column, return_counts=True)
    freq = dict(zip(unique, counts))
    # Calculating the distribution and the FM depth of each of the entries of the column.
    count_cumulative = 0
    for ocurrence in freq:
        count_cumulative += freq[ocurrence]
        indices = np.where(column == ocurrence)
        cumulative_distribution[indices] = count_cumulative / len(column)

    return cumulative_distribution


def FM_depth(fdgrid, pointwise = False):
    """Implementation of Fraiman and Muniz (FM) Depth for functional data.

    Each column is considered as the samples of an aleatory variable. The univariate depth of each of the samples of each column is
	calculated as follows:

    .. math::
      $$D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert$$

    Where F stands for the marginal univariate distribution function of each column.

    The depth of a sample is the result of adding the previously computed depth for each of its points.

    Args:
      fdgrid (FDataGrid): Object over whose samples the FM depth is going to be calculated.
      pointwise (boolean): Indicates if the pointwise depth is also returned.

    Returns:
      numpy.darray: Array containing the FM depth of the samples.

    """
    if fdgrid.ndim_domain > 2:
        raise NotImplementedError("Only support 1 or 2 dimensions on the domain.")

    if fdgrid.ndim_domain == 1:
        axis = 1
        npoints_sample = fdgrid.ncol
    else:
        axis = (1, 2)
        npoints_sample = len(fdgrid.sample_points[0]) * len(fdgrid.sample_points[1])

    univariate_depth = np.zeros(fdgrid.shape)

    if fdgrid.ndim_domain == 1:
        for i in range(fdgrid.ndim_image):
            for j in range(fdgrid.ncol):
                column = fdgrid.data_matrix[:, j, i]
                univariate_depth[:, j, i] = 1 - np.abs(0.5 - cumulative_distribution(column))
    else:
        for i in range(fdgrid.ndim_image):
            for j in range(len(fdgrid.sample_points[1])):
                for k in range(len(fdgrid.sample_points[0])):
                    column = fdgrid.data_matrix[:, k, j, i]
                    univariate_depth[:, k, j, i] = 1 - np.abs(0.5 - cumulative_distribution(column))
    
                    
    if pointwise:
        return univariate_depth, np.sum(univariate_depth, axis = axis) / npoints_sample
    else:
        return np.sum(univariate_depth, axis = axis) / npoints_sample


def directional_outlyingness(fdgrid, depth_method, dim_weights = None, pointwise_weights = None):

    if fdgrid.ndim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if dim_weights is not None and (len(dim_weights) != fdgrid.ndim_image or dim_weights.sum() != 1):
        raise ValueError(
            "There must be a weight in dim_weights for each dimension of the image and altogether must sum 1.")

    if pointwise_weights is not None and len(pointwise_weights) != fdgrid.ncol:
        raise ValueError("There must be a weight in pointwise_weights for each recorded time point.")

    depth = depth_method(fdgrid, pointwise=True)
    depth_pointwise = depth[0]
    depth = depth[1]

    if dim_weights is None:
        dim_weights = np.ones(fdgrid.ndim_image) / fdgrid.ndim_image

    if pointwise_weights is None:
        interval = fdgrid.sample_range[0, 1] - fdgrid.sample_range[0, 0]
        pointwise_weights = np.ones(fdgrid.ncol) / interval

    # Calculation of the depth of each multivariate sample with the corresponding weight.
    weighted_depth = depth * dim_weights
    sample_depth = weighted_depth.sum(axis=-1)

    # Obtaining the median sample Z, to caculate v(t) = {X(t) − Z(t)}/∥ X(t) − Z(t)∥
    median_index = np.argmax(sample_depth)
    median = fdgrid.data_matrix[median_index]
    v = fdgrid.data_matrix - median
    v_norm = LA.norm(v, axis=-1, keepdims=True)
    v_norm[np.where(v_norm == 0)] = 1
    v_unitary = v / v_norm

    # Calculation of the depth of each multivariate point sample with the corresponding weight.
    weighted_depth_pointwise = depth_pointwise * dim_weights
    sample_depth_pointwise = weighted_depth_pointwise.sum(axis=-1, keepdims=True)

    # Calcuation directinal outlyingness
    dir_outlyingness = (1 / sample_depth_pointwise - 1) * v_unitary

    # Calcuation mean directinal outlyingness
    pointwise_weights_1 = np.tile(pointwise_weights, (fdgrid.ndim_image, 1)).T
    weighted_dir_outlyingness = dir_outlyingness * pointwise_weights_1
    mean_dir_outl = weighted_dir_outlyingness.sum(axis=1)

    # Calcuation variation directinal outlyingness
    mean_dir_outl_pointwise = np.repeat(mean_dir_outl, fdgrid.ncol, axis=0).reshape(fdgrid.shape)
    norm = np.square(LA.norm(dir_outlyingness - mean_dir_outl_pointwise, axis=-1))
    weighted_norm = norm * pointwise_weights
    variation_dir_outl = weighted_norm.sum(axis=1)

    return dir_outlyingness, mean_dir_outl, variation_dir_outl
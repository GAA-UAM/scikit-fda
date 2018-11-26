"""Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones)."""

import numpy as np
from scipy.stats import rankdata
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
    nsamples_below = np.amin(ranks, axis=axis) - 1
    n = fdgrid.nsamples
    nchoose2 = n(n-1)/2
    depth = (nsamples_below * nsamples_above + fdgrid.nsamples - 1) / nchoose2

    return depth


def modified_band_depth(fdgrid):
    """Implementation of Modified Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of time its graph is contained in the bands determined by two
    sample curves.

    Args:
    fdgrid (FDataGrid): Object over whose samples the modified band depth is going to be calculated.

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

    ranks = rank_samples(fdgrid)
    nsamples_above = fdgrid.nsamples - ranks
    nsamples_below = ranks - 1
    match = nsamples_above * nsamples_below
    proportion = match.sum(axis=axis) / npoints_sample
    n = fdgrid.nsamples
    nchoose2 = n(n - 1) / 2
    depth = (proportion + fdgrid.nsamples - 1) / nchoose2

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


def FM_depth(fdgrid):
    """Implementation of Fraiman and Muniz (FM) Depth for functional data.

    Each column is considered as the samples of an aleatory variable. The univariate depth of each of the samples of each column is
	calculated as follows:

    .. math::
      $$D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert$$

    Where F stands for the marginal univariate distribution function of each column.

    The depth of a sample is the result of adding the previously computed depth for each of its points.

    Args:
      fdgrid (FDataGrid): Object over whose samples the FM depth is going to be calculated.

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

    return np.sum(univariate_depth, axis=axis) / npoints_sample
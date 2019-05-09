"""Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones)."""

import numpy as np
from scipy.stats import rankdata
from .. import FDataGrid
import itertools
from functools import reduce

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def _rank_samples(fdatagrid):
    """Ranks the he samples in the FDataGrid at each point of discretisation.

    Args:
        fdatagrid (FDataGrid): Object whose samples are ranked.

    Returns:
        numpy.darray: Array containing the ranks of the sample points.

    Examples:
        Univariate setting:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> _rank_samples(fd)
        array([[[ 4.],
                [ 4.],
                [ 4.],
                [ 4.],
                [ 4.],
                [ 4.]],
        <BLANKLINE>
               [[ 3.],
                [ 3.],
                [ 3.],
                [ 3.],
                [ 3.],
                [ 3.]],
        <BLANKLINE>
               [[ 1.],
                [ 1.],
                [ 2.],
                [ 2.],
                [ 2.],
                [ 2.]],
        <BLANKLINE>
               [[ 2.],
                [ 2.],
                [ 2.],
                [ 1.],
                [ 1.],
                [ 1.]]])

        Multivariate Setting:

        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> _rank_samples(fd)
        array([[[[ 1.,  1.],
                 [ 1.,  1.]],
        <BLANKLINE>
                [[ 1.,  2.],
                 [ 2.,  2.]],
        <BLANKLINE>
                [[ 1.,  2.],
                 [ 2.,  2.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 2.,  2.],
                 [ 2.,  2.]],
        <BLANKLINE>
                [[ 2.,  1.],
                 [ 1.,  1.]],
        <BLANKLINE>
                [[ 2.,  1.],
                 [ 1.,  1.]]]])
    """
    ranks = np.zeros(fdatagrid.shape)
    ncols_dim_image = np.asarray([range(fdatagrid.shape[i]) for i in range(len(fdatagrid.shape) - 1, 0, -1)])
    tuples = list(itertools.product(*ncols_dim_image))
    for t in tuples:
        ranks.T[t] = rankdata(fdatagrid.data_matrix.T[t], method='max')
    return ranks


def band_depth(fdatagrid, pointwise=False):
    """Implementation of Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of the bands determined by two sample
    curves containing the whole graph of the first one. In the case the fdatagrid domain dimension is 2, instead
    of curves, surfaces determine the bands. In larger dimensions, the hyperplanes determine the bands.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the band depth is going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is also returned. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the band depth of the samples.

    Returns:
        depth_pointwise (numpy.darray, optional): Array containing the band depth of
        the samples at each point of discretisation. Only returned if pointwise equals to True.

    Examples:
        Univariate setting:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> band_depth(fd)
        array([[ 0.5       ],
               [ 0.83333333],
               [ 0.5       ],
               [ 0.5       ]])

        Multivariate Setting:

        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]],
        ...                [[[4, 6], [4, 10]], [[45, 48], [38, 56]], [[8, 36], [10, 28]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> band_depth(fd)
        array([[ 0.66666667,  0.66666667],
               [ 0.66666667,  0.66666667],
               [ 1.        ,  1.        ]])

    """
    n = fdatagrid.nsamples
    nchoose2 = n * (n - 1) / 2

    ranks = _rank_samples(fdatagrid)
    axis = tuple(range(1, fdatagrid.ndim_domain + 1))
    nsamples_above = fdatagrid.nsamples - np.amax(ranks, axis=axis)
    nsamples_below = np.amin(ranks, axis=axis) - 1
    depth = (nsamples_below * nsamples_above + fdatagrid.nsamples - 1) / nchoose2

    if pointwise:
        _, depth_pointwise = modified_band_depth(fdatagrid, pointwise)
        return depth, depth_pointwise
    else:
        return depth


def modified_band_depth(fdatagrid, pointwise=False):
    """Implementation of Modified Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of time its graph is contained
    in the bands determined by two sample curves. In the case the fdatagrid domain dimension is 2, instead
    of curves, surfaces determine the bands. In larger dimensions, the hyperplanes determine the bands.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the modified band depth is going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is also returned. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the modified band depth of the samples.

    Returns:
        depth_pointwise (numpy.darray, optional): Array containing the modified band depth of
        the samples at each point of discretisation. Only returned if pointwise equals to True.

    Examples:
        Univariate setting specifying pointwise:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> modified_band_depth(fd, pointwise = True)
        (array([[ 0.5       ],
               [ 0.83333333],
               [ 0.72222222],
               [ 0.66666667]]), array([[[ 0.5       ],
                [ 0.5       ],
                [ 0.5       ],
                [ 0.5       ],
                [ 0.5       ],
                [ 0.5       ]],
        <BLANKLINE>
               [[ 0.83333333],
                [ 0.83333333],
                [ 0.83333333],
                [ 0.83333333],
                [ 0.83333333],
                [ 0.83333333]],
        <BLANKLINE>
               [[ 0.5       ],
                [ 0.5       ],
                [ 0.83333333],
                [ 0.83333333],
                [ 0.83333333],
                [ 0.83333333]],
        <BLANKLINE>
               [[ 0.83333333],
                [ 0.83333333],
                [ 0.83333333],
                [ 0.5       ],
                [ 0.5       ],
                [ 0.5       ]]]))

        Multivariate Setting without specifying pointwise:

        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]],
        ...                [[[4, 6], [4, 10]], [[45, 48], [38, 56]], [[34, 78], [10, 28]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> modified_band_depth(fd)
        array([[ 0.66666667,  0.72222222],
               [ 0.72222222,  0.66666667],
               [ 0.94444444,  0.94444444]])

    """
    n = fdatagrid.nsamples
    nchoose2 = n * (n - 1) / 2

    ranks = _rank_samples(fdatagrid)
    nsamples_above = fdatagrid.nsamples - ranks
    nsamples_below = ranks - 1
    match = nsamples_above * nsamples_below
    axis = tuple(range(1, fdatagrid.ndim_domain + 1))
    npoints_sample = reduce(lambda x, y: x * len(y), fdatagrid.sample_points, 1)
    proportion = match.sum(axis=axis) / npoints_sample
    depth = (proportion + fdatagrid.nsamples - 1) / nchoose2

    if pointwise:
        depth_pointwise = (match + fdatagrid.nsamples - 1) / nchoose2
        return depth, depth_pointwise
    else:
        return depth


def _cumulative_distribution(column):
    """Calculates the cumulative distribution function of the values passed to the function and evaluates it at each point.

    Args:
        column (numpy.darray): Array containing the values over which the distribution function is calculated.

    Returns:
        numpy.darray: Array containing the evaluation at each point of the distribution function.

    Examples:
        >>> _cumulative_distribution(np.array([1, 4, 5, 1, 2, 2, 4, 1, 1, 3]))
        array([ 0.4,  0.9,  1. ,  0.4,  0.6,  0.6,  0.9,  0.4,  0.4,  0.7])

    """
    if len(column.shape) != 1:
        raise ValueError("Only supported 1 dimensional arrays.")
    _, indexes, counts = np.unique(column, return_inverse=True, return_counts=True)
    count_cumulative = np.cumsum(counts) / len(column)
    return count_cumulative[indexes].reshape(column.shape)


def fraiman_muniz_depth(fdatagrid, pointwise=False):
    r"""Implementation of Fraiman and Muniz (FM) Depth for functional data.

    Each column is considered as the samples of an aleatory variable. The univariate depth of each of the samples of each column is
    calculated as follows:

    .. math::
        D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert

    Where :math:`F` stands for the marginal univariate distribution function of each column.

    The depth of a sample is the result of adding the previously computed depth for each of its points.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the FM depth is going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is also returned. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the FM depth of the samples.

    Returns:
        depth_pointwise (numpy.darray, optional): Array containing the FM depth of
        the samples at each point of discretisation. Only returned if pointwise equals to True.

    Examples:
        Univariate setting specifying pointwise:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fraiman_muniz_depth(fd, pointwise = True)
        (array([[ 0.5       ],
               [ 0.75      ],
               [ 0.91666667],
               [ 0.875     ]]), array([[[ 0.5 ],
                [ 0.5 ],
                [ 0.5 ],
                [ 0.5 ],
                [ 0.5 ],
                [ 0.5 ]],
        <BLANKLINE>
               [[ 0.75],
                [ 0.75],
                [ 0.75],
                [ 0.75],
                [ 0.75],
                [ 0.75]],
        <BLANKLINE>
               [[ 0.75],
                [ 0.75],
                [ 1.  ],
                [ 1.  ],
                [ 1.  ],
                [ 1.  ]],
        <BLANKLINE>
               [[ 1.  ],
                [ 1.  ],
                [ 1.  ],
                [ 0.75],
                [ 0.75],
                [ 0.75]]]))

        Multivariate Setting without specifying pointwise:

        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]],
        ...                [[[4, 6], [4, 10]], [[45, 48], [38, 56]], [[34, 78], [10, 28]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fraiman_muniz_depth(fd)
        array([[ 0.72222222,  0.66666667],
               [ 0.66666667,  0.72222222],
               [ 0.77777778,  0.77777778]])

    """
    univariate_depth = np.zeros(fdatagrid.shape)

    ncols_dim_image = np.asarray([range(fdatagrid.shape[i]) for i in range(len(fdatagrid.shape) - 1, 0, -1)])
    tuples = list(itertools.product(*ncols_dim_image))
    for t in tuples:
        column = fdatagrid.data_matrix.T[t]
        univariate_depth.T[t] = 1 - np.abs(0.5 - _cumulative_distribution(column))

    axis = tuple(range(1, fdatagrid.ndim_domain + 1))
    npoints_sample = reduce(lambda x, y: x * len(y), fdatagrid.sample_points, 1)

    if pointwise:
        return np.sum(univariate_depth, axis=axis) / npoints_sample, univariate_depth
    else:
        return np.sum(univariate_depth, axis=axis) / npoints_sample

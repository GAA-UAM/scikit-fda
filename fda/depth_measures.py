"""Depth Measures Module.

This module includes different methods to order functional data,
from the center (larger values) outwards(smaller ones)."""

import numpy as np
from scipy.stats import rankdata
from numpy import linalg as LA
from .grid import FDataGrid

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def rank_samples(fdatagrid):
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
        >>> rank_samples(fd)
        array([[[4.],
                [4.],
                [4.],
                [4.],
                [4.],
                [4.]],
        <BLANKLINE>
               [[3.],
                [3.],
                [3.],
                [3.],
                [3.],
                [3.]],
        <BLANKLINE>
               [[1.],
                [1.],
                [2.],
                [2.],
                [2.],
                [2.]],
        <BLANKLINE>
               [[2.],
                [2.],
                [2.],
                [1.],
                [1.],
                [1.]]])

        Multivariate Setting:
        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> rank_samples(fd)
        array([[[[1., 1.],
                 [1., 1.]],
        <BLANKLINE>
                [[1., 2.],
                 [2., 2.]],
        <BLANKLINE>
                [[1., 2.],
                 [2., 2.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[2., 2.],
                 [2., 2.]],
        <BLANKLINE>
                [[2., 1.],
                 [1., 1.]],
        <BLANKLINE>
                [[2., 1.],
                 [1., 1.]]]])
    """
    ranks = np.zeros(fdatagrid.shape)

    if fdatagrid.ndim_domain == 1:
        for i in range(fdatagrid.ndim_image):
            for j in range(fdatagrid.ncol):
                ranks[:, j, i] = rankdata(fdatagrid.data_matrix[:, j, i], method='max')
    else:
        for i in range(fdatagrid.ndim_image):
            for j in range(len(fdatagrid.sample_points[1])):
                for k in range(len(fdatagrid.sample_points[0])):
                    ranks[:, k, j, i] = rankdata(fdatagrid.data_matrix[:, k, j, i], method='max')

    return ranks


def band_depth(fdatagrid):
    """Implementation of Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of the bands determined by two sample
    curves containing the whole graph of the first one.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the band depth is going to be calculated.

    Returns:
        numpy.darray: Array containing the band depth of the samples.

    Examples:
        Univariate setting:
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> band_depth(fd)
        array([[0.5       ],
               [0.83333333],
               [0.5       ],
               [0.5       ]])

        Multivariate Setting:
        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]],
        ...                [[[4, 6], [4, 10]], [[45, 48], [38, 56]], [[8, 36], [10, 28]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> band_depth(fd)
        array([[0.66666667, 0.66666667],
               [0.66666667, 0.66666667],
               [1.        , 1.        ]])

    """
    if fdatagrid.ndim_domain > 2:
        raise NotImplementedError("Only support 1 or 2 dimensions on the domain.")

    if fdatagrid.ndim_domain == 1:
        axis = 1
    else:
        axis = (1, 2)

    n = fdatagrid.nsamples
    nchoose2 = n * (n - 1) / 2

    ranks = rank_samples(fdatagrid)
    nsamples_above = fdatagrid.nsamples - np.amax(ranks, axis=axis)
    nsamples_below = np.amin(ranks, axis = axis) - 1
    depth = (nsamples_below * nsamples_above + fdatagrid.nsamples - 1) / nchoose2

    return depth


def modified_band_depth(fdatagrid, pointwise = False):
    """Implementation of Modified Band Depth for functional data.

    The band depth of each sample is obtained by computing the fraction of time its graph is contained
    in the bands determined by two sample curves.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the modified band depth is going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is also returned. Defaults to False.
    
    Returns:
        depth (numpy.darray): Array containing the modified band depth of the samples.
        depth_pointwise (numpy.darray, optional): Array containing the modified band depth of
            the samples at each point of discretisation. Only returned if pointwise equals to True.

    Examples:
        Univariate setting specifying pointwise:
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> modified_band_depth(fd, pointwise = True)
        (array([[0.5       ],
               [0.83333333],
               [0.72222222],
               [0.66666667]]), array([[[0.5       ],
                [0.5       ],
                [0.5       ],
                [0.5       ],
                [0.5       ],
                [0.5       ]],
        <BLANKLINE>
               [[0.83333333],
                [0.83333333],
                [0.83333333],
                [0.83333333],
                [0.83333333],
                [0.83333333]],
        <BLANKLINE>
               [[0.5       ],
                [0.5       ],
                [0.83333333],
                [0.83333333],
                [0.83333333],
                [0.83333333]],
        <BLANKLINE>
               [[0.83333333],
                [0.83333333],
                [0.83333333],
                [0.5       ],
                [0.5       ],
                [0.5       ]]]))

        Multivariate Setting without specifying pointwise:
        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]],
        ...                [[[4, 6], [4, 10]], [[45, 48], [38, 56]], [[34, 78], [10, 28]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> modified_band_depth(fd)
        array([[0.66666667, 0.72222222],
               [0.72222222, 0.66666667],
               [0.94444444, 0.94444444]])

    """
    if fdatagrid.ndim_domain > 2:
        raise NotImplementedError("Only support 1 or 2 dimensions on the domain.")
    
    if fdatagrid.ndim_domain == 1:
        axis = 1
        npoints_sample = fdatagrid.ncol
    else:
        axis = (1, 2)
        npoints_sample = len(fdatagrid.sample_points[0]) * len(fdatagrid.sample_points[1])

    n = fdatagrid.nsamples
    nchoose2 = n * (n - 1) / 2
    
    ranks = rank_samples(fdatagrid)
    nsamples_above = fdatagrid.nsamples - ranks
    nsamples_below = ranks - 1
    match = nsamples_above * nsamples_below
    proportion = match.sum(axis = axis) / npoints_sample 
    depth = (proportion + fdatagrid.nsamples - 1) / nchoose2
    
    if pointwise:
        depth_pointwise = (match + fdatagrid.nsamples - 1) / nchoose2
        return depth, depth_pointwise
    else:
        return depth

def cumulative_distribution(column):
    """Calculates the cumulative distribution function of the values passed to the function and evaluates it at each point.

    Args:
        column (numpy.darray): Array containing the values over which the distribution function is calculated.

    Returns:
        numpy.darray: Array containing the evaluation at each point of the distribution function.

    Examples:
        >>> cumulative_distribution(np.array([1, 4, 5, 1, 2, 2, 4, 1, 1, 3]))
        array([0.4, 0.9, 1. , 0.4, 0.6, 0.6, 0.9, 0.4, 0.4, 0.7])

    """
    if len(column.shape)!= 1:
        raise ValueError("Only supported 1 dimensional arrays.")
    _, indexes, counts = np.unique(column, return_inverse=True, return_counts=True)
    count_cumulative = np.cumsum(counts) / len(column)
    return count_cumulative[indexes].reshape(column.shape)


def Fraiman_Muniz_depth(fdatagrid, pointwise = False):
    r"""Implementation of Fraiman and Muniz (FM) Depth for functional data.

    Each column is considered as the samples of an aleatory variable. The univariate depth of each of the samples of each column is
	calculated as follows:

    .. math::
        $$D(x) = 1 - \left\lvert \frac{1}{2}- F(x)\right\rvert$$

    Where F stands for the marginal univariate distribution function of each column.

    The depth of a sample is the result of adding the previously computed depth for each of its points.

    Args:
        fdatagrid (FDataGrid): Object over whose samples the FM depth is going to be calculated.
        pointwise (boolean, optional): Indicates if the pointwise depth is also returned. Defaults to False.

    Returns:
        depth (numpy.darray): Array containing the FM depth of the samples.
        depth_pointwise (numpy.darray, optional): Array containing the modified band depth of
            the samples at each point of discretisation. Only returned if pointwise equals to True.

    Examples:
        Univariate setting specifying pointwise:
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> Fraiman_Muniz_depth(fd, pointwise = True)
        (array([[0.5       ],
               [0.75      ],
               [0.91666667],
               [0.875     ]]), array([[[0.5 ],
                [0.5 ],
                [0.5 ],
                [0.5 ],
                [0.5 ],
                [0.5 ]],
        <BLANKLINE>
               [[0.75],
                [0.75],
                [0.75],
                [0.75],
                [0.75],
                [0.75]],
        <BLANKLINE>
               [[0.75],
                [0.75],
                [1.  ],
                [1.  ],
                [1.  ],
                [1.  ]],
        <BLANKLINE>
               [[1.  ],
                [1.  ],
                [1.  ],
                [0.75],
                [0.75],
                [0.75]]]))

        Multivariate Setting without specifying pointwise:
        >>> data_matrix = [[[[1, 3], [2, 6]], [[23, 54], [43, 76]], [[2, 45], [12, 65]]],
        ...                [[[21, 34], [8, 16]], [[67, 43], [32, 21]], [[10, 24], [3, 12]]],
        ...                [[[4, 6], [4, 10]], [[45, 48], [38, 56]], [[34, 78], [10, 28]]]]
        >>> sample_points = [[2, 4, 6], [3, 6]]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> Fraiman_Muniz_depth(fd)
        array([[0.72222222, 0.66666667],
               [0.66666667, 0.72222222],
               [0.77777778, 0.77777778]])

    """
    if fdatagrid.ndim_domain > 2:
        raise NotImplementedError("Only support 1 or 2 dimensions on the domain.")

    if fdatagrid.ndim_domain == 1:
        axis = 1
        npoints_sample = fdatagrid.ncol
    else:
        axis = (1, 2)
        npoints_sample = len(fdatagrid.sample_points[0]) * len(fdatagrid.sample_points[1])

    univariate_depth = np.zeros(fdatagrid.shape)

    if fdatagrid.ndim_domain == 1:
        for i in range(fdatagrid.ndim_image):
            for j in range(fdatagrid.ncol):
                column = fdatagrid.data_matrix[:, j, i]
                univariate_depth[:, j, i] = 1 - np.abs(0.5 - cumulative_distribution(column))
    else:
        for i in range(fdatagrid.ndim_image):
            for j in range(len(fdatagrid.sample_points[1])):
                for k in range(len(fdatagrid.sample_points[0])):
                    column = fdatagrid.data_matrix[:, k, j, i]
                    univariate_depth[:, k, j, i] = 1 - np.abs(0.5 - cumulative_distribution(column))
    
                    
    if pointwise:
        return np.sum(univariate_depth, axis = axis) / npoints_sample, univariate_depth
    else:
        return np.sum(univariate_depth, axis = axis) / npoints_sample


def directional_outlyingness(fdatagrid,  depth_method = modified_band_depth, dim_weights = None, pointwise_weights = None):
    """Calculates both the mean and the variation of the  directional outlyingness of the samples in the data set.
    The first one describes the relative position (including both distance and direction) of the samples on average to
    the center curve and its norm can be regarded as the magnitude outlyingness. The second one measures the change of
    the directional outlyingness in terms of both norm and direction across the whole design interval and can be
    regarded as the shape outlyingness.

    Firstly, the directional outlyingness is calculated as follows:
    O(X(t) , F X(t) ) = o(X(t) , F X(t) ) · v(t) = 1 / d(X(t) , F X(t) ) − 1 · v(t)
    where X is a stochastic process with probability distribution F, d a depth function and v(t) = { X(t) − Z(t) } /∥ X(t) − Z(t) ∥
    is the spatial sign of { X(t) − Z(t) } , Z(t) denotes the unique median and ∥ · ∥ denotes the L 2 norm.

    From the above formula, we define the mean directional outlyingness as:
        MO(X , F X ) = ∫ O(X(t) , F X(t) ) w (t)dt ;
    and the variation of the directional outlyingness as:
        VO(X , F X ) = ∫ ∥ O(X(t) , F X(t) ) − MO(X , F X ) ∥ 2 w (t)dt .
    where w(t) a weight function defined on the domain of X, I.

    Then, the total functional outlyingness can be computed using these values:
        FO(X , F X ) = ∥ MO(X , F X ) ∥ 2 + VO(X , F X ) .


    . If we call :math:`D` the
    data_matrix, :math:`D^1` the derivative of order 1 and :math:`T` the
    vector contaning the points of discretisation; :math:`D^1` is
    calculated as it follows:

        .. math::

            D^{1}_{ij} = \begin{cases}
            \frac{D_{i1} - D_{i2}}{ T_{1} - T_{2}}  & \mbox{if } j = 1 \\
            \frac{D_{i(m-1)} - D_{im}}{ T_{m-1} - T_m}  & \mbox{if }
                j = m \\
            \frac{D_{i(j-1)} - D_{i(j+1)}}{ T_{j-1} - T_{j+1}} & \mbox{if }
            1 < j < m
            \end{cases}

        Where m is the number of columns of the matrix :math:`D`.

        Order > 1 derivatives are calculated by using derivative recursively.

    Args:
        fdatagrid (FDataGrid): Object containing the samples to be ordered according to
            the directional outlyingness.
        depth_method (depth function, optional): Method used to order the data (Fraiman_Muniz_depth,
        band_depth, modified_band_depth). Defaults to modified_band_depth.
    	dim_weights (array_like, optional): an array containing the weights of each of
    	    the dimensions of the image. Defaults to the same weight for each of the
    	    dimensions: 1/ndim_image.
        pointwise_weights (array_like, optional): an array containing the weights of each
            point of discretisation where values have been recorded. Defaults to the same
            weight for each of the points: 1/len(interval).

    Returns:
        mean_dir_outl (array_like): List containing the values of the magnitude
            outlyingness for each of the samples.
        variation_dir_outl (array_like): List containing the values of the shape
            outlyingness for each of the samples.
    """

    if fdatagrid.ndim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if dim_weights is not None and (len(dim_weights) != fdatagrid.ndim_image or dim_weights.sum() != 1):
        raise ValueError(
            "There must be a weight in dim_weights for each dimension of the image and altogether must sum 1.")

    if pointwise_weights is not None and (len(pointwise_weights) != fdatagrid.ncol or pointwise_weights.sum() != 1):
        raise ValueError("There must be a weight in pointwise_weights for each recorded time point and altogether must sum 1.")

    depth, depth_pointwise = depth_method(fdatagrid, pointwise=True)

    if dim_weights is None:
        dim_weights = np.ones(fdatagrid.ndim_image) / fdatagrid.ndim_image

    if pointwise_weights is None:
        pointwise_weights = np.ones(fdatagrid.ncol) / fdatagrid.ncol

    # Calculation of the depth of each multivariate sample with the corresponding weight.
    weighted_depth = depth * dim_weights
    sample_depth = weighted_depth.sum(axis=-1)

    # Obtaining the median sample Z, to caculate v(t) = {X(t) − Z(t)}/∥ X(t) − Z(t)∥
    median_index = np.argmax(sample_depth)
    median = fdatagrid.data_matrix[median_index]
    v = fdatagrid.data_matrix - median
    v_norm = LA.norm(v, axis=-1, keepdims=True)
    #To avoid ZeroDivisionError, the zeros are substituted by ones.
    v_norm[np.where(v_norm == 0)] = 1
    v_unitary = v / v_norm

    # Calculation of the depth of each point of each sample with the corresponding weight.
    weighted_depth_pointwise = depth_pointwise * dim_weights
    sample_depth_pointwise = weighted_depth_pointwise.sum(axis=-1, keepdims=True)

    # Calcuation directinal outlyingness
    dir_outlyingness = (1 / sample_depth_pointwise - 1) * v_unitary

    # Calcuation mean directinal outlyingness
    pointwise_weights_1 = np.tile(pointwise_weights, (fdatagrid.ndim_image, 1)).T
    weighted_dir_outlyingness = dir_outlyingness * pointwise_weights_1
    mean_dir_outl = weighted_dir_outlyingness.sum(axis=1)

    # Calcuation variation directinal outlyingness
    mean_dir_outl_pointwise = np.repeat(mean_dir_outl, fdatagrid.ncol, axis=0).reshape(fdatagrid.shape)
    norm = np.square(LA.norm(dir_outlyingness - mean_dir_outl_pointwise, axis=-1))
    weighted_norm = norm * pointwise_weights
    variation_dir_outl = weighted_norm.sum(axis=1)

    return mean_dir_outl, variation_dir_outl
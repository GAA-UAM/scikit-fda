"""Magnitude-Shape Plot Module.

This module contains the necessary functions to construct the Magnitude-Shape Plot.
First the directional outlingness is calculated and then, an outliers detection method is implemented.

"""

import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import f, variation
from numpy import linalg as la


from .grid import FDataGrid
from fda.depth_measures import *

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"

def directional_outlyingness(fdatagrid,  depth_method = modified_band_depth, dim_weights = None, pointwise_weights = None):
    r"""Calculates both the mean and the variation of the  directional outlyingness of the samples in the data set.

    The first one describes the relative position (including both distance and direction) of the samples on average to
    the center curve and its norm can be regarded as the magnitude outlyingness.

    The second one measures the change of the directional outlyingness in terms of both norm and direction across the whole design interval and can be
    regarded as the shape outlyingness.

    Firstly, the directional outlyingness is calculated as follows:

    .. math::
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) = \left\{\frac{1}{d\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right)} - 1\right\} \cdot \mathbf{v}(t)

    where :math:`\mathbf{X}` is a stochastic process with probability distribution :math:`F`, :math:`d` a depth function and
    :math:`\mathbf{v}(t) = \left\{\mathbf{X}(t) - \mathbf{Z}(t)\right\} / \lVert \mathbf{X}(t) - \mathbf{Z}(t) \rVert`
    is the spatial sign of :math:`\left\{\mathbf{X}(t) - \mathbf{Z}(t)\right\}`, :math:`\mathbf{Z}(t)` denotes the median and ∥ · ∥ denotes the :math:`L_2` norm.

    From the above formula, we define the mean directional outlyingness as:

    .. math::
        \mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I  \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) \cdot w(t) dt ;

    and the variation of the directional outlyingness as:

    .. math::
        VO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I  \lVert\mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right)-\mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right)  \rVert \cdot w(t) dt

    where :math:`w(t)` a weight function defined on the domain of :math:`\mathbf{X}`, :math:`I`.

    Then, the total functional outlyingness can be computed using these values:

    .. math::
        FO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \lVert \mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right)\rVert^2 +  VO\left(\mathbf{X} , F_{\mathbf{X}}\right) .

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
        (tuple): tuple containing:

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
    v_norm = la.norm(v, axis=-1, keepdims=True)
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
    norm = np.square(la.norm(dir_outlyingness - mean_dir_outl_pointwise, axis=-1))
    weighted_norm = norm * pointwise_weights
    variation_dir_outl = weighted_norm.sum(axis=1)

    return mean_dir_outl, variation_dir_outl

def magnitude_shape_plot(fdatagrid, ax=None, depth_method=modified_band_depth, dim_weights=None,
                         pointwise_weights=None, alpha = 0.993, color = 'blue' , outliercol = 'red',
                         xlabel = 'MO', ylabel='VO', title='MS-Plot'):

    r"""Implementation of the magnitude-shape plot which is based on the
    calculation of the directional outlyingness of each of the samples and
    serves as a visualization tool for the centrality of curves. Furthermore,
    an outlier detection procedure is included.

    The norm of the mean of the directional outlyingness (:math:`\lVert\mathbf{MO}\rVert`)
    is plotted in the x-axis, and the variation of the directional outlyingness (:math:`VO`)
    in the y-axis.

    Considering :math:`\mathbf{Y} = \left(\mathbf{MO}^T, VO\right)^T`, the outlier detection method
    is implemented as described below.

    First, the square robust Mahalanobis distance of :math:`\mathbf{Y}` is calculated with the minimum
    covariance determinant (MCD) estimators for shape and location of the data:
    :math:`RMD^2\left( \mathbf{Y}, \mathbf{\tilde{Y}}_J\right)`, where :math:`J` denotes the group
    of :math:`h \left(h < fdatagrid.nsamples\right)` samples that minimizes the determinant and
    :math:`\mathbf{\tilde{Y}}_J = h^{-1}\sum_{i\in{J}}\mathbf{Y}_i`.

    Then, the tail of this distance distribution is approximated as follows:

    .. math::
        \frac{c\left(m − p\right)}{m\left(p + 1\right)}RMD^2\left( \mathbf{Y}\right)\sim F_{p+1, m-p}

    where :math:`c` and :math:`m` are parameters determining the degrees of freedom of the :math:`F`-distribution
    and the scaling factor.

    .. math::
        c = E \left[s_{jj}\right]

    where :math:`s_{jj}` are the diagonal elements of MCD and

    .. math::
        m = \frac{2}{CV^2}

    where :math:`CV` is the estimated coefficient of variation of the diagonal elements of the  MCD shape estimator.

    Finally, we choose a cutoff value, C , as the α quantile of :math:`F_{p+1, m-p}`. We set :math:`\alpha = 0.993`,
    which is used in the classical boxplot for detecting outliers under a normal distribution.

    Args:
        fdatagrid (FDataGrid): Object to be visualized.
        ax (axis object, optional): axis over which the graph is plotted.
                Defaults to matplotlib current axis.
        depth_method (depth function, optional): Method used to order the data (Fraiman_Muniz_depth,
            band_depth, modified_band_depth). Defaults to modified_band_depth.
        dim_weights (array_like, optional): an array containing the weights of each of
            the dimensions of the image.
        pointwise_weights (array_like, optional): an array containing the weights of each
            point of discretisation where values have been recorded.
        alpha(int, optional): Denotes the quantile to choose the cutoff value for detecting outliers
            Defaults to 0.993,  which is used in the classical boxplot.
        color (matplotlib.colors, optional): color of the points plotted. Defaults to 'blue'.
        outliercol (matplotlib.colors, optional): color of the outlier points plotted. Defaults to 'red'.
        xlabel (string, optional): Label of the x-axis. Defaults to 'MO', mean of the  directional outlyingness.
        ylabel (string, optional): Label of the y-axis. Defaults to 'VO', variation of the  directional outlyingness.
        title (string, optional): Title of the plot. defaults to 'MS-Plot'.


    Returns:
        (tuple): tuple containing:

            points(numpy.ndarray): 2-dimensional matrix where each row
                contains the points plotted in the graph.
            outliers (1-D array: (fdatagrid.nsamples,)): Contains outliercol or color to denote if a point is
                an outlier or not, respecively.

    """
    if fdatagrid.ndim_image > 1:
        raise NotImplementedError("Only support 1 dimension on the image.")

    # The depths of the samples are calculated giving them an ordering.
    mean_dir_outl, variation_dir_outl = directional_outlyingness(fdatagrid, depth_method, dim_weights, pointwise_weights)
    points = np.array(list(zip(mean_dir_outl, variation_dir_outl)))

    #The square mahalanobis distances of the samples are calulated using MCD.
    cov = MinCovDet(store_precision=True).fit(points)
    rmd_2 = cov.mahalanobis(points)

    #Calculation of the degrees of freedom of the F-distribution (approximation of the tail of the distance distribution).
    s_jj = np.diag(cov.covariance_)
    c = np.mean(s_jj)
    m = 2 / np.square(variation(s_jj))
    p = fdatagrid.ndim_image
    dfn = p + 1
    dfd = m - p

    #Calculation of the cutoff value and scaling factor to identify outliers.
    cutoff_value = f.ppf(alpha, dfn, dfd, loc=0, scale=1)
    scaling = c * dfd / m / dfn
    outliers = (scaling * rmd_2 > cutoff_value)

    #Plotting the data
    outliers = outliers.astype(str)
    outliers[np.where(outliers == 'True')] = outliercol
    outliers[np.where(outliers == 'False')] = color

    if ax is None:
        ax = matplotlib.pyplot.gca()
    ax.scatter(mean_dir_outl, variation_dir_outl, color=outliers)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return points, outliers
import typing

from numpy import linalg as la
import scipy.integrate

import numpy as np

from ... import FDataGrid
from ..depth import fraiman_muniz_depth


class DirectionalOutlyingnessStats(typing.NamedTuple):
    directional_outlyingness: np.ndarray
    functional_directional_outlyingness: np.ndarray
    mean_directional_outlyingness: np.ndarray
    variation_directional_outlyingness: np.ndarray


def directional_outlyingness_stats(
        fdatagrid: FDataGrid,
        depth_method=fraiman_muniz_depth,
        pointwise_weights=None) -> DirectionalOutlyingnessStats:
    r"""Computes the directional outlyingness of the functional data.

    Furthermore, it calculates functional, mean and the variational
    directional outlyingness of the samples in the data set, which are also
    returned.

    The functional directional outlyingness can be seen as the overall
    outlyingness, analog to other functional outlyingness measures.

    The mean directional outlyingness, describes the relative
    position (including both distance and direction) of the samples on average
    to the center curve; its norm can be regarded as the magnitude
    outlyingness.

    The variation of the directional outlyingness, measures
    the change of the directional outlyingness in terms of both norm and
    direction across the whole design interval and can be regarded as the
    shape outlyingness.

    Firstly, the directional outlyingness is calculated as follows:

    .. math::
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) =
        \left\{\frac{1}{d\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right)} - 1
        \right\} \cdot \mathbf{v}(t)

    where :math:`\mathbf{X}` is a stochastic process with probability
    distribution :math:`F`, :math:`d` a depth function and :math:`\mathbf{v}(t)
    = \left\{ \mathbf{X}(t) - \mathbf{Z}(t)\right\} / \lVert \mathbf{X}(t) -
    \mathbf{Z}(t) \rVert` is the spatial sign of :math:`\left\{\mathbf{X}(t) -
    \mathbf{Z}(t)\right\}`, :math:`\mathbf{Z}(t)` denotes the median and
    :math:`\lVert \cdot \rVert` denotes the :math:`L_2` norm.

    From the above formula, we define the mean directional outlyingness as:

    .. math::
        \mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) \cdot w(t) dt;

    and the variation of the directional outlyingness as:

    .. math::
        VO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I  \lVert\mathbf{O}
        \left(\mathbf{X}(t), F_{\mathbf{X}(t)}\right)-\mathbf{MO}\left(
        \mathbf{X} , F_{\mathbf{X}}\right)  \rVert^2 \cdot w(t) dt

    where :math:`w(t)` a weight function defined on the domain of
    :math:`\mathbf{X}`, :math:`I`.

    Then, the total functional outlyingness can be computed using these values:

    .. math::
        FO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \lVert \mathbf{MO}\left(
        \mathbf{X} , F_{\mathbf{X}}\right)\rVert^2 +  VO\left(\mathbf{X} ,
        F_{\mathbf{X}}\right) .

    Args:
        fdatagrid (FDataGrid): Object containing the samples to be ordered
            according to the directional outlyingness.
        depth_method (:ref:`depth measure <depth-measures>`, optional): Method
            used to order the data. Defaults to :func:`modified band depth
            <fda.depth_measures.modified_band_depth>`.
        pointwise_weights (array_like, optional): an array containing the
            weights of each point of discretisation where values have been
            recorded. Defaults to the same weight for each of the points:
            1/len(interval).

    Returns:
        DirectionalOutlyingnessStats object.

    Example:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> stats = directional_outlyingness_stats(fd)
        >>> stats.directional_outlyingness
        array([[[ 1.        ],
                [ 1.        ],
                [ 1.        ],
                [ 1.        ],
                [ 1.        ],
                [ 1.        ]],
               [[ 0.33333333],
                [ 0.33333333],
                [ 0.33333333],
                [ 0.33333333],
                [ 0.33333333],
                [ 0.33333333]],
               [[-0.33333333],
                [-0.33333333],
                [ 0.        ],
                [ 0.        ],
                [ 0.        ],
                [ 0.        ]],
               [[ 0.        ],
                [ 0.        ],
                [ 0.        ],
                [-0.33333333],
                [-0.33333333],
                [-0.33333333]]])

    >>> stats.functional_directional_outlyingness
    array([ 3.93209877,  3.27366255,  3.23765432,  3.25823045])

    >>> stats.mean_directional_outlyingness
    array([[ 1.66666667],
           [ 0.55555556],
           [-0.16666667],
           [-0.27777778]])

    >>> stats.variation_directional_outlyingness
    array([ 0.74074074,  0.08230453,  0.0462963 ,  0.06687243])

    """
    if fdatagrid.ndim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if (pointwise_weights is not None and
        (len(pointwise_weights) != len(fdatagrid.sample_points[0]) or
         pointwise_weights.sum() != 1)):
        raise ValueError(
            "There must be a weight in pointwise_weights for each recorded "
            "time point and altogether must sum 1.")

    if pointwise_weights is None:
        pointwise_weights = np.ones(
            len(fdatagrid.sample_points[0])) / len(fdatagrid.sample_points[0])

    _, depth_pointwise = depth_method(fdatagrid, pointwise=True)
    assert depth_pointwise.shape == fdatagrid.data_matrix.shape[:-1]

    # Obtaining the pointwise median sample Z, to calculate
    # v(t) = {X(t) − Z(t)}/|| X(t) − Z(t) ||
    median_index = np.argmax(depth_pointwise, axis=0)
    pointwise_median = fdatagrid.data_matrix[
        median_index, range(fdatagrid.data_matrix.shape[1])]
    assert pointwise_median.shape == fdatagrid.shape[1:]
    v = fdatagrid.data_matrix - pointwise_median
    assert v.shape == fdatagrid.data_matrix.shape
    v_norm = la.norm(v, axis=-1, keepdims=True)

    # To avoid ZeroDivisionError, the zeros are substituted by ones (the
    # reference implementation also does this).
    v_norm[np.where(v_norm == 0)] = 1

    v_unitary = v / v_norm

    # Calculation directinal outlyingness
    dir_outlyingness = (1 / depth_pointwise[..., np.newaxis] - 1) * v_unitary
    assert dir_outlyingness.shape == fdatagrid.data_matrix.shape

    # Calculation mean directional outlyingness
    weighted_dir_outlyingness = (dir_outlyingness
                                 * pointwise_weights[:, np.newaxis])
    assert weighted_dir_outlyingness.shape == dir_outlyingness.shape

    mean_dir_outlyingness = scipy.integrate.simps(weighted_dir_outlyingness,
                                                  fdatagrid.sample_points[0],
                                                  axis=1)
    assert mean_dir_outlyingness.shape == (
        fdatagrid.nsamples, fdatagrid.ndim_codomain)

    # Calculation variation directional outlyingness
    norm = np.square(la.norm(dir_outlyingness -
                             mean_dir_outlyingness[:, np.newaxis, :], axis=-1))
    weighted_norm = norm * pointwise_weights
    variation_dir_outlyingness = scipy.integrate.simps(
        weighted_norm, fdatagrid.sample_points[0], axis=1)
    assert variation_dir_outlyingness.shape == (fdatagrid.nsamples,)

    functional_dir_outlyingness = (np.square(la.norm(mean_dir_outlyingness))
                                   + variation_dir_outlyingness)
    assert functional_dir_outlyingness.shape == (fdatagrid.nsamples,)

    return DirectionalOutlyingnessStats(
        directional_outlyingness=dir_outlyingness,
        functional_directional_outlyingness=functional_dir_outlyingness,
        mean_directional_outlyingness=mean_dir_outlyingness,
        variation_directional_outlyingness=variation_dir_outlyingness)

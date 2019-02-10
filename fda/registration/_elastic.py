

import numpy as np
import scipy.integrate


from ..functional_data import FData
from ..grid import FDataGrid, GridSplineInterpolator
import optimum_reparamN


def to_srsf(fdatagrid, eval_points=None):
    """Calculate the square-root slope function (SRSF) transform.

    Let :math:`f_i : [a,b] \\rightarrow \mathbb{R}` be an absolutely continuous
    function, the SRSF transform is defined as

    .. math::
        SRSF(f_i(t)) = sgn(f_i(t)) \\sqrt{|Df_i(t)|} = q_i(t)

    This representation it is used to compute the extended non-parametric
    Fisher-Rao distance between functions, wich under the SRSF representation
    becomes the usual :math:`\mathbb{L}^2` distance between functions.
    See [SK16-4-6-1]_ .

    Args:
        fdatagrid (:class:`FDataGrid`): Functions to be transformed.
        eval_points: (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of the
            fdatagrid.

    Returns:
        :class:`FDataGrid`: SRSF functions.

    Raises:
        ValueError: If functions are multidimensional.

    References:
        ..  [SK16-4-6-1] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Square-Root Slope Function
            Representation* (pp. 91-93). Springer.

    """

    if fdatagrid.ndim_domain > 1:
        raise ValueError("Only support functional objects with unidimensional "
                         "domain.")

    elif fdatagrid.ndim_image > 1:
        raise ValueError("Only support functional objects with unidimensional "
                         "image.")

    elif eval_points is None:
        eval_points = fdatagrid.sample_points[0]

    g = fdatagrid.derivative()

    # Evaluation with the corresponding interpolation
    g_data_matrix = g(eval_points, keepdims=False)

    # SRVF(f) = sign(f) * sqrt|Df|
    q_data_matrix = np.sign(g_data_matrix) * np.sqrt(np.abs(g_data_matrix))

    return fdatagrid.copy(data_matrix=q_data_matrix, sample_points=eval_points)


def from_srsf(fdatagrid, initial=None, *, eval_points=None):
    """Given a SRSF calculate the corresponding function in the original space.

    Let :math:`f_i : [a,b]\\rightarrow \mathbb{R}` be an absolutely continuous
    function, the SRSF transform is defined as

    .. math::
        SRSF(f_i(t)) = sgn(f_i(t)) \sqrt{|Df_i(t)|} = q_i(t)

    This transformation is a mapping up to constant. Given the srsf and the
    initial value the original function can be obtained as

    .. math::
        f_i(t) = f(a) + \int_{a}^t q(t)|q(t)|dt

    This representation it is used to compute the extended non-parametric
    Fisher-Rao distance between functions, wich under the SRSF representation
    becomes the usual :math:`\mathbb{L}^2` distance between functions.
    See [SK16-4-6-2]_ .

    Args:
        fdatagrid (:class:`FDataGrid`): SRSF to be transformed.
        initial (array_like): List of values of initial values of the original
            functions.
        eval_points: (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of the
            fdatagrid.

    Returns:
        :class:`FDataGrid`: Functions in the original space.

    Raises:
        ValueError: If functions are multidimensional.

    References:
        ..  [SK16-4-6-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Square-Root Slope Function
            Representation* (pp. 91-93). Springer.

    """

    if fdatagrid.ndim_domain > 1:
        raise ValueError("Only support functional objects with "
                         "unidimensional domain.")

    elif fdatagrid.ndim_image > 1:
        raise ValueError("Only support functional objects with unidimensional "
                         "image.")

    elif eval_points is None:
        eval_points = fdatagrid.sample_points[0]


    q_data_matrix = fdatagrid(eval_points, keepdims=True)

    f_data_matrix = q_data_matrix * np.abs(q_data_matrix)

    f_data_matrix = scipy.integrate.cumtrapz(f_data_matrix,
                                             x=eval_points,
                                             axis=1,
                                             initial=0)

    if initial is not None:
        initial = initial.reshape(fdatagrid.nsamples, 1, fdatagrid.ndim_image)
        initial = np.repeat(initial, len(eval_points), axis=1)
        f_data_matrix +=  initial


    return fdatagrid.copy(data_matrix=f_data_matrix, sample_points=eval_points)


def _normalize_scale(t, a=0, b=1):
    """Perfoms an afine translation to normalize an interval.

    Args:
        t (numpy.ndarray): Array of dim 1 or 2 with at least 2 values by row.
        a (float): Starting point of the new interval. Defaults 0.
        b (float): Stopping point of the new interval. Defaults 1.
    """

    t = t.T # Broadcast to normalize multiple arrays
    t1 = t - t[0] # Translation to [0, t[-1] - t[0]]
    t1 *= (b - a) / (t[-1] - t[0]) # Scale to [0, b-a]
    t1 += a # Translation to [a, b]
    t1[0] = a # Fix possible round errors
    t1[-1] = b

    t1 = t1.T

    return t1

def elastic_registration_warping(fdatagrid, template=None, *, lam=0.,
                                 eval_points=None, fdatagrid_srsf=None,
                                 template_srsf=None):
    """Calculate the warping to align a FDatagrid using the SRSF framework.

    Let :math:`f` be a function of the functional data object wich will be
    aligned to the template :math:`g`. Calculates the warping wich minimises
    the Fisher-Rao distance between :math:`g` and the registered function
    :math:`f^*(t)=f \\circ \\gamma^*`.

    .. math::
        \\gamma^* = argmin_{\\gamma \\in \\Gamma} d_{\\lambda}(f \\circ
        \\gamma, g)

    Where :math:`d_{\\lambda}` denotes the extended Fisher-Rao distance with a
    penalisation term, used to control the amount of warping.

    .. math::
        d_{\\lambda}^2(f \\circ \\gamma, g) = \| SRSF(f \\circ \\gamma)
        \\sqrt{\\dot{\\gamma}} - SRSF(g)\|_{\\mathbb{L}^2}^2 + \|
        \\sqrt{\\dot{\\gamma}} - 1 \|_{\\mathbb{L}^2}^2

    The registered function :math:`f^*(t)` can be calculated using the
    composition :math:`f^*(t)=f(\\gamma^*(t))`.

    If the template is not specified it is used the Karcher mean of the set of
    functions under the Fisher-Rao metric to perform the alignment. See
    #REFERENCE TO KARCHER MEAN FUNCTION.

    See [SK16-4-1]_ for an extensive explanation.

    Args:
        fdatagrid (:class:`FDataGrid`): Functional data object to be aligned.
        template (:class:`FDataGrid` or callable, optional):
        lam (float, optional): Controls the amount of elasticity. Defaults to 0.
        eval_points (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of the
            fdatagrid.
        fdatagrid_srsf (:class:`FDataGrid`, optional): SRSF of the fdatagrid,
            may be passed to avoid repeated calculation.
        template_srsf (:class:`FDataGrid`, optional): SRSF of the template,
            may be passed to avoid repeated calculation.

    Returns:
        (:class:`FDataGrid`): Warping to align the given fdatagrid to the
        template.

    Raises:
        ValueError: If functions are multidimensional or the number of samples
            are different.

    References:
        ..  [SK16-4-1] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Functional Data and Elastic
            Registration* (pp. 73-122). Springer.

    """

    # Check of params
    if fdatagrid.ndim_domain != 1 or fdatagrid.ndim_image != 1:

        raise ValueError("Not supported multidimensional functional objects.")

    elif isinstance(template, FData) and (
        (template.nsamples != 1 and template.nsamples != fdatagrid.nsamples) or
        template.ndim_domain != 1 or template.ndim_image != 1):

        raise ValueError("The template should contain one sample to align all"
                         "the curves to the same function or the same number "
                         "of samples than the fdatagrid")
    elif template is None:
        # Defaults uses karcher mean in the future
        template = fdatagrid.mean()

    elif not isinstance(template, FData):

        # Defaults calls to construct template
        template = template(fdatagrid)

    # Construction of srsfs
    if fdatagrid_srsf is None:
        fdatagrid_srsf = to_srsf(fdatagrid, eval_points=eval_points)

    if template_srsf is None:
        template_srsf = to_srsf(template, eval_points=eval_points)

    if eval_points is None:
        eval_points = fdatagrid_srsf.sample_points[0]

    # Discretizacion in evaluation points
    q_data = fdatagrid_srsf(eval_points, keepdims=False).squeeze()
    template_data = template_srsf(eval_points, keepdims=False).squeeze()

    # Select cython function
    if template_data.ndim == 1 and q_data.ndim == 1:
        reparam = optimum_reparamN.coptimum_reparam

    elif template_data.ndim == 1:
        reparam = optimum_reparamN.coptimum_reparamN

    else:
        reparam = optimum_reparamN.coptimum_reparamN2

    # Values of the warping
    gamma = reparam(np.ascontiguousarray(template_data.T),
                    np.ascontiguousarray(_normalize_scale(eval_points)),
                    np.ascontiguousarray(q_data.T),
                    lam).T

    # Normalize warping to original interval
    gamma = _normalize_scale(gamma, a=eval_points[0], b=eval_points[-1])

    # Interpolator
    interpolator = GridSplineInterpolator(interpolation_order=3, monotone=True)

    return FDataGrid(gamma, eval_points, interpolator=interpolator)

def elastic_registration(fdatagrid, template, *, lam=0., eval_points=None,
                       fdatagrid_srsf=None, template_srsf=None):
    """Calculate the warping to align a FDatagrid using the SRSF framework.

    Let :math:`f` be a function of the functional data object wich will be
    aligned to the template :math:`g`. Calculates the warping wich minimises
    the Fisher-Rao distance between :math:`g` and the registered function
    :math:`f^*(t)=f \\circ \\gamma^*`.

    .. math::
        \\gamma^* = argmin_{\\gamma \\in \\Gamma} d_{\\lambda}(f \\circ
        \\gamma, g)

    Where :math:`d_{\\lambda}` denotes the extended Fisher-Rao distance with a
    penalisation term, used to control the amount of warping.

    .. math::
        d_{\\lambda}^2(f \\circ \\gamma, g) = \| SRSF(f \\circ \\gamma)
        \\sqrt{\\dot{\\gamma}} - SRSF(g)\|_{\\mathbb{L}^2}^2 + \|
        \\sqrt{\\dot{\\gamma}} - 1 \|_{\\mathbb{L}^2}^2

    The registered function :math:`f^*(t)` can be calculated using the
    composition :math:`f^*(t)=f(\\gamma^*(t))`.

    If the template is not specified it is used the Karcher mean of the set of
    functions under the Fisher-Rao metric to perform the alignment. See
    #REFERENCE TO KARCHER MEAN FUNCTION.

    See [SK16-4-2]_ for an extensive explanation.

    Args:
        fdatagrid (:class:`FDataGrid`): Functional data object to be aligned.
        template (:class:`FDataGrid` or callable, optional):
        lam (float, optional): Controls the amount of elasticity. Defaults to 0.
        eval_points (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of the
            fdatagrid.
        fdatagrid_srsf (:class:`FDataGrid`, optional): SRSF of the fdatagrid,
            may be passed to avoid repeated calculation.
        template_srsf (:class:`FDataGrid`, optional): SRSF of the template,
            may be passed to avoid repeated calculation.

    Returns:
        (:class:`FDataGrid`): Warping to align the given fdatagrid to the
        template.

    Raises:
        ValueError: If functions are multidimensional or the number of samples
            are different.

    References:
        ..  [SK16-4-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Functional Data and Elastic
            Registration* (pp. 73-122). Springer.

    """

    # Calculates corresponding set of warpings
    warping = elastic_registration_warping(fdatagrid, template, lam=lam,
                                           eval_points=eval_points,
                                           fdatagrid_srsf=fdatagrid_srsf,
                                           template_srsf=template_srsf)

    return fdatagrid.compose(warping, eval_points=eval_points)


def amplitude_distance(fdatagrid1, fdatagrid2):
    pass

def phase_distance(fdatagrid1, fdatagrid2):
    pass

def warping_phase_distance(fdatagrid1, fdatagrid2):
    pass

def elastic_mean(fdatagrid, lam, eval_points, iter=20, tol=1e-2):
    pass

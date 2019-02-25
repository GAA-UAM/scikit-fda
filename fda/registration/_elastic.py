

import numpy as np
import scipy.integrate


from ..functional_data import FData
from ..grid import FDataGrid, GridSplineInterpolator
from ..math import metric
from . import invert_warping

import optimum_reparamN

import matplotlib.pyplot as plt



def to_srsf(fdatagrid, eval_points=None):
    """Calculate the square-root slope function (SRSF) transform.

    Let :math:`f_i : [a,b] \\rightarrow \\mathbb{R}` be an absolutely continuous
    function, the SRSF transform is defined as

    .. math::
        SRSF(f_i(t)) = sgn(f_i(t)) \\sqrt{|Df_i(t)|} = q_i(t)

    This representation it is used to compute the extended non-parametric
    Fisher-Rao distance between functions, wich under the SRSF representation
    becomes the usual :math:`\\mathbb{L}^2` distance between functions.
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

    # SRSF(f) = sign(f) * sqrt|Df|
    q_data_matrix = np.sign(g_data_matrix) * np.sqrt(np.abs(g_data_matrix))

    return fdatagrid.copy(data_matrix=q_data_matrix, sample_points=eval_points)


def from_srsf(fdatagrid, initial=None, *, eval_points=None):
    """Given a SRSF calculate the corresponding function in the original space.

    Let :math:`f_i : [a,b]\\rightarrow \\mathbb{R}` be an absolutely continuous
    function, the SRSF transform is defined as

    .. math::
        SRSF(f_i(t)) = sgn(f_i(t)) \\sqrt{|Df_i(t)|} = q_i(t)

    This transformation is a mapping up to constant. Given the srsf and the
    initial value the original function can be obtained as

    .. math::
        f_i(t) = f(a) + \\int_{a}^t q(t)|q(t)|dt

    This representation it is used to compute the extended non-parametric
    Fisher-Rao distance between functions, wich under the SRSF representation
    becomes the usual :math:`\\mathbb{L}^2` distance between functions.
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
        initial = np.atleast_1d(initial)
        initial = initial.reshape(fdatagrid.nsamples, 1, fdatagrid.ndim_image)
        initial = np.repeat(initial, len(eval_points), axis=1)
        f_data_matrix +=  initial


    return fdatagrid.copy(data_matrix=f_data_matrix, sample_points=eval_points)


def _normalize_scale(t, a=0, b=1):
    """Perfoms an afine translation to normalize an interval.

    Args:
        t (numpy.ndarray): Array of dim 1 or 2 with at least 2 values.
        a (float): Starting point of the new interval. Defaults 0.
        b (float): Stopping point of the new interval. Defaults 1.

    Returns:
        (numpy.ndarray): Array with the transformed interval.
    """

    t = t.T # Broadcast to normalize multiple arrays
    t1 = t - t[0] # Translation to [0, t[-1] - t[0]]
    t1 *= (b - a) / (t[-1] - t[0]) # Scale to [0, b-a]
    t1 += a # Translation to [a, b]
    t1[0] = a # Fix possible round errors
    t1[-1] = b

    return t1.T

def _elastic_alignment_array(template_data, q_data, eval_points, lam):
    """Wrapper between cython interface and python.

    Selects the corresponding routine depending on the dimensions of the arrays.

    Args:
        template_data (numpy.ndarray): Array with the srsf of the template.
        q_data (numpy.ndarray): Array with the srsf of the curves to be aligned.
        eval_points (numpy.ndarray): Discretisation points of the functions.
        lam (float): Penalisation term.

    Return:
        (numpy.ndarray): Array with the same shape than q_data with the srsf of
        the functions aligned to the template(s).
    """

    # Select cython function
    if template_data.ndim == 1 and q_data.ndim == 1:
        reparam = optimum_reparamN.coptimum_reparam

    elif template_data.ndim == 1:
        reparam = optimum_reparamN.coptimum_reparamN

    else:
        reparam = optimum_reparamN.coptimum_reparamN2


    return reparam(np.ascontiguousarray(template_data.T),
                   np.ascontiguousarray(eval_points),
                   np.ascontiguousarray(q_data.T),
                   lam).T

def elastic_registration_warping(fdatagrid, template=None, *, lam=0.,
                                 eval_points=None, fdatagrid_srsf=None,
                                 template_srsf=None, **kwargs):
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
    :func:`elastic_mean`.

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


    if template is None:
        template = elastic_mean(fdatagrid, lam=lam, eval_points=eval_points,
                                **kwargs)

    elif ((template.nsamples != 1 and template.nsamples != fdatagrid.nsamples) or
        template.ndim_domain != 1 or template.ndim_image != 1):

        raise ValueError("The template should contain one sample to align all"
                         "the curves to the same function or the same number "
                         "of samples than the fdatagrid")


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


    # Values of the warping
    gamma = _elastic_alignment_array(template_data, q_data,
                                     _normalize_scale(eval_points), lam)

    # Normalize warping to original interval
    gamma = _normalize_scale(gamma, a=eval_points[0], b=eval_points[-1])

    # Interpolator
    interpolator = GridSplineInterpolator(interpolation_order=3, monotone=True)

    return FDataGrid(gamma, eval_points, interpolator=interpolator)

def elastic_registration(fdatagrid, template=None, *, lam=0., eval_points=None,
                        fdatagrid_srsf=None, template_srsf=None, **kwargs):
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
    :func:`elastic_mean`.

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
                                           template_srsf=template_srsf,
                                           **kwargs)

    return fdatagrid.compose(warping, eval_points=eval_points)


def amplitude_distance(fdatagrid1, fdatagrid2):
    pass

def phase_distance(fdatagrid1, fdatagrid2):
    pass

def warping_phase_distance(fdatagrid1, fdatagrid2):
    pass


def warping_mean(warping, *, iter=20, tol=1e-5, step_size=1.,
                 eval_points=None, return_shooting=False):
    """Compute the karcher mean of a set of warpings under the fisher-rao metric.

    Let :math:`\\gamma_i i=1...n` a set of warping functions
    :math:`\\gamma_i:[a,b] \\rightarrow [a,b]` in :math:`\\Gamma`, monotone
    increasing and with the restriction :math:`\\gamma_i(a)=a \, and
    \\gamma_i(b)=b`.

    The karcher mean :math:`\\bar \\gamma` is defined as the warping that
    minimises locally the sum of squared distances, under the Fisher-Rao metric.
    [SK16-8-3-2]_.

    .. math::
        \\bar \\gamma = \\argmin_{\\gamma in \\Gamma} \\sum_{i=1}^{n}
         d_{FR}^2(\\gamma, \\gamma_i)

    The computation is performed using the structure of Hilbert Sphere obtained
    after a transformation of the warpings, see [S11-3-3]_.

    Args:
        warping (:class:`FDataGrid`): Set of warpings.
        iter (int): Maximun number of interations. Defaults to 20.
        tol (float): Convergence criterion, if the norm of the mean of the
            shooting vectors, :math:`\| \\bar v \|<tol`, the algorithm will stop.
            Defaults to 1e-5.
        step_size (float): Step size :math:`\\epsilon` used to update the mean.
            Default to 1.
        eval_points (array_like): Discretisation points of the warpings.
        shooting (boolean): If true it is returned a tuple with the mean and the
            shooting vectors, otherwise only the mean is returned.

    Return:
        (:class:`FDataGrid`) Fdatagrid with the mean of the warpings. If
        shooting is True the shooting vectors will be returned in a tuple with
        the mean.

    References:
        ..  [SK16-8-3-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Template: Center of the Mean Orbit*
            (pp. 274-277). Springer.

        .. [S11-3-3] Srivastava, Anuj et. al. Registration of Functional Data
        Using Fisher-Rao Metric (2011). In *Center of an Orbit* (pp. 9-10).
        arXiv:1103.3817v2.


    """

    if eval_points is None:
        eval_points = warping.sample_points[0]

    original_eval_points = eval_points

    if warping.sample_points[0][0] != 0 or warping.sample_points[0][-1] != 1:

        eval_points = _normalize_scale(eval_points)
        warping = FDataGrid(_normalize_scale(warping.data_matrix[..., 0]),
                            sample_points=_normalize_scale(warping.sample_points[0]))


    psi = to_srsf(warping, eval_points=eval_points).data_matrix[..., 0].T
    mu = to_srsf(warping.mean(), eval_points=eval_points).data_matrix[0]
    dot_aux = np.empty(psi.shape)

    n_points = mu.shape[0]

    sine = np.empty((warping.nsamples, 1))


    for _ in range(iter):
        # Dot product
        # <psi, mu> = S psi(t) mu(t) dt
        dot = scipy.integrate.simps(np.multiply(psi, mu, out=dot_aux),
                                    eval_points, axis=0)

        # Theorically is not possible (Cauchy–Schwarz inequallity), but due to
        # numerical approximation could be greater than 1
        dot[dot < -1] = -1
        dot[dot > 1] = 1
        theta = np.arccos(dot)[:, np.newaxis]

        # Be carefully with tangent vectors and division by 0
        idx = theta[:, 0] > tol
        sine[idx] = theta[idx] / np.sin(theta[idx])
        sine[~idx] = 0.


        # compute shooting vector
        cos_theta = np.repeat(np.cos(theta), n_points, axis=1)
        shooting =  np.multiply(sine, (psi - np.multiply(cos_theta.T, mu)).T)


        # Mean of shooting vectors
        vmean = shooting.mean(axis=0, keepdims=True)
        v_norm = scipy.integrate.simps(np.square(vmean[0]))**(.5)

        # Convergence criterion
        if v_norm < tol:
            break

        # Update of mu
        mu *= np.cos(step_size*v_norm)
        vmean += np.sin(step_size * v_norm) / v_norm
        mu += vmean.T


    # Recover mean in original gamma space
    warping_mean = scipy.integrate.cumtrapz(np.square(mu, out=mu)[:,0],
                                            x=eval_points, initial=0)

    # Affine traslation
    warping_mean = _normalize_scale(warping_mean,
                                    a=original_eval_points[0],
                                    b=original_eval_points[-1])

    monotone_interpolator = GridSplineInterpolator(interpolation_order=3,
                                                   monotone=True)

    mean = FDataGrid([warping_mean],sample_points=original_eval_points,
                     interpolator=monotone_interpolator)

    # Shooting vectors are used in models based in the amplitude-phase
    # decomposition under this metric.
    if return_shooting:
        return mean, shooting

    return mean


def elastic_mean(fdatagrid, *, lam=0., center=True, iter=20, tol=1e-3,
                 initial=None, eval_points=None, fdatagrid_srsf=None, **kwargs):
    """Compute the karcher mean of a set of functional observations in the
    amplitude space.

    Args:
        fdatagrid (:class:`FDataGrid`): Set of functions to compute the mean.
        lam (float): Penalisation term. Defaults to 0.
        center (boolean): If true it is computed the mean of the warpings and
            used to select a central mean. Defaults True.
        iter (int): Maximun number of iterations. Defaults to 20.
        tol (float): Convergence criterion, the algorithm will stop if
            :math:´\|mu^{(\\nu)} - mu^{(\\nu - 1)} \|_2 / \| mu^{(\\nu-1)} \|_2 < tol´.
        initial (float): Value of the mean at the starting point. By default
            takes the average of the initial points of the samples.
        eval_points (array_like): Points of discretization of the fdatagrid.
        fdatagrid_srsf (:class:`FDataGrid`): SRSF if the fdatagrid, if it is
            passed it is not computed in the algorithm.
        ** kwargs : Named options to be pased to :func:`warping_mean`.

    Return:
        (:class:`FDataGrid`): FDatagrid with the mean of the functions.

    Raises:
        ValueError: If the object is multidimensional or the shape of the srsf
            do not match with the fdatagrid.


    """


    if fdatagrid.ndim_domain != 1 or fdatagrid.ndim_image != 1:
        raise ValueError("Not supported multidimensional functional objects.")

    if fdatagrid_srsf is not None and (fdatagrid_srsf.ndim_domain != 1 or
                                       fdatagrid_srsf.ndim_image != 1):
        raise ValueError("Not supported multidimensional functional objects.")

    elif fdatagrid_srsf is None:
        fdatagrid_srsf = to_srsf(fdatagrid, eval_points=eval_points)


    if eval_points is not None:
        eval_points = np.asarray(eval_points)
    else:
        eval_points = fdatagrid.sample_points[0]


    eval_points_normalized = _normalize_scale(eval_points)
    y_scale = eval_points[-1] - eval_points[0]

    interpolator = GridSplineInterpolator(interpolation_order=3, monotone=True)

    # Discretisation points
    fdatagrid_normalized = FDataGrid(fdatagrid(eval_points) / y_scale,
                                     sample_points=eval_points_normalized)

    srsf = fdatagrid_srsf(eval_points, keepdims=False)

    # Initialize with function closest to the L2 mean with the L2 distance
    centered = (srsf.T - srsf.mean(axis=0, keepdims=True).T).T

    distances = scipy.integrate.simps(np.square(centered, out=centered),
                                      eval_points_normalized, axis=1)


    # Initialization of iteration
    mu = srsf[np.argmin(distances)]
    mu_aux = np.empty(mu.shape)
    mu_1 = np.empty(mu.shape)

    # Main iteration
    for _ in range(iter):

        gammas = _elastic_alignment_array(mu, srsf, eval_points_normalized, lam)
        gammas = FDataGrid(gammas, sample_points=eval_points_normalized,
                           interpolator=interpolator)

        fdatagrid_normalized = fdatagrid_normalized.compose(gammas)
        srsf = to_srsf(fdatagrid_normalized).data_matrix[..., 0]

        # Next iteration
        mu_1 = srsf.mean(axis=0, out=mu_1)

        # Convergence criterion
        mu_norm = np.sqrt(scipy.integrate.simps(np.square(mu, out=mu_aux),
                                                eval_points_normalized))

        mu_diff = np.sqrt(scipy.integrate.simps(np.square(mu - mu_1, out=mu_aux),
                                                eval_points_normalized))

        if mu_diff / mu_norm < tol:
            break

        mu = mu_1


    if initial is None:
        initial = fdatagrid.data_matrix[:,0].mean()


    # Karcher mean orbit in space L2/Gamma
    karcher_mean = from_srsf(fdatagrid.copy(data_matrix=[mu],
                                            sample_points=eval_points),
                             initial=initial)


    if center:
        # Gamma mean in Hilbert Sphere
        mean_normalized = warping_mean(gammas, return_shooting=False, **kwargs)

        gamma_mean = FDataGrid(_normalize_scale(mean_normalized.data_matrix[...,0],
                                                a=eval_points[0],
                                                b=eval_points[-1]),
                               sample_points=eval_points)

        gamma_inverse = invert_warping(gamma_mean)

        karcher_mean = karcher_mean.compose(gamma_inverse)

    # Return center of the orbit
    return karcher_mean

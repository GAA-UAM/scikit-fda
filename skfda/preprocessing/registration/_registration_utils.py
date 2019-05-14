"""Registration of functional data module.

This module contains routines related to the registration procedure.
"""
import collections

import numpy
import scipy.integrate
from scipy.interpolate import PchipInterpolator

__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"


def mse_decomposition(original_fdata, registered_fdata, warping_function=None,
                      *, eval_points=None):
    r"""Compute mean square error measures for amplitude and phase variation.

    Once the registration has taken place, this function computes two mean
    squared error measures, one for amplitude variation, and the other for
    phase variation. It also computes a squared multiple correlation index
    of the amount of variation in the unregistered functions is due to phase.

    Let :math:`x_i(t),y_i(t)` be the unregistered and registered functions
    respectively. The total mean square error measure (see [RGS09-8-5]_) is
    defined as


    .. math::
        \text{MSE}_{total}=
        \frac{1}{N}\sum_{i=1}^{N}\int[x_i(t)-\overline x(t)]^2dt

    We define the constant :math:`C_R` as

    .. math::

        C_R = 1 + \frac{\frac{1}{N}\sum_{i}^{N}\int [Dh_i(t)-\overline{Dh}(t)]
        [ y_i^2(t)- \overline{y^2}(t) ]dt}
        {\frac{1}{N} \sum_{i}^{N} \int y_i^2(t)dt}

    Whose structure is related to the covariation between the deformation
    functions :math:`Dh_i(t)` and the squared registered functions
    :math:`y_i^2(t)`. When these two sets of functions are independents
    :math:`C_R=1`, as in the case of shift registration.

    The measures of amplitude and phase mean square error are

    .. math::
        \text{MSE}_{amp} =  C_R \frac{1}{N}
        \sum_{i=1}^{N} \int \left [ y_i(t) - \overline{y}(t) \right ]^2 dt

    .. math::
        \text{MSE}_{phase}=
        \int \left [C_R \overline{y}^2(t) - \overline{x}^2(t) \right]dt

    It can be shown that

    .. math::
        \text{MSE}_{total} = \text{MSE}_{amp} + \text{MSE}_{phase}

    The squared multiple correlation index of the proportion of the total
    variation due to phase is defined as:

    .. math::
        R^2 = \frac{\text{MSE}_{phase}}{\text{MSE}_{total}}

    See [KR08-3]_ for a detailed explanation.


    Args:
        original_fdata (:class:`FData`): Unregistered functions.
        regfd (:class:`FData`): Registered functions.
        warping_function (:class:`FData`): Warping functions.
        eval_points: (array_like, optional): Set of points where the
            functions are evaluated to obtain a discrete representation.


    Returns:
        :class:`collections.namedtuple`: Tuple with amplitude mean square error
        :math:`\text{MSE}_{amp}`, phase mean square error
        :math:`\text{MSE}_{phase}`, squared correlation index :math:`R^2`
        and constant :math:`C_R`.

    Raises:
        ValueError: If the curves do not have the same number of samples.

    References:
        ..  [KR08-3] Kneip, Alois & Ramsay, James. (2008).  Quantifying
            amplitude and phase variation. In *Combining Registration and
            Fitting for Functional Models* (pp. 14-15). Journal of the American
            Statistical Association.
        ..  [RGS09-8-5] Ramsay J.O., Giles Hooker & Spencer Graves (2009). In
            *Functional Data Analysis with R and Matlab* (pp. 125-126).
            Springer.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import (landmark_registration_warping,
        ...                               mse_decomposition)


        We will create and register data.

        >>> fd = make_multimodal_samples(n_samples=3, random_state=1)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        >>> landmarks = landmarks.squeeze()
        >>> warping = landmark_registration_warping(fd, landmarks)
        >>> fd_registered = fd.compose(warping)
        >>> mse_amp, mse_pha, rsq, cr = mse_decomposition(fd, fd_registered, warping)

        Mean square error produced by the amplitude variation.

        >>> f'{mse_amp:.6f}'
        '0.000987'

        In this example we can observe that the main part of the mean square
        error is due to the phase variation.

        >>> f'{mse_pha:.6f}'
        '0.115769'

        Nearly 99% of the variation is due to phase.

        >>> f'{rsq:.6f}'
        '0.991549'

    """

    if registered_fdata.ndim_domain != 1 or registered_fdata.ndim_image != 1:
        raise NotImplementedError

    if original_fdata.nsamples != registered_fdata.nsamples:
        raise ValueError(f"the registered and unregistered curves must have "
                         f"the same number of samples "
                         f"({registered_fdata.nsamples})!= "
                         f"({original_fdata.nsamples})")

    if warping_function is not None and (warping_function.nsamples
                                         != original_fdata.nsamples):
        raise ValueError(f"the registered curves and the warping functions must"
                         f" have the same number of samples "
                         f"({registered_fdata.nsamples})"
                         f"!=({warping_function.nsamples})")

    # Creates the mesh to discretize the functions
    if eval_points is None:
        try:
            eval_points = registered_fdata.sample_points[0]

        except AttributeError:
            nfine = max(registered_fdata.basis.nbasis * 10 + 1, 201)
            domain_range = registered_fdata.domain_range[0]
            eval_points = numpy.linspace(*domain_range, nfine)
    else:
        eval_points = numpy.asarray(eval_points)

    x_fine = original_fdata.evaluate(eval_points, keepdims=False)
    y_fine = registered_fdata.evaluate(eval_points, keepdims=False)
    mu_fine = x_fine.mean(axis=0)  # Mean unregistered function
    eta_fine = y_fine.mean(axis=0)  # Mean registered function
    mu_fine_sq = numpy.square(mu_fine)
    eta_fine_sq = numpy.square(eta_fine)

    # Total mean square error of the original funtions
    # mse_total = scipy.integrate.simps(
    #    numpy.mean(numpy.square(x_fine - mu_fine), axis=0),
    #    eval_points)

    cr = 1.  # Constant related to the covariation between the deformation
    # functions and y^2

    # If the warping functions are not provided, are suppose to be independent
    if warping_function is not None:
        # Derivates warping functions
        dh_fine = warping_function.evaluate(eval_points, derivative=1,
                                            keepdims=False)
        dh_fine_mean = dh_fine.mean(axis=0)
        dh_fine_center = dh_fine - dh_fine_mean

        y_fine_sq = numpy.square(y_fine)  # y^2
        y_fine_sq_center = numpy.subtract(
            y_fine_sq, eta_fine_sq)  # y^2 - E[y^2]

        covariate = numpy.inner(dh_fine_center.T, y_fine_sq_center.T)
        covariate = covariate.mean(axis=0)
        cr += numpy.divide(scipy.integrate.simps(covariate,
                                                 eval_points),
                           scipy.integrate.simps(eta_fine_sq,
                                                 eval_points))

    # mse due to phase variation
    mse_pha = scipy.integrate.simps(cr*eta_fine_sq - mu_fine_sq, eval_points)

    # mse due to amplitude variation
    # mse_amp = mse_total - mse_pha
    y_fine_center = numpy.subtract(y_fine, eta_fine)
    y_fine_center_sq = numpy.square(y_fine_center, out=y_fine_center)
    y_fine_center_sq_mean = y_fine_center_sq.mean(axis=0)

    mse_amp = scipy.integrate.simps(y_fine_center_sq_mean, eval_points)

    # Total mean square error of the original funtions
    mse_total = mse_pha + mse_amp

    # squared correlation measure of proportion of phase variation
    rsq = mse_pha / (mse_total)

    mse_decomp = collections.namedtuple('mse_decomposition',
                                        'mse_amp mse_pha rsq cr')

    return mse_decomp(mse_amp, mse_pha, rsq, cr)


def invert_warping(fdatagrid, *, eval_points=None):
    r"""Compute the inverse of a diffeomorphism.

    Let :math:`\gamma : [a,b] \\rightarrow [a,b]` be a function strictly
    increasing, calculates the corresponding inverse
    :math:`\gamma^{-1} : [a,b] \\rightarrow [a,b]` such that
    :math:`\gamma^{-1} \circ \gamma = \gamma \circ \gamma^{-1} = \gamma_{id}`.

    Uses a PCHIP interpolator to compute approximately the inverse.

    Args:
        fdatagrid (:class:`FDataGrid`): Functions to be inverted.
        eval_points: (array_like, optional): Set of points where the
            functions are interpolated to obtain the inverse, by default uses
            the sample points of the fdatagrid.

    Returns:
        :class:`FDataGrid`: Inverse of the original functions.

    Raises:
        ValueError: If the functions are not strictly increasing or are
            multidimensional.

    Examples:

        >>> import numpy as np
        >>> from skfda import FDataGrid
        >>> from skfda.preprocessing.registration import invert_warping

        We will construct the warping :math:`\gamma : [0,1] \\rightarrow [0,1]`
        wich maps t to t^3.

        >>> t = np.linspace(0, 1)
        >>> gamma = FDataGrid(t**3, t)
        >>> gamma
        FDataGrid(...)

        We will compute the inverse.

        >>> inverse = invert_warping(gamma)
        >>> inverse
        FDataGrid(...)

        The result of the composition should be approximately the identity
        function .

        >>> identity = gamma.compose(inverse)
        >>> identity([0, 0.25, 0.5, 0.75, 1]).round(3)
        array([[ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]])

    """

    if fdatagrid.ndim_image != 1 or fdatagrid.ndim_domain != 1:
        raise ValueError("Multidimensional object not supported.")

    if eval_points is None:
        eval_points = fdatagrid.sample_points[0]

    y = fdatagrid(eval_points, keepdims=False)

    data_matrix = numpy.empty((fdatagrid.nsamples, len(eval_points)))

    for i in range(fdatagrid.nsamples):
        data_matrix[i] = PchipInterpolator(y[i], eval_points)(eval_points)

    return fdatagrid.copy(data_matrix=data_matrix, sample_points=eval_points)


def _normalize_scale(t, a=0, b=1):
    """Perfoms an afine translation to normalize an interval.

    Args:
        t (numpy.ndarray): Array of dim 1 or 2 with at least 2 values.
        a (float): Starting point of the new interval. Defaults 0.
        b (float): Stopping point of the new interval. Defaults 1.

    Returns:
        (numpy.ndarray): Array with the transformed interval.
    """

    t = t.T  # Broadcast to normalize multiple arrays
    t1 = (t - t[0]).astype(float)  # Translation to [0, t[-1] - t[0]]
    t1 *= (b - a) / (t[-1] - t[0])  # Scale to [0, b-a]
    t1 += a  # Translation to [a, b]
    t1[0] = a  # Fix possible round errors
    t1[-1] = b

    return t1.T


def normalize_warping(warping, domain_range=None):
    """Rescale a warping to normalize their domain.

    Given a set of warpings :math:`\\gamma_i:[a,b] \\rightarrow [a,b]` it is
    used an affine traslation to change the domain of the transformation to
    other domain, :math:`\\hat \\gamma_i:[\\hat a,\\hat b] \\rightarrow
    [\\hat a, \\hat b]`.

    Args:
        warping (:class:`FDatagrid`): Set of warpings to rescale.
        domain_range (tuple, optional): New domain range of the warping. By
            default it is used the same domain range.
    Return:
        (:class:`FDataGrid`): FDataGrid with the warpings normalized.

    """

    if domain_range is None:
        domain_range = warping.domain_range[0]

    data_matrix = _normalize_scale(warping.data_matrix[..., 0], *domain_range)
    sample_points = _normalize_scale(warping.sample_points[0], *domain_range)

    return warping.copy(data_matrix=data_matrix, sample_points=sample_points,
                        domain_range=domain_range)

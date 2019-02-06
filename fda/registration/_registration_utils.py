"""Registration of functional data module.

This module contains routines related to the registration procedure.
"""
import collections

import numpy
import scipy.integrate


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

        >>> from fda.datasets import make_multimodal_landmarks
        >>> from fda.datasets import make_multimodal_samples
        >>> from fda.registration import (landmark_registration_warping,
        ...                               mse_decomposition)


        We will create and register data.

        >>> fd = make_multimodal_samples(n_samples=3, random_state=1)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        >>> landmarks = landmarks.squeeze()
        >>> warping = landmark_registration_warping(fd, landmarks)
        >>> fd_registered = fd.compose(warping)
        >>> mse_amp, mse_pha, rsq, cr = mse_decomposition(fd, fd_registered, warping)

        Mean square error produced by the amplitude variation.

        >>> round(mse_amp, 6)
        0.000987

        In this example we can observe that the main part of the mean square
        error is due to the phase variation.

        >>> round(mse_pha, 6)
        0.115769

        Nearly 99% of the variation is due to phase.

        >>> round(rsq, 6)
        0.991549



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
    mu_fine = x_fine.mean(axis=0) # Mean unregistered function
    eta_fine = y_fine.mean(axis=0) # Mean registered function
    mu_fine_sq = numpy.square(mu_fine)
    eta_fine_sq = numpy.square(eta_fine)


    # Total mean square error of the original funtions
    #mse_total = scipy.integrate.simps(
    #    numpy.mean(numpy.square(x_fine - mu_fine), axis=0),
    #    eval_points)

    cr = 1. # Constant related to the covariation between the deformation
            # functions and y^2

    # If the warping functions are not provided, are suppose to be independent
    if warping_function is not None:
        # Derivates warping functions
        dh_fine = warping_function.evaluate(eval_points, derivative=1,
                                            keepdims=False)
        dh_fine_mean = dh_fine.mean(axis=0)
        dh_fine_center = dh_fine - dh_fine_mean

        y_fine_sq = numpy.square(y_fine) # y^2
        y_fine_sq_center = numpy.subtract(y_fine_sq, eta_fine_sq) # y^2 - E[y^2]

        covariate = numpy.inner(dh_fine_center.T, y_fine_sq_center.T)
        covariate = covariate.mean(axis=0)
        cr += numpy.divide(scipy.integrate.simps(covariate,
                                                 eval_points),
                           scipy.integrate.simps(eta_fine_sq,
                                                 eval_points))


    # mse due to phase variation
    mse_pha = scipy.integrate.simps(cr*eta_fine_sq - mu_fine_sq ,
                                    eval_points)

    # mse due to amplitude variation
    #mse_amp = mse_total - mse_pha

    y_fine_center = numpy.subtract(y_fine, eta_fine)
    y_fine_center_sq = numpy.square(y_fine_center, out =y_fine_center)
    y_fine_center_sq_mean = y_fine_center_sq.mean(axis=0)

    mse_amp = scipy.integrate.simps(y_fine_center_sq_mean, eval_points)

    # Total mean square error of the original funtions
    mse_total = mse_pha + mse_amp

    # squared correlation measure of proportion of phase variation
    rsq = mse_pha / (mse_total)

    mse_decomp = collections.namedtuple('mse_decomposition',
                                        'mse_amp mse_pha rsq cr')

    return mse_decomp(mse_amp, mse_pha, rsq, cr)

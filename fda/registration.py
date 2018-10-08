"""Registration of functional data module.

This module contains the methods to perform the registration of
functional data and related routines, in basis form as well in discretized form.

"""
from enum import Enum

import numpy
import scipy.integrate


class Extrapolation(Enum):
    r"""Enum with extrapolation types. Defines the extrapolation mode for
        elements outside the domain range.
    """
    default = "default" #:Uses the default method of the functional data object.
    extrapolation = "extrapolation" #: The values are extrapolated by evaluate.
    periodic = "periodic" #: Extends the domain range periodically.
    const = "const" #: Uses the boundary value.
    slice = "slice" #: Avoids extrapolation restricting the domain.

def mse_decomposition(fd, fdreg, h=None, tfine=None):
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
        C_R \int \left [\overline{y}^2(t) - \overline{x}^2(t) \right]dt

    It can be shown that

    .. math::
        \text{MSE}_{total} = \text{MSE}_{amp} + \text{MSE}_{phase}

    The squared multiple correlation index of the proportion of the total
    variation due to phase is defined as:

    .. math::
        R^2 = \frac{\text{MSE}_{phase}}{\text{MSE}_{total}}

    See [KR08-3]_ for a detailed explanation.


    Args:
        fd (:obj:`FDataBasis` or :obj:`FDataGrid`): Unregistered functions.
        regfd (:obj:`FDataBasis` or :obj:`FDataGrid`): Registered functions.
        h (:obj:`FDataBasis` or :obj:`FDataGrid`, optional): Warping functions.
        tfine: (array_like, optional): Set of points where the functions are
            evaluated to obtain a discrete representation.


    Returns:
        Tuple: Tuple with amplitude mean square error :math:`\text{MSE}_{amp}`,
        phase mean square error :math:`\text{MSE}_{phase}`, squared correlation
        index :math:`R^2` and constant :math:`C_R`.

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
    """

    if fd.nsamples != fdreg.nsamples:
        raise ValueError("the registered and unregistered curves must have "
                         "the same number of samples ({})!=({})"
                         .format(fdreg.nsamples, fd.nsamples))

    if h is not None and h.nsamples != fd.nsamples:
        raise ValueError("the registered curves and the warping functions must "
                         "have the same number of samples ({})!=({})"
                         .format(fdreg.nsamples, h.nsamples))

    # Creates the mesh to discretize the functions
    if tfine is None:
        nfine = max(fdreg.basis.nbasis * 10 + 1, 201)
        tfine = numpy.linspace(fdreg.domain_range[0],
                               fdreg.domain_range[1], nfine)
    else:
        tfine = numpy.asarray(tfine)

    x_fine = fd.evaluate(tfine) # Unregistered function
    y_fine = fdreg.evaluate(tfine) # Registered function
    mu_fine = x_fine.mean(axis=0) # Mean unregistered function
    eta_fine = y_fine.mean(axis=0) # Mean registered function
    mu_fine_sq = numpy.square(mu_fine)
    eta_fine_sq = numpy.square(eta_fine)


    # Total mean square error of the original funtions
    mse_total = scipy.integrate.simps(
        numpy.mean(numpy.square(x_fine - mu_fine), axis=0),tfine)

    cr = 1. # Constant related to the covariation between the deformation
            # functions and y^2

    # If the warping functions are not provided, are suppose to be independent
    if h is not None:

        dh_fine = h.evaluate(tfine, derivative=1) # Derivate of warping functions
        dh_fine_mean = dh_fine.mean(axis=0)
        dh_fine_center = dh_fine - dh_fine_mean

        y_fine_sq = numpy.square(y_fine) # y^2
        y_fine_sq_center = numpy.subtract(y_fine_sq, eta_fine_sq) # y^2 - E[y^2]

        covariate = numpy.inner(dh_fine_center.T, y_fine_sq_center.T)
        covariate = covariate.mean(axis=0)
        cr += numpy.divide(scipy.integrate.simps(covariate, tfine),
                           scipy.integrate.simps(eta_fine_sq, tfine))


    # mse due to phase variation
    mse_pha = scipy.integrate.simps(cr*eta_fine_sq - mu_fine_sq , tfine)

    # mse due to amplitude variation
    mse_amp = mse_total - mse_pha

    # squared correlation measure of proportion of phase variation
    rsq = mse_pha / (mse_total)

    return mse_amp, mse_pha, rsq, cr



def shift_registration(fd, maxiter=5, tol=1e-2, ext="default", alpha=1,
                       initial=None, tfine=None, shifts_array=False, **kwargs):
    r"""Perform a shift registration of the curves.

        Realizes a registration of the curves, using shift aligment, as is
        defined in [RS05-7-2]_. Calculates :math:`\delta_{i}` for each sample
        such that :math:`x_i(t + \delta_{i})` minimizes the least squares
        criterion:

        .. math::
            \text{REGSSE} = \sum_{i=1}^{N} \int_{\mathcal{T}}
            [x_i(t + \delta_i) - \hat\mu(t)]^2 ds

        Estimates the shift parameter :math:`\delta_i` iteratively by
        using a modified Newton-Raphson algorithm, updating the mean
        in each iteration, as is described in detail in [RS05-7-9-1]_.

    Args:
        fd (:obj:`FDataBasis` or :obj:`FDataGrid`): Functional data object.
        maxiter (int, optional): Maximun number of iterations.
            Defaults to 5.
        tol (float, optional): Tolerance allowable. The process will stop if
            :math:`\max_{i}|\delta_{i}^{(\nu)}-\delta_{i}^{(\nu-1)}|<tol`.
            Default sets to 1e-2.
        ext (str or Extrapolation, optional): Controls the extrapolation
            mode for elements outside the domain range.

            * If ext='default' or Extrapolation.default default
                method defined in the fd object is used.
            * If ext='extrapolation' or Extrapolation.extrapolation uses
                the extrapolated values by the basis.
            * If ext='periodic' or Extrapolation.periodic extends the
                domain range periodically.
            * If ext='const' or Extrapolation.const uses the boundary
                value
            * If ext='slice' or Extrapolation.slice avoids extrapolation
                restricting the domain.
        alpha (int or float, optional): Parameter to adjust the rate of
            convergence in the Newton-Raphson algorithm, see [RS05-7-9-1]_.
            Defaults to 1.
        initial (array_like, optional): Initial estimation of shifts.
            Default uses a list of zeros for the initial shifts.
        tfine (array_like, optional): Set of points where the
            functions are evaluated to obtain the discrete
            representation of the object to integrate. If an None is
            passed it calls numpy.linspace with bounds equal to the ones defined
            in fd.domain_range and the number of points the maximum
            between 201 and 10 times the number of basis plus 1.
        shifts_array (bool, optional): If True returns an array with the
            shifts instead of a :obj:`FDataBasis` with the registered
            curves. Default sets to False.
        **kwargs: Keyword arguments to be passed to :meth:`from_data`.

    Returns:
        :obj:`FDataBasis` or :obj:`ndarray`: A :obj:`FDataBasis` object with
        the curves registered or if shifts_array is True a :obj:`ndarray`
        with the shifts.

    Raises:
        ValueError: If the initial array has different length than the
            number of samples.

    References:
        ..  [RS05-7-2] Ramsay, J., Silverman, B. W. (2005). Shift
            registration. In *Functional Data Analysis* (pp. 129-132).
            Springer.
        ..  [RS05-7-9-1] Ramsay, J., Silverman, B. W. (2005). Shift
            registration by the Newton-Raphson algorithm. In *Functional
            Data Analysis* (pp. 142-144). Springer.
    """

    # Initial estimation of the shifts
    if initial is None:
        delta = numpy.zeros(fd.nsamples)

    elif len(initial) != fd.nsamples:
        raise ValueError("the initial shift ({}) must have the same length "
                         "than the number of samples ({})"
                         .format(len(initial), fd.nsamples))
    else:
        delta = numpy.asarray(initial)

    # Fine equispaced mesh to evaluate the samples
    if tfine is None:
        nfine = max(fd.nbasis*10+1, 201)
        tfine = numpy.linspace(fd.basis.domain_range[0],
                               fd.basis.domain_range[1],
                               nfine)
    else:
        nfine = len(tfine)
        tfine = numpy.asarray(tfine)


    extrapolation = Extrapolation(ext)

    if extrapolation is Extrapolation.default:
        extrapolation = fd.default_extrapolation

    # Auxiliar arrays to avoid multiple memory allocations
    delta_aux = numpy.empty(fd.nsamples)
    tfine_aux = numpy.empty(nfine)

    # Computes the derivate of originals curves in the mesh points
    D1x = fd.evaluate(tfine, 1)

    # Second term of the second derivate estimation of REGSSE. The
    # first term has been dropped to improve convergence (see references)
    d2_regsse = scipy.integrate.trapz(numpy.square(D1x), tfine, axis=1)

    max_diff = tol + 1
    iter = 0

    # Auxiliar array if the domain will be restricted
    if extrapolation is Extrapolation.slice:
        D1x_tmp = D1x
        tfine_tmp = tfine
        tfine_aux_tmp = tfine_aux
        domain = numpy.empty(nfine, dtype=numpy.dtype(bool))

    # Newton-Rhapson iteration
    while max_diff > tol and iter < maxiter:

        # Updates the limits for non periodic functions ignoring the ends
        if extrapolation is Extrapolation.slice:
            # Calculates the new limits
            a = fd.domain_range[0] - min(numpy.min(delta), 0)
            b = fd.domain_range[1] - max(numpy.max(delta), 0)

            # New interval is (a,b)
            numpy.logical_and(tfine_tmp >= a, tfine_tmp <= b, out=domain)
            tfine = tfine_tmp[domain]
            tfine_aux = tfine_aux_tmp[domain]
            D1x = D1x_tmp[:, domain]
            # Reescale the second derivate could be other approach
            # d2_regsse =
            #     d2_regsse_original * ( 1 + (a - b) / (domain[1] - domain[0]))
            d2_regsse = scipy.integrate.trapz(numpy.square(D1x), tfine, axis=1)

        # Computes the new values shifted
        x = fd.evaluate_shifted(tfine, delta, ext=extrapolation)
        x.mean(axis=0, out=tfine_aux)

        # Calculates x - mean
        numpy.subtract(x, tfine_aux, out=x)

        d1_regsse = scipy.integrate.trapz(numpy.multiply(x, D1x, out=x),
                                          tfine, axis=1)
        # Updates the shifts by the Newton-Rhapson iteration
        # delta = delta - alpha * d1_regsse / d2_regsse
        numpy.divide(d1_regsse, d2_regsse, out=delta_aux)
        numpy.multiply(delta_aux, alpha, out=delta_aux)
        numpy.subtract(delta, delta_aux, out=delta)

        # Updates convergence criterions
        max_diff = numpy.abs(delta_aux, out=delta_aux).max()
        iter += 1

    # If shifts_array is True returns the delta array
    if shifts_array: return delta

    # Computes the values with the final shift to construct the FDataBasis
    return fd.shift(delta, ext=ext, tfine=tfine, **kwargs)


def landmark_shift(fd, landmarks, location=None, ext='default', tfine=None,
                   shifts_array=False, **kwargs):
    r"""Perform a shift registration of the curves to align the landmarks at
        the same mark time.

        Args:
            fd (:obj:`FDataBasis` or :obj:`FDataGrid`): Functional data object.
            landmarks (array_like): List with the landmarks of the samples.
            location (numeric or callable, optional): Defines where
                the landmarks will be alligned. If a numeric value is passed the
                landmarks will be alligned to it. In case of a callable is
                passed the location will be the result of the the call, the
                function should be accept as an unique parameter a numpy array
                with the list of landmarks.
                By default it will be used as location :math:`\frac{1}{2}(max(
                \text{landmarks})+ min(\text{landmarks}))` wich minimizes the
                max shift.
            ext (str or Extrapolation, optional): Controls the extrapolation
                mode for elements outside the domain range.

                * If ext='default' or Extrapolation.default default
                    method defined in the fd object is used.
                * If ext='extrapolation' or Extrapolation.extrapolation uses
                    the extrapolated values by the basis.
                * If ext='periodic' or Extrapolation.periodic extends the
                    domain range periodically.
                * If ext='const' or Extrapolation.const uses the boundary
                    value
                * If ext='slice' or Extrapolation.slice avoids extrapolation
                    restricting the domain.
                The default value is 'default'.
            tfine (array_like, optional): Set of points where the
                functions are evaluated to obtain the discrete
                representation of the object to integrate. If an empty list is
                passed it calls numpy.linspace with bounds equal to the ones defined
                in fd.domain_range and the number of points the maximum
                between 201 and 10 times the number of basis plus 1.
            shifts_array (bool, optional): If True returns an array with the
                shifts instead of a :obj:`FDataBasis` with the registered
                curves. Default sets to False.
            **kwargs: Keyword arguments to be passed to :meth:`from_data`.
    """

    if len(landmarks) != fd.nsamples:
        raise ValueError("landmark list ({}) must have the same length "
                         "than the number of samples ({})"
                         .format(len(landmarks), fd.nsamples))

    landmarks = numpy.asarray(landmarks)

    # Parses location
    if location is None:
        p = (numpy.max(landmarks) + numpy.min(landmarks)) / 2.
    elif callable(location):
        p = location(landmarks)
    else:
        try:
            p = float(location)
        except:
            raise ValueError("Invalid location, must be None, a callable or a "
                             "number in the domain")

    shifts = landmarks - p

    if shifts_array:
        return shifts

    return fd.shift(shifts, ext=ext, tfine=tfine, **kwargs)

"""Shift Registration of functional data module.

This module contains methods to perform the registration of
functional data using shifts, in basis as well in discretized form.
"""

import numpy
import scipy.integrate

__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"


def shift_registration_deltas(fd, *, maxiter=5, tol=1e-2, restrict_domain=False,
                              extrapolation=None, step_size=1, initial=None,
                              eval_points=None):
    r"""Return the lists of shifts used in the shift registration procedure.

        Realizes a registration of the curves, using shift aligment, as is
        defined in [RS05-7-2-1]_. Calculates :math:`\delta_{i}` for each sample
        such that :math:`x_i(t + \delta_{i})` minimizes the least squares
        criterion:

        .. math::
            \text{REGSSE} = \sum_{i=1}^{N} \int_{\mathcal{T}}
            [x_i(t + \delta_i) - \hat\mu(t)]^2 ds

        Estimates the shift parameter :math:`\delta_i` iteratively by
        using a modified Newton-Raphson algorithm, updating the mean
        in each iteration, as is described in detail in [RS05-7-9-1-1]_.

        Method only implemented for Funtional objects with domain and image
        dimension equal to 1.

    Args:
        fd (:class:`FData`): Functional data object to be registered.
        maxiter (int, optional): Maximun number of iterations.
            Defaults to 5.
        tol (float, optional): Tolerance allowable. The process will stop if
            :math:`\max_{i}|\delta_{i}^{(\nu)}-\delta_{i}^{(\nu-1)}|<tol`.
            Default sets to 1e-2.
        restrict_domain (bool, optional): If True restricts the domain to
            avoid evaluate points outside the domain using extrapolation.
            Defaults uses extrapolation.
        extrapolation (str or :class:`Extrapolation`, optional): Controls the
            extrapolation mode for elements outside the domain range.
            By default uses the method defined in fd. See :module:
            `extrapolation` to obtain more information.
        step_size (int or float, optional): Parameter to adjust the rate of
            convergence in the Newton-Raphson algorithm, see [RS05-7-9-1-1]_.
            Defaults to 1.
        initial (array_like, optional): Initial estimation of shifts.
            Default uses a list of zeros for the initial shifts.
        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain the discrete
            representation of the object to integrate. If None is
            passed it calls numpy.linspace in FDataBasis and uses the
            `sample_points` in FDataGrids.

    Returns:
        :class:`numpy.ndarray`: list with the shifts.

    Raises:
        ValueError: If the initial array has different length than the
            number of samples.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.basis import Fourier
        >>> from skfda.preprocessing.registration import shift_registration_deltas
        >>> fd = make_sinusoidal_process(n_samples=2, error_std=0, random_state=1)

        Registration of data in discretized form:

        >>> shift_registration_deltas(fd).round(3)
        array([-0.022,  0.03 ])

        Registration of data in basis form:

        >>> fd = fd.to_basis(Fourier())
        >>> shift_registration_deltas(fd).round(3)
        array([-0.022,  0.03 ])


    References:
        ..  [RS05-7-2-1] Ramsay, J., Silverman, B. W. (2005). Shift
            registration. In *Functional Data Analysis* (pp. 129-132).
            Springer.
        ..  [RS05-7-9-1-1] Ramsay, J., Silverman, B. W. (2005). Shift
            registration by the Newton-Raphson algorithm. In *Functional
            Data Analysis* (pp. 142-144). Springer.
    """

    # Initial estimation of the shifts

    if fd.ndim_image > 1 or fd.ndim_domain > 1:
        raise NotImplementedError("Method for unidimensional data.")

    domain_range = fd.domain_range[0]

    if initial is None:
        delta = numpy.zeros(fd.nsamples)

    elif len(initial) != fd.nsamples:
        raise ValueError(f"the initial shift ({len(initial)}) must have the "
                         f"same length than the number of samples "
                         f"({fd.nsamples})")
    else:
        delta = numpy.asarray(initial)

    # Fine equispaced mesh to evaluate the samples
    if eval_points is None:

        try:
            eval_points = fd.sample_points[0]
            nfine = len(eval_points)
        except AttributeError:
            nfine = max(fd.nbasis*10+1, 201)
            eval_points = numpy.linspace(*domain_range, nfine)

    else:
        nfine = len(eval_points)
        eval_points = numpy.asarray(eval_points)

    # Auxiliar arrays to avoid multiple memory allocations
    delta_aux = numpy.empty(fd.nsamples)
    tfine_aux = numpy.empty(nfine)

    # Computes the derivate of originals curves in the mesh points
    D1x = fd.evaluate(eval_points, derivative=1, keepdims=False)

    # Second term of the second derivate estimation of REGSSE. The
    # first term has been dropped to improve convergence (see references)
    d2_regsse = scipy.integrate.trapz(numpy.square(D1x), eval_points,
                                      axis=1)

    max_diff = tol + 1
    iter = 0

    # Auxiliar array if the domain will be restricted
    if restrict_domain:
        D1x_tmp = D1x
        tfine_tmp = eval_points
        tfine_aux_tmp = tfine_aux
        domain = numpy.empty(nfine, dtype=numpy.dtype(bool))

    ones = numpy.ones(fd.nsamples)
    eval_points_rep = numpy.outer(ones, eval_points)

    # Newton-Rhapson iteration
    while max_diff > tol and iter < maxiter:

        # Updates the limits for non periodic functions ignoring the ends
        if restrict_domain:
            # Calculates the new limits
            a = domain_range[0] - min(numpy.min(delta), 0)
            b = domain_range[1] - max(numpy.max(delta), 0)

            # New interval is (a,b)
            numpy.logical_and(tfine_tmp >= a, tfine_tmp <= b, out=domain)
            eval_points = tfine_tmp[domain]
            tfine_aux = tfine_aux_tmp[domain]
            D1x = D1x_tmp[:, domain]
            # Reescale the second derivate could be other approach
            # d2_regsse =
            #     d2_regsse_original * ( 1 + (a - b) / (domain[1] - domain[0]))
            d2_regsse = scipy.integrate.trapz(numpy.square(D1x),
                                              eval_points, axis=1)
            eval_points_rep = numpy.outer(ones, eval_points)

        # Computes the new values shifted
        x = fd.evaluate(eval_points_rep + numpy.atleast_2d(delta).T,
                        aligned_evaluation=False,
                        extrapolation=extrapolation,
                        keepdims=False)

        x.mean(axis=0, out=tfine_aux)

        # Calculates x - mean
        numpy.subtract(x, tfine_aux, out=x)

        d1_regsse = scipy.integrate.trapz(numpy.multiply(x, D1x, out=x),
                                          eval_points, axis=1)
        # Updates the shifts by the Newton-Rhapson iteration
        # delta = delta - step_size * d1_regsse / d2_regsse
        numpy.divide(d1_regsse, d2_regsse, out=delta_aux)
        numpy.multiply(delta_aux, step_size, out=delta_aux)
        numpy.subtract(delta, delta_aux, out=delta)

        # Updates convergence criterions
        max_diff = numpy.abs(delta_aux, out=delta_aux).max()
        iter += 1

    return delta


def shift_registration(fd, *, maxiter=5, tol=1e-2, restrict_domain=False,
                       extrapolation=None, step_size=1, initial=None,
                       eval_points=None, **kwargs):
    r"""Perform shift registration of the curves.

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
        fd (:class:`FData`): Functional data object to be registered.
        maxiter (int, optional): Maximun number of iterations.
            Defaults to 5.
        tol (float, optional): Tolerance allowable. The process will stop if
            :math:`\max_{i}|\delta_{i}^{(\nu)}-\delta_{i}^{(\nu-1)}|<tol`.
            Default sets to 1e-2.
        restrict_domain (bool, optional): If True restricts the domain to
            avoid evaluate points outside the domain using extrapolation.
            Defaults uses extrapolation.
        extrapolation (str or :class:`Extrapolation`, optional): Controls the
            extrapolation mode for elements outside the domain range.
            By default uses the method defined in fd. See :module:
            `extrapolation` to obtain more information.
        step_size (int or float, optional): Parameter to adjust the rate of
            convergence in the Newton-Raphson algorithm, see [RS05-7-9-1]_.
            Defaults to 1.
        initial (array_like, optional): Initial estimation of shifts.
            Default uses a list of zeros for the initial shifts.
        eval_points (array_like, optional): Set of points where the
            functions are evaluated to obtain the discrete
            representation of the object to integrate. If None is
            passed it calls numpy.linspace in FDataBasis and uses the
            `sample_points` in FDataGrids.
        **kwargs: Keyword arguments to be passed to :func:`shift`.

    Returns:
        :class:`FData` A :class:`FData` object with the curves registered.

    Raises:
        ValueError: If the initial array has different length than the
            number of samples.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.basis import Fourier
        >>> from skfda.preprocessing.registration import shift_registration
        >>> fd = make_sinusoidal_process(n_samples=2, error_std=0, random_state=1)

        Registration of data in discretized form:

        >>> shift_registration(fd)
        FDataGrid(...)

        Registration of data in basis form:

        >>> fd = fd.to_basis(Fourier())
        >>> shift_registration(fd)
        FDataBasis(...)

    References:
        ..  [RS05-7-2] Ramsay, J., Silverman, B. W. (2005). Shift
            registration. In *Functional Data Analysis* (pp. 129-132).
            Springer.
        ..  [RS05-7-9-1] Ramsay, J., Silverman, B. W. (2005). Shift
            registration by the Newton-Raphson algorithm. In *Functional
            Data Analysis* (pp. 142-144). Springer.
    """

    delta = shift_registration_deltas(fd, maxiter=maxiter, tol=tol,
                                      restrict_domain=restrict_domain,
                                      extrapolation=extrapolation,
                                      step_size=step_size, initial=initial,
                                      eval_points=eval_points)

    # Computes the values with the final shift to construct the FDataBasis
    return fd.shift(delta, restrict_domain=restrict_domain,
                    extrapolation=extrapolation,
                    eval_points=eval_points, **kwargs)

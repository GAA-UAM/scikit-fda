"""Landmark Registration of functional data module.

This module contains methods to perform the landmark registration.
"""

import numpy

from ... import FDataGrid
from ...representation.interpolation import SplineInterpolator

__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"


def landmark_shift_deltas(fd, landmarks, location=None):
    r"""Returns the corresponding shifts to align the landmarks of the curves.

        Let :math:`t^*` the time where the landmarks of the curves will be
        aligned, and :math:`t_i` the location of the landmarks for each curve.
        The function will calculate the corresponding :math:`\delta_i` shuch
        that :math:`t_i = t^* + \delta_i`.

        This procedure will work independent of the dimension of the domain
        and the image.

    Args:
        fd (:class:`FData`): Functional data object.
        landmarks (array_like): List with the landmarks of the samples.
        location (numeric or callable, optional): Defines where
            the landmarks will be alligned. If a numer or list is passed the
            landmarks will be alligned to it. In case of a callable is
            passed the location will be the result of the the call, the
            function should be accept as an unique parameter a numpy array
            with the list of landmarks.
            By default it will be used as location :math:`\frac{1}{2}(max(
            \text{landmarks})+ min(\text{landmarks}))` wich minimizes the
            max shift.

    Returns:
        :class:`numpy.ndarray`: Array containing the corresponding shifts.

    Raises:
        ValueError: If the list of landmarks does not match with the number of
            samples.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import landmark_shift_deltas

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, random_state=1)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        >>> landmarks = landmarks.squeeze()

        The function will return the corresponding shifts

        >>> shifts = landmark_shift_deltas(fd, landmarks)
        >>> shifts.round(3)
        array([ 0.25 , -0.25 , -0.231])

        The registered samples can be obtained with a shift

        >>> fd.shift(shifts)
        FDataGrid(...)

    """

    if len(landmarks) != fd.nsamples:
        raise ValueError(f"landmark list ({len(landmarks)}) must have the same "
                         f"length than the number of samples ({fd.nsamples})")

    landmarks = numpy.atleast_1d(landmarks)

    # Parses location
    if location is None:
        p = (numpy.max(landmarks, axis=0) + numpy.min(landmarks, axis=0)) / 2.
    elif callable(location):
        p = location(landmarks)
    else:
        try:
            p = numpy.atleast_1d(location)
        except:
            raise ValueError("Invalid location, must be None, a callable or a "
                             "number in the domain")

    shifts = landmarks - p

    return shifts


def landmark_shift(fd, landmarks, location=None, *, restrict_domain=False,
                   extrapolation=None, eval_points=None, **kwargs):
    r"""Perform a shift of the curves to align the landmarks.

        Let :math:`t^*` the time where the landmarks of the curves will be
        aligned, :math:`t_i` the location of the landmarks for each curve
        and :math:`\delta_i= t_i - t^*`.

        The registered samples will have their feature aligned.

        .. math::
            x_i^*(t^*)=x_i(t^* + \delta_i)=x_i(t_i)

    Args:
        fd (:class:`FData`): Functional data object.
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
        restrict_domain (bool, optional): If True restricts the domain to
            avoid evaluate points outside the domain using extrapolation.
            Defaults uses extrapolation.
        extrapolation (str or Extrapolation, optional): Controls the
            extrapolation mode for elements outside the domain range.
            By default uses the method defined in fd. See extrapolation to
            more information.
        eval_points (array_like, optional): Set of points where
            the functions are evaluated in :func:`shift`.
        **kwargs: Keyword arguments to be passed to :func:`shift`.

    Returns:
        :class:`FData`: Functional data object with the registered samples.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import landmark_shift

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, random_state=1)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        >>> landmarks = landmarks.squeeze()

        The function will return the sample registered

        >>> landmark_shift(fd, landmarks)
        FDataGrid(...)

    """

    shifts = landmark_shift_deltas(fd, landmarks, location=location)

    return fd.shift(shifts, restrict_domain=restrict_domain,
                    extrapolation=extrapolation,
                    eval_points=eval_points, **kwargs)


def landmark_registration_warping(fd, landmarks, *, location=None,
                                  eval_points=None):
    """Calculate the transformation used in landmark registration.

        Let :math:`t_{ij}` the time where the sample :math:`i` has the feature
        :math:`j` and :math:`t^*_j` the new time for the feature. The warping
        function will transform the new time in the old time, i.e.,
        :math:`h_i(t^*_j)=t_{ij}`.
        The registered samples can be obtained as :math:`x^*_i(t)=x_i(h_i(t))`.

        See [RS05-7-3-1]_ for a detailed explanation.

    Args:
        fd (:class:`FData`): Functional data object.
        landmarks (array_like): List containing landmarks for each samples.
        location (array_like, optional): Defines where
            the landmarks will be alligned. By default it will be used as
            location the mean of the landmarks.
        eval_points (array_like, optional): Set of points where
            the functions are evaluated to obtain a discrete
            representation of the object.
    Returns:
        :class:`FDataGrid`: FDataGrid with the warpings function needed to
        register the functional data object.

    Raises:
        ValueError: If the object to be registered has domain dimension greater
            than 1 or the list of landmarks or locations does not match with the
            number of samples.

    References:

    ..  [RS05-7-3-1] Ramsay, J., Silverman, B. W. (2005). Feature or landmark
        registration. In *Functional Data Analysis* (pp. 132-136). Springer.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import landmark_registration_warping

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
        ...                                       random_state=9)
        >>> landmarks = landmarks.squeeze()

        The function will return the corresponding warping function

        >>> warping = landmark_registration_warping(fd, landmarks)
        >>> warping
        FDataGrid(...)

        The registered function can be obtained using function composition

        >>> fd.compose(warping)
        FDataGrid(...)
    """

    if fd.ndim_domain > 1:
        raise NotImplementedError("Method only implemented for objects with"
                                  "domain dimension up to 1.")

    if len(landmarks) != fd.nsamples:
        raise ValueError("The number of list of landmarks should be equal to "
                         "the number of samples")

    landmarks = numpy.asarray(landmarks).reshape((fd.nsamples, -1))

    n_landmarks = landmarks.shape[-1]

    data_matrix = numpy.empty((fd.nsamples, n_landmarks + 2))

    data_matrix[:, 0] = fd.domain_range[0][0]
    data_matrix[:, -1] = fd.domain_range[0][1]

    data_matrix[:, 1:-1] = landmarks

    if location is None:
        sample_points = numpy.mean(data_matrix, axis=0)

    elif n_landmarks != len(location):

        raise ValueError(f"Number of landmark locations should be equal than "
                         f"the number of landmarks ({len(location)}) != "
                         f"({n_landmarks})")
    else:
        sample_points = numpy.empty(n_landmarks + 2)
        sample_points[0] = fd.domain_range[0][0]
        sample_points[-1] = fd.domain_range[0][1]
        sample_points[1:-1] = location

    interpolator = SplineInterpolator(interpolation_order=3, monotone=True)

    warping = FDataGrid(data_matrix=data_matrix,
                        sample_points=sample_points,
                        interpolator=interpolator,
                        extrapolation='bounds')

    try:
        warping_points = fd.sample_points
    except AttributeError:
        warping_points = [numpy.linspace(*domain, 201)
                          for domain in fd.domain_range]

    return warping.to_grid(warping_points)


def landmark_registration(fd, landmarks, *, location=None, eval_points=None):
    """Perform landmark registration of the curves.

        Let :math:`t_{ij}` the time where the sample :math:`i` has the feature
        :math:`j` and :math:`t^*_j` the new time for the feature. The registered
        samples will have their features aligned, i.e.,
        :math:`x^*_i(t^*_j)=x_i(t_{ij})`.

        See [RS05-7-3]_ for a detailed explanation.

    Args:
        fd (:class:`FData`): Functional data object.
        landmarks (array_like): List containing landmarks for each samples.
        location (array_like, optional): Defines where
            the landmarks will be alligned. By default it will be used as
            location the mean of the landmarks.
        eval_points (array_like, optional): Set of points where
            the functions are evaluated to obtain a discrete
            representation of the object. In case of objects with
            multidimensional domain a list axis with points of evaluation
            for each dimension.

    Returns:
        :class:`FData`: FData with the functional data object registered.

    References:

    ..  [RS05-7-3] Ramsay, J., Silverman, B. W. (2005). Feature or landmark
        registration. In *Functional Data Analysis* (pp. 132-136). Springer.

    Examples:

        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import landmark_registration
        >>> from skfda.representation.basis import BSpline

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, n_modes=2, random_state=9)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
        ...                                       random_state=9)
        >>> landmarks = landmarks.squeeze()

        The function will return the registered curves

        >>> landmark_registration(fd, landmarks)
        FDataGrid(...)

        This method will work for FDataBasis as for FDataGrids

        >>> fd = fd.to_basis(BSpline(nbasis=12, domain_range=(-1,1)))
        >>> landmark_registration(fd, landmarks)
        FDataBasis(...)

    """

    warping = landmark_registration_warping(fd, landmarks, location=location,
                                            eval_points=eval_points)

    return fd.compose(warping)

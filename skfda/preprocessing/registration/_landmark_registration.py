"""Landmark Registration of functional data module.

This module contains methods to perform the landmark registration.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from ...representation import FData, FDataGrid
from ...representation.extrapolation import ExtrapolationLike
from ...representation.interpolation import SplineInterpolation
from ...typing._base import GridPointsLike
from ...typing._numpy import ArrayLike, NDArrayFloat

_FixedLocation = Union[float, Sequence[float]]
_LocationCallable = Callable[[np.ndarray], _FixedLocation]


def landmark_shift_deltas(
    fd: FData,
    landmarks: ArrayLike,
    location: Union[_FixedLocation, _LocationCallable, None] = None,
) -> NDArrayFloat:
    r"""Return the corresponding shifts to align the landmarks of the curves.

        Let :math:`t^*` the time where the landmarks of the curves will be
        aligned, and :math:`t_i` the location of the landmarks for each curve.
        The function will calculate the corresponding :math:`\delta_i` shuch
        that :math:`t_i = t^* + \delta_i`.

        This procedure will work independent of the dimension of the
        :term:`domain` and the :term:`codomain`.

    Args:
        fd: Functional data object.
        landmarks: List with the landmarks of the samples.
        location: Defines where
            the landmarks will be alligned. If a number or list is passed the
            landmarks will be alligned to it. In case of a callable is
            passed the location will be the result of the the call, the
            function should be accept as an unique parameter a numpy array
            with the list of landmarks.
            By default it will be used as location the mean of the original
            locations of the landmarks.

    Returns:
        Array containing the corresponding shifts.

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
        array([ 0.327, -0.173, -0.154])

        The registered samples can be obtained with a shift

        >>> fd.shift(shifts)
        FDataGrid(...)

    """
    landmarks = np.atleast_1d(landmarks)

    if len(landmarks) != fd.n_samples:
        raise ValueError(
            f"landmark list ({len(landmarks)}) must have the same"
            f" length than the number of samples ({fd.n_samples})",
        )

    loc_array: Union[float, Sequence[float], NDArrayFloat]

    # Parses location
    if location is None:
        loc_array = np.mean(landmarks)
    elif callable(location):
        loc_array = location(landmarks)
    else:
        loc_array = location

    loc_array = np.atleast_1d(loc_array)

    return landmarks - loc_array


def landmark_shift(
    *args: Any,
    **kwargs: Any,
) -> FDataGrid:

    warnings.warn(
        "Function 'landmark_shift' has been renamed. "
        "Use 'landmark_shift_registration' instead.",
        DeprecationWarning,
    )

    return landmark_shift_registration(*args, **kwargs)


def landmark_shift_registration(
    fd: FData,
    landmarks: ArrayLike,
    location: Union[_FixedLocation, _LocationCallable, None] = None,
    *,
    restrict_domain: bool = False,
    extrapolation: Optional[ExtrapolationLike] = None,
    grid_points: Optional[GridPointsLike] = None,
) -> FDataGrid:
    r"""
    Perform a shift of the curves to align the landmarks.

    Let :math:`t^*` the time where the landmarks of the curves will be
    aligned, :math:`t_i` the location of the landmarks for each curve
    and :math:`\delta_i= t_i - t^*`.

    The registered samples will have their feature aligned.

    .. math::
        x_i^*(t^*)=x_i(t^* + \delta_i)=x_i(t_i)

    Args:
        fd: Functional data object.
        landmarks: List with the landmarks of the samples.
        location: Defines where
            the landmarks will be alligned. If a numeric value is passed the
            landmarks will be alligned to it. In case of a callable is
            passed the location will be the result of the the call, the
            function should be accept as an unique parameter a numpy array
            with the list of landmarks.
            By default it will be used as location :math:`\frac{1}{2}(max(
            \text{landmarks})+ min(\text{landmarks}))` wich minimizes the
            max shift.
        restrict_domain: If True restricts the domain to
            avoid evaluate points outside the domain using extrapolation.
            Defaults uses extrapolation.
        extrapolation: Controls the
            extrapolation mode for elements outside the domain range.
            By default uses the method defined in fd. See extrapolation to
            more information.
        grid_points: Grid of points where
            the functions are evaluated in :func:`shift`.

    Returns:
        Functional data object with the registered samples.

    Examples:
        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import (
        ...     landmark_shift_registration,
        ... )

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, random_state=1)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, random_state=1)
        >>> landmarks = landmarks.squeeze()

        The function will return the sample registered

        >>> landmark_shift_registration(fd, landmarks)
        FDataGrid(...)

    """
    shifts = landmark_shift_deltas(fd, landmarks, location=location)

    return fd.shift(
        shifts,
        restrict_domain=restrict_domain,
        extrapolation=extrapolation,
        grid_points=grid_points,
    )


def landmark_elastic_registration_warping(
    fd: FData,
    landmarks: ArrayLike,
    *,
    location: Optional[ArrayLike] = None,
    grid_points: Optional[GridPointsLike] = None,
) -> FDataGrid:
    """Calculate the transformation used in landmark registration.

        Let :math:`t_{ij}` the time where the sample :math:`i` has the feature
        :math:`j` and :math:`t^*_j` the new time for the feature. The warping
        function will transform the new time in the old time, i.e.,
        :math:`h_i(t^*_j)=t_{ij}`.
        The registered samples can be obtained as :math:`x^*_i(t)=x_i(h_i(t))`.

        See :footcite:`ramsay+silverman_2005_functional_landmark`
        for a detailed explanation.

    Args:
        fd: Functional data object.
        landmarks: List containing landmarks for each samples.
        location: Defines where
            the landmarks will be alligned. By default it will be used as
            location the mean of the landmarks.
        grid_points: Grid of points where
            the functions are evaluated to obtain a discrete
            representation of the object.

    Returns:
        FDataGrid with the warpings function needed to
        register the functional data object.

    Raises:
        ValueError: If the object to be registered has domain dimension greater
            than 1 or the list of landmarks or locations does not match with
            the number of samples.

    References:
        .. footbibliography::

    Examples:
        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import (
        ...      landmark_elastic_registration_warping)

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, n_modes=2,
        ...                              random_state=9)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
        ...                                       random_state=9)
        >>> landmarks = landmarks.squeeze()

        The function will return the corresponding warping function

        >>> warping = landmark_elastic_registration_warping(fd, landmarks)
        >>> warping
        FDataGrid(...)

        The registered function can be obtained using function composition

        >>> fd.compose(warping)
        FDataGrid(...)

    """
    landmarks = np.asarray(landmarks)

    if fd.dim_domain > 1:
        raise NotImplementedError(
            "Method only implemented for objects with "
            "domain dimension up to 1.",
        )

    if len(landmarks) != fd.n_samples:
        raise ValueError(
            "The number of list of landmarks should be equal to "
            "the number of samples",
        )

    landmarks = landmarks.reshape((fd.n_samples, -1))

    location = (
        np.mean(landmarks, axis=0)
        if location is None
        else np.asarray(location)
    )

    assert isinstance(location, np.ndarray)

    n_landmarks = landmarks.shape[-1]

    data_matrix = np.empty((fd.n_samples, n_landmarks + 2))
    data_matrix[:, 0] = fd.domain_range[0][0]
    data_matrix[:, -1] = fd.domain_range[0][1]
    data_matrix[:, 1:-1] = landmarks

    if n_landmarks == len(location):
        if grid_points is None:
            grid_points = np.empty(n_landmarks + 2)
            grid_points[0] = fd.domain_range[0][0]
            grid_points[-1] = fd.domain_range[0][1]
            grid_points[1:-1] = location

    else:
        raise ValueError(
            f"Number of landmark locations should be equal than "
            f"the number of landmarks ({len(location)}) != ({n_landmarks})",
        )

    interpolation = SplineInterpolation(interpolation_order=3, monotone=True)

    warping = FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
        interpolation=interpolation,
        extrapolation='bounds',
    )

    try:
        warping_points = fd.grid_points
    except AttributeError:
        warping_points = None

    return warping.to_grid(warping_points)


def landmark_registration(
    *args: Any,
    **kwargs: Any,
) -> FDataGrid:

    warnings.warn(
        "Function 'landmark_registration' has been renamed. "
        "Use 'landmark_elastic_registration' instead.",
        DeprecationWarning,
    )

    return landmark_elastic_registration(*args, **kwargs)


def landmark_elastic_registration(
    fd: FData,
    landmarks: ArrayLike,
    *,
    location: Optional[ArrayLike] = None,
    grid_points: Optional[GridPointsLike] = None,
) -> FDataGrid:
    """
    Perform landmark registration of the curves.

    Let :math:`t_{ij}` the time where the sample :math:`i` has the feature
    :math:`j` and :math:`t^*_j` the new time for the feature.
    The registered samples will have their features aligned, i.e.,
    :math:`x^*_i(t^*_j)=x_i(t_{ij})`.

    See :footcite:`ramsay+silverman_2005_functional_landmark`
    for a detailed explanation.

    Args:
        fd: Functional data object.
        landmarks: List containing landmarks for each samples.
        location: Defines where
            the landmarks will be alligned. By default it will be used as
            location the mean of the landmarks.
        grid_points: Grid of points where
            the functions are evaluated to obtain a discrete
            representation of the object. In case of objects with
            multidimensional :term:`domain` a list axis with points of
            evaluation for each dimension.

    Returns:
        FDataGrid with the functional data object registered.

    References:
        .. footbibliography::

    Examples:
        >>> from skfda.datasets import make_multimodal_landmarks
        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.preprocessing.registration import (
        ...     landmark_elastic_registration,
        ... )
        >>> from skfda.representation.basis import BSplineBasis

        We will create a data with landmarks as example

        >>> fd = make_multimodal_samples(n_samples=3, n_modes=2,
        ...                              random_state=9)
        >>> landmarks = make_multimodal_landmarks(n_samples=3, n_modes=2,
        ...                                       random_state=9)
        >>> landmarks = landmarks.squeeze()

        The function will return the registered curves

        >>> landmark_elastic_registration(fd, landmarks)
        FDataGrid(...)

        This method will work for FDataBasis as for FDataGrids

        >>> fd = fd.to_basis(BSplineBasis(n_basis=12))
        >>> landmark_elastic_registration(fd, landmarks)
        FDataGrid(...)

    """
    warping = landmark_elastic_registration_warping(
        fd,
        landmarks,
        location=location,
        grid_points=grid_points,
    )

    return fd.to_grid(grid_points).compose(warping)

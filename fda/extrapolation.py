"""Module with the extrapolation methods.

Defines methods to evaluate points outside the domain range.

"""

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class ExtrapolationType(Enum):
    """Enum with extrapolation types. Defines the extrapolation mode for
        elements outside the domain range.
    """

    NONE = "none" #: Uses directly the value given by the FData object
    PERIODIC = "periodic" #: Extends the domain range periodically.
    BOUNDS = "bounds" #: Uses the boundary value.
    EXCEPTION = "exception" #: Raise an exception
    NAN = "nan" #: Fill the points outside the domain with NaN
    ZEROS = "zeros" #: Fill the points outside the domain with zeros

class Extrapolator(ABC):
    """Defines the structure of a extrapolator.

    Extrapolators defines how to evaluate points outside the domain of a
    :class:´FData´. They are called internally by `evaluate`. Custom
    extrapolators could be done with a subclass of `Extrapolator`.

    """

    @abstractmethod
    def extrapolate(self, fdata, eval_points, derivative=0, keepdims=False):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (numpy.ndarray): Numpy array with the evalation points
                outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image`, where the entry (i,j) will be
                the coordinate j of the time i, or `nsamples` x `n_eval_points`
                x `ndim_image`, where the first dimension indicates the sample.
            derivate (numeric, optional): Order of derivative to be evaluated.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points.
        """

        pass

    def __call__(self, fdata, eval_points, derivative=0, keepdims=False):
        """
        Evaluate points outside the domain range. This function is a wrapper of
        the method `extrapolate`.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (numpy.ndarray): Numpy array with the evalation points
                outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image`, where the entry (i,j) will be
                the coordinate j of the time i, or `nsamples` x `n_eval_points`
                x `ndim_image`, where the first dimension indicates the sample.
            derivate (numeric, optional): Order of derivative to be evaluated.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points.
        """

        return self.extrapolate(fdata, eval_points, derivative=derivative,
                                keepdims=keepdims)



class PeriodicExtrapolation(Extrapolator):
    """Extends the domain range periodically."""

    def extrapolate(self, fdata, eval_points, derivative=0, keepdims=False):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (numpy.ndarray): Numpy array with the evalation points
                outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image`, where the entry (i,j) will be
                the coordinate j of the time i, or `nsamples` x `n_eval_points`
                x `ndim_image`, where the first dimension indicates the sample.
            derivate (numeric, optional): Order of derivative to be evaluated.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points.
        """

        domain_range = fdata.domain_range


        # Extends the domain periodically in each dimension
        for i in range(fdata.ndim_domain):
            a, b = domain_range[i]
            np.subtract(eval_points[..., i], a, eval_points[..., i])
            np.mod(eval_points[..., i], b - a, eval_points[..., i])
            np.add(eval_points[..., i], a, eval_points[..., i])



        # Case evaluation at different times for each sample
        if fdata.ndim_domain == 1 and eval_points.ndim == 3:
            eval_points = eval_points.reshape((eval_points.shape[0],
                                               eval_points.shape[1]))

        # Case unidimensional domain
        elif fdata.ndim_domain == 1:
            eval_points = eval_points.reshape(eval_points.shape[0])


        return fdata(eval_points, derivative=derivative,
                     extrapolation=ExtrapolationType.NONE, keepdims=keepdims)


class BoundaryExtrapolation(Extrapolator):
    """Extends the domain range using the boundary values."""

    def extrapolate(self, fdata, eval_points, derivative=0, keepdims=False):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (numpy.ndarray): Numpy array with the evalation points
                outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image`, where the entry (i,j) will be
                the coordinate j of the time i, or `nsamples` x `n_eval_points`
                x `ndim_image`, where the first dimension indicates the sample.
            derivate (numeric, optional): Order of derivative to be evaluated.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points.
        """

        domain_range = fdata.domain_range

        for i in range(fdata.ndim_domain):
            a, b = domain_range[i]
            eval_points[eval_points[..., i] < a, i] = a
            eval_points[eval_points[..., i] > b, i] = b

        if fdata.ndim_domain == 1 and eval_points.ndim == 3:
            eval_points = eval_points.reshape((eval_points.shape[0],
                                               eval_points.shape[1]))

        elif fdata.ndim_domain == 1:
            eval_points = eval_points.reshape(eval_points.shape[0])


        return fdata(eval_points, derivative=derivative,
                     extrapolation=ExtrapolationType.NONE, keepdims=keepdims)


class ExceptionExtrapolation(Extrapolator):
    """Raise and exception if a point is evaluated outside the domain range."""

    def extrapolate(self, fdata, eval_points, derivative=0, keepdims=False):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (numpy.ndarray): Numpy array with the evalation points
                outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image`, where the entry (i,j) will be
                the coordinate j of the time i, or `nsamples` x `n_eval_points`
                x `ndim_image`, where the first dimension indicates the sample.
            derivate (numeric, optional): Order of derivative to be evaluated.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points.
        """

        if eval_points.ndim == 3:
            n_points = eval_points.shape[1]
        else:
            n_points = eval_points.shape[0]

        raise ValueError(f"Attempt to evaluate {n_points} points outside the "
                         f"domain range.")

class FillExtrapolation(Extrapolator):
    """The values outside the domain range will be filled with a fixed value."""

    def __init__(self, fill_value):
        self.fill_value = fill_value

        super().__init__()

    def extrapolate(self, fdata, eval_points, derivative=0, keepdims=False):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (numpy.ndarray): Numpy array with the evalation points
                outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image`, where the entry (i,j) will be
                the coordinate j of the time i, or `nsamples` x `n_eval_points`
                x `ndim_image`, where the first dimension indicates the sample.
            derivate (numeric, optional): Order of derivative to be evaluated.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points.
        """

        # Case all the samples evaluates at same time
        if eval_points.ndim == 2:
            if keepdims or fdata.ndim_image > 1:
                shape = (fdata.nsamples, eval_points.shape[0],
                         fdata.ndim_image)
            else:
                shape = (fdata.nsamples, eval_points.shape[1])

            res = np.full(shape, self.fill_value)

        else:

            idx_ext = _extrapolation_index(eval_points, fdata.domain_range)

            if fdata.ndim_image == 1:
                eval_points = eval_points.reshape(eval_points.shape[0],
                                                  eval_points.shape[1])

            res =  fdata(eval_points, derivative=derivative,
                         extrapolation=ExtrapolationType.NONE, keepdims=keepdims)

            if res.ndim == 3:
                res[idx_ext, :] = self.fill_value
            else:
                res[idx_ext] = self.fill_value

        return res


def _parse_extrapolation(extrapolation, fdata=None):
    """Parse the argument `extrapolation` in 'evaluate'.

    Args:
        extrapolation (:class:´Extrapolator´, str or Callable): Argument
            extrapolation to be parsed.
        fdata (:class:´FData´): Object with the default extrapolation.

    Returns:
        (:class:´Extrapolator´ or Callable): Extrapolation method.

    """

    # If extrapolation is None returns the object default
    if extrapolation is None:

        if fdata is not None:
            return fdata.extrapolation
        else:
            return None
    elif callable(extrapolation):
        return extrapolation

    # If not callable parses from string or ExtrapolationType enum
    type = ExtrapolationType(extrapolation)

    if type is ExtrapolationType.NONE:
        return None
    elif type is ExtrapolationType.EXCEPTION:
        return ExceptionExtrapolation()
    elif type is ExtrapolationType.PERIODIC:
        return PeriodicExtrapolation()
    elif type is ExtrapolationType.BOUNDS:
        return BoundaryExtrapolation()
    elif type is ExtrapolationType.NAN:
        return FillExtrapolation(np.nan)
    elif type is ExtrapolationType.ZEROS:
        return FillExtrapolation(0)


def _extrapolation_index(eval_points, domain_range):
    """Checks the points that need to be extrapolated.

    Args:
        eval_points (numpy.ndarray): Array with shape `n_eval_points` x
            `ndim_image` with the evaluation points, or shape ´nsamples´ x
            `n_eval_points` x `ndim_image` with different evaluation points
            for each sample.

        domain_range (list of tuples): List with the bounds of each dimenson
            of the domain range.

    Returns:

        (numpy.ndarray): Array with boolean index. The positions with True in
            the index are outside the domain range.

    """

    # Case all samples evaluated with same times
    if eval_points.ndim == 2:
        index = np.full(eval_points.shape[0], False)

    else: # Different times for each sample
        index = np.full((eval_points.shape[0], eval_points.shape[1]) , False)

    for i in range(len(domain_range)):
        a, b = domain_range[i]
        np.logical_or(index, eval_points[..., i] < a, index)
        np.logical_or(index, eval_points[..., i] > b, index)

    return index

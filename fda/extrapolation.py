"""Module with the extrapolation methods.

Defines methods to evaluate points outside the domain range.

"""

from abc import ABC, abstractmethod

import numpy as np

class Extrapolator(ABC):
    """Defines the structure of a extrapolator.

    Extrapolators defines how to evaluate points outside the domain of a
    :class:´FData´. They are called internally by `evaluate`. Custom
    extrapolators could be done with a subclass of `Extrapolator`.

    """

    @abstractmethod
    def __call__(self, fdata, eval_points, *, derivative=0):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (:class: numpy.ndarray): Numpy array with the evaluation
                points outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image` or `nsamples` x `n_eval_points`
                x `ndim_image`.
            derivate (numeric, optional): Order of derivative to be evaluated.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points in
            a matrix with shape `nsamples` x `n_eval_points`x `ndim_image`.
        """

        pass



class PeriodicExtrapolation(Extrapolator):
    """Extends the domain range periodically."""

    def __call__(self, fdata, eval_points, *, derivative=0):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (:class: numpy.ndarray): Numpy array with the evalation
                points outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image` or `nsamples` x `n_eval_points`
                x `ndim_image`.
            derivate (numeric, optional): Order of derivative to be evaluated.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points in
            a matrix with shape `nsamples` x `n_eval_points`x `ndim_image`.
        """

        domain_range = np.asarray(fdata.domain_range)

        # Extends the domain periodically in each dimension
        eval_points -= domain_range[:, 0]
        eval_points %= domain_range[:, 1] - domain_range[:, 0]
        eval_points += domain_range[:, 0]


        if eval_points.ndim == 3:
            res = fdata._evaluate_composed(eval_points, derivative=derivative)
        else:
            res = fdata._evaluate(eval_points, derivative=derivative)

        return res


class BoundaryExtrapolation(Extrapolator):
    """Extends the domain range using the boundary values."""

    def __call__(self, fdata, eval_points, *, derivative=0):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (:class: numpy.ndarray): Numpy array with the evalation
                points outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image` or `nsamples` x `n_eval_points`
                x `ndim_image`.
            derivate (numeric, optional): Order of derivative to be evaluated.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points in
            a matrix with shape `nsamples` x `n_eval_points`x `ndim_image`.
        """

        domain_range = fdata.domain_range

        for i in range(fdata.ndim_domain):
            a, b = domain_range[i]
            eval_points[eval_points[..., i] < a, i] = a
            eval_points[eval_points[..., i] > b, i] = b


        if eval_points.ndim == 3:

            res = fdata._evaluate_composed(eval_points, derivative=derivative)
        else:

            res = fdata._evaluate(eval_points, derivative=derivative)

        return res


class ExceptionExtrapolation(Extrapolator):
    """Raise and exception if a point is evaluated outside the domain range."""

    def __call__(self, fdata, eval_points, *, derivative=0):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (:class: numpy.ndarray): Numpy array with the evalation
                points outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image` or `nsamples` x `n_eval_points`
                x `ndim_image`.
            derivate (numeric, optional): Order of derivative to be evaluated.

        Raises:
            ValueError: when the extrapolation method is called.
        """

        n_points = eval_points.shape[-2]

        raise ValueError(f"Attempt to evaluate {n_points} points outside the "
                         f"domain range.")

class FillExtrapolation(Extrapolator):
    """The values outside the domain range will be filled with a fixed value."""

    def __init__(self, fill_value):

        self.fill_value = fill_value

        super().__init__()

    def __call__(self, fdata, eval_points, *, derivative=0):
        """
        Evaluate points outside the domain range.

        Args:
            fdata (:class:´FData´): Object where the evaluation is taken place.
            eval_points (:class: numpy.ndarray): Numpy array with the evalation
                points outside the domain range. The shape of the array may be
                `n_eval_points` x `ndim_image` or `nsamples` x `n_eval_points`
                x `ndim_image`.
            derivate (numeric, optional): Order of derivative to be evaluated.

        Returns:
            (numpy.ndarray): numpy array with the evaluation of the points in
            a matrix with shape `nsamples` x `n_eval_points`x `ndim_image`.
        """


        shape = (fdata.nsamples, eval_points.shape[-2], fdata.ndim_image)

        return np.full(shape, self.fill_value)



#: Dictionary with the extrapolation methods.
extrapolation_methods = {"bounds" : BoundaryExtrapolation(),
                         "exception" : ExceptionExtrapolation(),
                         "nan" : FillExtrapolation(np.nan),
                         "none" : None,
                         "periodic" : PeriodicExtrapolation(),
                         "zeros" : FillExtrapolation(0)}

"""Module with the extrapolation methods.

Defines methods to evaluate points outside the domain range.

"""

from abc import ABC, abstractmethod

import numpy as np

class Extrapolator(ABC):
    """Defines the structure of an extrapolator.

    An extrapolator defines how to evaluate points outside the domain of a
    functional object. They are called internally by :func:`evaluate`. Custom
    extrapolators could be done subclassing `Extrapolator` or with a compatible
    callable.

    The callable should accept 3 arguments, a :class:`FData` instance, a numpy
    array with the evaluation points to be extrapolated and optionally the order
    of derivation.

    The shape of the array with the evaluation points received by the
    extrapolator may be `n_eval_points` x `ndim_image`, case in wich the samples
    would be evaluated at the same time or `nsamples` x `n_eval_points`
    x `ndim_image` with different evaluation points per sample.

    The extrapolator should return a matrix with points extrapolation in a
    matrix with shape `n_samples` x `n_eval_points`x `ndim_image`.
    """

    @abstractmethod
    def __call__(self, fdata, eval_points, *, derivative=0):
        """Evaluate points outside the domain range.

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
    """Extends the domain range periodically.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.extrapolation import PeriodicExtrapolation
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = PeriodicExtrapolation()
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[-0.724,  0.976, -0.724],
               [-1.086,  0.759, -1.086]])

        This extrapolator is equivalent to the string `"periodic"`

        >>> fd.extrapolation = 'periodic'
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[-0.724,  0.976, -0.724],
               [-1.086,  0.759, -1.086]])
    """

    def __call__(self, fdata, eval_points, *, derivative=0):
        """Evaluate points outside the domain range.

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
    """Extends the domain range using the boundary values.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.extrapolation import BoundaryExtrapolation
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = BoundaryExtrapolation()
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[ 0.976,  0.976,  0.797],
               [ 0.759,  0.759,  1.125]])

        This extrapolator is equivalent to the string `"bounds"`.

        >>> fd.extrapolation = 'bounds'
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[ 0.976,  0.976,  0.797],
               [ 0.759,  0.759,  1.125]])
    """

    def __call__(self, fdata, eval_points, *, derivative=0):
        """Evaluate points outside the domain range.

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
    """Raise and exception if a point is evaluated outside the domain range.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.extrapolation import ExceptionExtrapolation
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = ExceptionExtrapolation()
        >>> try:
        ...     fd([-.5, 0, 1.5]).round(3)
        ... except ValueError as e:
        ...     print(e)
        Attempt to evaluate 2 points outside the domain range.

        This extrapolator is equivalent to the string `"exception"`.

        >>> fd.extrapolation = 'exception'
        >>> try:
        ...     fd([-.5, 0, 1.5]).round(3)
        ... except ValueError as e:
        ...     print(e)
        Attempt to evaluate 2 points outside the domain range.

    """

    def __call__(self, fdata, eval_points, *, derivative=0):
        """Evaluate points outside the domain range.

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
    """Values outside the domain range will be filled with a fixed value.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.extrapolation import FillExtrapolation
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = FillExtrapolation(0)
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[ 0.   ,  0.976,  0.   ],
               [ 0.   ,  0.759,  0.   ]])

        The previous extrapolator is equivalent to the string `"zeros"`.
        In the same way FillExtrapolation(np.nan) is equivalent to `"nan"`.

        >>> fd.extrapolation = "nan"
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[   nan,  0.976,    nan],
               [   nan,  0.759,    nan]])
    """

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

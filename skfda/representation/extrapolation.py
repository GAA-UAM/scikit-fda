"""Module with the extrapolation methods.

Defines methods to evaluate points outside the domain range.

"""

from abc import ABC, abstractmethod

from .evaluator import EvaluatorConstructor, Evaluator, GenericEvaluator

import numpy as np


class PeriodicExtrapolation(EvaluatorConstructor):
    """Extends the domain range periodically.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import PeriodicExtrapolation
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

    def evaluator(self, fdata):
        """Returns the evaluator used by class:`FData`.

        Returns:
            (:class:`Evaluator`): Evaluator of the periodic extrapolation.

        """
        return GenericEvaluator(fdata, _periodic_evaluation)


def _periodic_evaluation(fdata, eval_points, *, derivative=0):
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


class BoundaryExtrapolation(EvaluatorConstructor):
    """Extends the domain range using the boundary values.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import BoundaryExtrapolation
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

    def evaluator(self, fdata):
        """Returns the evaluator used by class:`FData`.

        Returns:
            (:class:`Evaluator`): Evaluator of the periodic boundary.

        """
        return GenericEvaluator(fdata, _boundary_evaluation)


def _boundary_evaluation(fdata, eval_points, *, derivative=0):
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


class ExceptionExtrapolation(EvaluatorConstructor):
    """Raise and exception.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import ExceptionExtrapolation
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

    def evaluator(self, fdata):
        """Returns the evaluator used by class:`FData`.

        Returns:
            (:class:`Evaluator`): Evaluator of the periodic extrapolation.

        """
        return GenericEvaluator(fdata, _exception_evaluation)


def _exception_evaluation(fdata, eval_points, *, derivative=0):
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


class FillExtrapolation(EvaluatorConstructor):
    """Values outside the domain range will be filled with a fixed value.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import FillExtrapolation
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
        """Returns the evaluator used by class:`FData`.

        Returns:
            (:class:`Evaluator`): Evaluator of the periodic extrapolation.

        """
        self._fill_value = fill_value

        super().__init__()

    @property
    def fill_value(self):
        """Returns the fill value of the extrapolation"""
        return self._fill_value

    def __eq__(self, other):
        """Equality operator bethween evaluator constructors"""
        return super().__eq__(other) and (self.fill_value == other.fill_value
                                          or self.fill_value is other.fill_value)

    def evaluator(self, fdata):

        return FillExtrapolationEvaluator(fdata, self.fill_value)


class FillExtrapolationEvaluator(Evaluator):

    def __init__(self, fdata, fill_value):
        self.fill_value = fill_value
        self.fdata = fdata

    def _fill(self, eval_points):
        shape = (self.fdata.nsamples, eval_points.shape[-2],
                 self.fdata.ndim_image)
        return np.full(shape, self.fill_value)

    def evaluate(self, eval_points, *, derivative=0):
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
        return self._fill(eval_points)

    def evaluate_composed(self, eval_points, *, derivative=0):
        """Evaluation method.

        Evaluates the samples at different evaluation points. The evaluation
        call will receive a 3-d array with the evaluation points for each sample.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is False.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(n_samples, number_eval_points, ndim_domain)` with the
                 evaluation points for each sample.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Numpy 3d array with shape `(n_samples,
                number_eval_points, ndim_image)` with the result of the
                evaluation. The entry (i,j,k) will contain the value k-th image
                dimension of the i-th sample, at the j-th evaluation point.

        """
        return self._fill(eval_points)


def _parse_extrapolation(extrapolation):
    """Parse the argument `extrapolation` of `FData`.

    If extrapolation is None returns the default extrapolator.

    Args:
        extrapolation (:class:´Extrapolator´, str or Callable): Argument
            extrapolation to be parsed.
        fdata (:class:´FData´): Object with the default extrapolation.

    Returns:
        (:class:´Extrapolator´ or Callable): Extrapolation method.

    """
    if extrapolation is None:
        return None

    elif isinstance(extrapolation, str):
        return extrapolation_methods[extrapolation.lower()]

    else:
        return extrapolation


#: Dictionary with the extrapolation methods.
extrapolation_methods = {"bounds": BoundaryExtrapolation(),
                         "exception": ExceptionExtrapolation(),
                         "nan": FillExtrapolation(np.nan),
                         "none": None,
                         "periodic": PeriodicExtrapolation(),
                         "zeros": FillExtrapolation(0)}

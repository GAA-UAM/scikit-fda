"""Module with the extrapolation methods.

Defines methods to evaluate points outside the :term:`domain` range.

"""

from typing import Optional, Union

import numpy as np

from .evaluator import Evaluator


class PeriodicExtrapolation(Evaluator):
    """Extends the :term:`domain` range periodically.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import (
        ...     PeriodicExtrapolation)
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = PeriodicExtrapolation()
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[[-0.724],
                [ 0.976],
                [-0.724]],
               [[-1.086],
                [ 0.759],
                [-1.086]]])

        This extrapolator is equivalent to the string `"periodic"`

        >>> fd.extrapolation = 'periodic'
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[[-0.724],
                [ 0.976],
                [-0.724]],
               [[-1.086],
                [ 0.759],
                [-1.086]]])
    """

    def evaluate(self, fdata, eval_points, *, aligned=True):

        domain_range = np.asarray(fdata.domain_range)

        # Extends the domain periodically in each dimension
        eval_points -= domain_range[:, 0]
        eval_points %= domain_range[:, 1] - domain_range[:, 0]
        eval_points += domain_range[:, 0]

        res = fdata(eval_points, aligned=aligned)

        return res


class BoundaryExtrapolation(Evaluator):
    """Extends the :term:`domain` range using the boundary values.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import (
        ...     BoundaryExtrapolation)
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = BoundaryExtrapolation()
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[[ 0.976],
                [ 0.976],
                [ 0.797]],
               [[ 0.759],
                [ 0.759],
                [ 1.125]]])

        This extrapolator is equivalent to the string `"bounds"`.

        >>> fd.extrapolation = 'bounds'
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[[ 0.976],
                [ 0.976],
                [ 0.797]],
               [[ 0.759],
                [ 0.759],
                [ 1.125]]])
    """

    def evaluate(self, fdata, eval_points, *, aligned=True):

        domain_range = fdata.domain_range

        for i in range(fdata.dim_domain):
            a, b = domain_range[i]
            eval_points[eval_points[..., i] < a, i] = a
            eval_points[eval_points[..., i] > b, i] = b

        res = fdata(eval_points, aligned=aligned)

        return res


class ExceptionExtrapolation(Evaluator):
    """Raise and exception.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import (
        ...     ExceptionExtrapolation)
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

    def evaluate(self, fdata, eval_points, *, aligned=True):

        n_points = eval_points.shape[-2]

        raise ValueError(f"Attempt to evaluate {n_points} points outside the "
                         f"domain range.")


class FillExtrapolation(Evaluator):
    """
    Values outside the :term:`domain` range will be filled with a fixed value.

    Examples:

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.extrapolation import FillExtrapolation
        >>> fd = make_sinusoidal_process(n_samples=2, random_state=0)

        We can set the default type of extrapolation

        >>> fd.extrapolation = FillExtrapolation(0)
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[[ 0.   ],
                [ 0.976],
                [ 0.   ]],
               [[ 0.   ],
                [ 0.759],
                [ 0.   ]]])

        The previous extrapolator is equivalent to the string `"zeros"`.
        In the same way FillExtrapolation(np.nan) is equivalent to `"nan"`.

        >>> fd.extrapolation = "nan"
        >>> fd([-.5, 0, 1.5]).round(3)
        array([[[   nan],
                [ 0.976],
                [   nan]],
               [[   nan],
                [ 0.759],
                [   nan]]])
    """

    def __init__(self, fill_value):
        self.fill_value = fill_value

    def _fill(self, fdata, eval_points):
        shape = (fdata.n_samples, eval_points.shape[-2],
                 fdata.dim_codomain)
        return np.full(shape, self.fill_value)

    def evaluate(self, fdata, eval_points, *, aligned=True):

        return self._fill(fdata, eval_points)

    def __repr__(self):
        """repr method of FillExtrapolation"""
        return (f"{type(self).__name__}("
                f"fill_value={self.fill_value})")

    def __eq__(self, other):
        """Equality operator bethween FillExtrapolation instances."""
        return (super().__eq__(other) and
                self.fill_value == other.fill_value
                # NaNs compare unequal. Should we distinguish between
                # different NaN types and payloads?
                or np.isnan(self.fill_value) and np.isnan(other.fill_value))


def _parse_extrapolation(
    extrapolation: Optional[Union[str, Evaluator]],
) -> Optional[Evaluator]:
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

"""Module with the extrapolation methods.

Defines methods to evaluate points outside the :term:`domain` range.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, Optional, Union, overload

import numpy as np
from typing_extensions import Literal

from ..typing._base import EvaluationPoints
from ..typing._numpy import NDArrayFloat
from .evaluator import Evaluator

if TYPE_CHECKING:
    from ._functional_data import FData

ExtrapolationLike = Union[
    Evaluator,
    Literal["bounds", "exception", "nan", "none", "periodic", "zeros"],
]


class PeriodicExtrapolation(Evaluator):
    """Extend the :term:`domain` range periodically.

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

    def _evaluate(  # noqa: D102
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        domain_range = np.asarray(fdata.domain_range)

        # Extends the domain periodically in each dimension
        eval_points -= domain_range[:, 0]
        eval_points %= domain_range[:, 1] - domain_range[:, 0]
        eval_points += domain_range[:, 0]

        return fdata(eval_points, aligned=aligned)


class BoundaryExtrapolation(Evaluator):
    """Extend the :term:`domain` range using the boundary values.

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

    def _evaluate(  # noqa: D102
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        domain_range = fdata.domain_range

        eval_points = np.asarray(eval_points)

        for i in range(fdata.dim_domain):
            a, b = domain_range[i]
            eval_points[eval_points[..., i] < a, i] = a
            eval_points[eval_points[..., i] > b, i] = b

        return fdata(eval_points, aligned=aligned)


class ExceptionExtrapolation(Evaluator):
    """Raise an exception.

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
        Attempt to evaluate points outside the domain range.

        This extrapolator is equivalent to the string `"exception"`.

        >>> fd.extrapolation = 'exception'
        >>> try:
        ...     fd([-.5, 0, 1.5]).round(3)
        ... except ValueError as e:
        ...     print(e)
        Attempt to evaluate points outside the domain range.

    """

    def _evaluate(  # noqa: D102
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NoReturn:

        raise ValueError(
            "Attempt to evaluate points outside the domain range.",
        )


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

    def __init__(self, fill_value: float) -> None:
        self.fill_value = fill_value

    def _evaluate(  # noqa: D102
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        shape = (
            fdata.n_samples,
            eval_points.shape[-2],
            fdata.dim_codomain,
        )
        return np.full(shape, self.fill_value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"fill_value={self.fill_value})"
        )

    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and (
                self.fill_value == other.fill_value
                # NaNs compare unequal. Should we distinguish between
                # different NaN types and payloads?
                or (np.isnan(self.fill_value) and np.isnan(other.fill_value))
            )
        )


@overload
def _parse_extrapolation(
    extrapolation: None,
) -> None:
    pass


@overload
def _parse_extrapolation(
    extrapolation: ExtrapolationLike,
) -> Evaluator:
    pass


def _parse_extrapolation(
    extrapolation: Optional[ExtrapolationLike],
) -> Optional[Evaluator]:
    """Parse the argument `extrapolation` of `FData`.

    If extrapolation is None returns the default extrapolator.

    Args:
        extrapolation (:class:´Extrapolator´, str or Callable): Argument
            extrapolation to be parsed.

    Returns:
        (:class:´Extrapolator´ or Callable): Extrapolation method.

    """
    if extrapolation is None:
        return None

    elif isinstance(extrapolation, str):
        return extrapolation_methods[extrapolation.lower()]

    return extrapolation


#: Dictionary with the extrapolation methods.
extrapolation_methods = {
    "bounds": BoundaryExtrapolation(),
    "exception": ExceptionExtrapolation(),
    "nan": FillExtrapolation(np.nan),
    "none": None,
    "periodic": PeriodicExtrapolation(),
    "zeros": FillExtrapolation(0),
}

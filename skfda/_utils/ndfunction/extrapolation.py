"""
Module with the extrapolation methods.

Defines methods to evaluate points outside the :term:`domain` range.

"""
from __future__ import annotations

import math
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from typing_extensions import override

from ._array_api import Array, DType, RealDtype, Shape
from .evaluator import Evaluator
from .utils._points import input_points_batch_shape

if TYPE_CHECKING:
    from ._ndfunction import NDFunction

A = TypeVar('A', bound=Array[Shape, DType])
RealArray = TypeVar('RealArray', bound=Array[Shape, RealDtype])

ExtrapolationLike = Union[
    Evaluator[A],
    Literal["bounds", "exception", "nan", "none", "periodic", "zeros"],
]


class PeriodicExtrapolation(Evaluator[RealArray]):
    """
    Extend the :term:`domain` range periodically.

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

    @override
    def __call__(  # noqa:D102
        self,
        function: NDFunction[RealArray],
        /,
        eval_points: RealArray,
        *,
        aligned: bool = True,
    ) -> RealArray:

        lower, upper = function.domain.bounding_box

        # Extends the domain periodically in each dimension
        domain_len = upper - lower

        displacement = (eval_points - lower) % domain_len

        return function(
            lower + displacement,
            aligned=aligned,
            extrapolation=None,
        )


class BoundaryExtrapolation(Evaluator[RealArray]):
    """
    Extend the :term:`domain` range using the boundary values.

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

    @override
    def __call__(  # noqa:D102
        self,
        function: NDFunction[RealArray],
        /,
        eval_points: RealArray,
        *,
        aligned: bool = True,
    ) -> RealArray:

        lower, upper = function.domain.bounding_box

        xp = function.array_backend

        eval_points = xp.where(eval_points < lower, lower, eval_points)
        eval_points = xp.where(eval_points > upper, upper, eval_points)

        return function(
            eval_points,
            aligned=aligned,
            extrapolation=None,
        )


class ExceptionExtrapolation(Evaluator[A]):
    """
    Raise an exception.

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

    @override
    def __call__(  # noqa:D102
        self,
        function: NDFunction[A],
        /,
        eval_points: A,
        *,
        aligned: bool = True,
    ) -> A:

        raise ValueError(
            "Attempt to evaluate points outside the domain range.",
        )


class FillExtrapolation(Evaluator[A]):
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

    @override
    def __call__(  # noqa:D102
        self,
        function: NDFunction[A],
        /,
        eval_points: A,
        *,
        aligned: bool = True,
    ) -> A:

        shape = (
            function.shape
            + input_points_batch_shape(
                eval_points,
                function=function,
                aligned=aligned,
            )
            + function.output_shape
        )
        return function.array_backend.full(  # type: ignore[no-any-return]
            shape,
            self.fill_value,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(fill_value={self.fill_value})"
        )

    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and (
                self.fill_value == other.fill_value
                # NaNs compare unequal. Should we distinguish between
                # different NaN types and payloads?
                or (
                    math.isnan(self.fill_value)
                    and math.isnan(other.fill_value)
                )
            )
        )


@overload
def _parse_extrapolation(
    extrapolation: None,
) -> None:
    pass


@overload
def _parse_extrapolation(
    extrapolation: ExtrapolationLike[A],
) -> Evaluator[A]:
    pass


def _parse_extrapolation(
    extrapolation: ExtrapolationLike[A] | None,
) -> Evaluator[A] | None:
    """
    Parse a extrapolation.

    If extrapolation is ``None`` returns ``None``.

    Args:
        extrapolation: Extrapolation to be parsed.

    Returns:
        Extrapolation method.

    """
    if extrapolation is None:
        return None

    elif isinstance(extrapolation, str):
        return extrapolation_methods[extrapolation.lower()]

    return extrapolation


#: Dictionary with the extrapolation methods.
extrapolation_methods: Mapping[
    str, Evaluator[Any] | None,
] = {
    "bounds": BoundaryExtrapolation(),
    "exception": ExceptionExtrapolation(),
    "nan": FillExtrapolation(np.nan),
    "none": None,
    "periodic": PeriodicExtrapolation(),
    "zeros": FillExtrapolation(0),
}

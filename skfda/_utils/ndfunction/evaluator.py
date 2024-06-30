"""
This module contains the evaluator protocol.

An evaluator is a callback protocol used for function evaluation, such
as the one performed for extrapolation and interpolation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, TypeVar

from ._array_api import Array, DType, Shape

if TYPE_CHECKING:
    from ._ndfunction import NDFunction


A = TypeVar("A", bound=Array[Shape, DType])


class Evaluator(Protocol[A]):
    """
    Structure of an evaluator.

    An evaluator defines how to evaluate points of a function.
    It can be used as extrapolator to evaluate points outside the
    :term:`domain` range or as interpolation in classes that present
    empty regions between measurements.

    """

    @abstractmethod
    def __call__(
        self,
        function: NDFunction[A],
        /,
        eval_points: A,
        *,
        aligned: bool = True,
    ) -> A:
        """
        Evaluate the samples at evaluation points.

        The evaluation call will receive a 2-d array with the
        evaluation points, or a 3-d array with the evaluation points per
        sample if ``aligned`` is ``False``.

        Args:
            function: Object to evaluate.
            eval_points: Array with shape
                ``(number_eval_points, dim_domain)`` with the
                evaluation points.
            aligned: Whether the input points are the same for each function,
                or an array of different points per function is passed.

        Returns:
            Array with shape
            ``(input_shape, number_eval_points, output_shape)`` with the
            result of the evaluation.

        """

    def __repr__(self) -> str:
        return f"{type(self)}()"

    def __eq__(self, other: object) -> bool:
        """Equality operator between evaluators."""
        return isinstance(other, type(self))

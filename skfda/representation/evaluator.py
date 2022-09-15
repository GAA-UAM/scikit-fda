"""
This module contains the structure of the evaluator.

The evaluator is the core of the FData object for extrapolation and
evaluation of FDataGrids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import Protocol

from ..typing._base import EvaluationPoints
from ..typing._numpy import ArrayLike, NDArrayFloat

if TYPE_CHECKING:
    from ._functional_data import FData


class Evaluator(ABC):
    """
    Structure of an evaluator.

    An evaluator defines how to evaluate points of a functional object, it
    can be used as extrapolator to evaluate points outside the :term:`domain`
    range or as interpolation in a :class:`FDataGrid`. The corresponding
    examples of Interpolation and Extrapolation shows the basic usage of
    this class.

    The evaluator is called internally by :func:`evaluate`.

    """

    @abstractmethod
    def _evaluate(
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:
        """
        Evaluate the samples at evaluation points.

        Must be overriden in subclasses.

        """
        pass

    def __call__(
        self,
        fdata: FData,
        eval_points: ArrayLike,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:
        """
        Evaluate the samples at evaluation points.

        The evaluation call will receive a 2-d array with the
        evaluation points, or a 3-d array with the evaluation points per
        sample if ``aligned`` is ``False``.

        Args:
            fdata: Object to evaluate.
            eval_points: Numpy array with shape
                ``(number_eval_points, dim_domain)`` with the
                evaluation points.
            aligned: Whether the input points are
                the same for each sample, or an array of points per sample is
                passed.

        Returns:
            (numpy.darray): Numpy 3d array with shape
                ``(n_samples, number_eval_points, dim_codomain)`` with the
                result of the evaluation. The entry ``(i,j,k)`` will contain
                the value k-th image dimension of the i-th sample, at the
                j-th evaluation point.

        """
        from ..misc.validation import validate_evaluation_points

        eval_points = validate_evaluation_points(
            eval_points,
            aligned=aligned,
            n_samples=fdata.n_samples,
            dim_domain=fdata.dim_domain,
        )

        return self._evaluate(
            fdata=fdata,
            eval_points=eval_points,
            aligned=aligned,
        )

    def __repr__(self) -> str:
        return f"{type(self)}()"

    def __eq__(self, other: Any) -> bool:
        """Equality operator between evaluators."""
        return isinstance(other, type(self))


class EvaluateFunction(Protocol):
    """Callback of an evaluation function."""

    def __call__(
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:
        """
        Evaluate the samples at evaluation points.

        The evaluation call will receive a 2-d array with the
        evaluation points, or a 3-d array with the evaluation points per
        sample if ``aligned`` is ``False``.

        Args:
            fdata: Object to evaluate.
            eval_points: Numpy array with shape
                ``(number_eval_points, dim_domain)`` with the
                evaluation points.
            aligned: Whether the input points are
                the same for each sample, or an array of points per sample is
                passed.

        Returns:
            Numpy 3d array with shape
            ``(n_samples, number_eval_points, dim_codomain)`` with the
            result of the evaluation. The entry ``(i,j,k)`` will contain
            the value k-th image dimension of the i-th sample, at the
            j-th evaluation point.

        """
        pass


class GenericEvaluator(Evaluator):
    """Generic Evaluator.

    Generic evaluator that recibes a functions to construct the evaluator.
    The function will recieve an :class:`FData` as first argument, a numpy
    array with the eval_points and the ``aligned`` parameter.

    """

    def __init__(self, evaluate_function: EvaluateFunction) -> None:
        self.evaluate_function = evaluate_function

    def _evaluate(  # noqa: D102
        self,
        fdata: FData,
        eval_points: EvaluationPoints,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        return self.evaluate_function(fdata, eval_points, aligned=aligned)

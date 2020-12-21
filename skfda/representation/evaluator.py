"""
This module contains the structure of the evaluator.

The evaluator is the core of the FData object for extrapolation and
evaluation of FDataGrids.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from typing_extensions import Protocol


class Evaluator(ABC):
    """
    Structure of an evaluator.

    An evaluator defines how to evaluate points of a functional object, it
    can be used as extrapolator to evaluate points outside the :term:`domain`
    range or as interpolation in a :class:`FDataGrid`. The corresponding
    examples of Interpolation and Extrapolation shows the basic usage of
    this class.

    The evaluator is called internally by :func:`evaluate`.

    Should implement the methods :func:`evaluate` and
    :func:`evaluate_composed`.

    """

    @abstractmethod
    def evaluate(
        self,
        fdata: Callable[[np.ndarray], np.ndarray],
        eval_points: np.ndarray,
        *,
        aligned: bool = True,
    ) -> np.ndarray:
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
        pass

    def __repr__(self) -> str:
        return f"{type(self)}()"

    def __eq__(self, other: Any) -> bool:
        """Equality operator between evaluators."""
        return isinstance(other, type(self))


class EvaluateFunction(Protocol):
    """Callback of an evaluation function."""

    def __call__(
        self,
        fdata: Callable[[np.ndarray], np.ndarray],
        eval_points: np.ndarray,
        *,
        aligned: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the samples at evaluation points.

        The evaluation call will receive a 2-d array with the
        evaluation points, or a 3-d array with the evaluation points per
        sample if ``aligned`` is ``False``.

        Args:
            fdata: Object to evaluate.
            eval_points (numpy.ndarray): Numpy array with shape
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
        pass


class GenericEvaluator(Evaluator):
    """Generic Evaluator.

    Generic evaluator that recibes a functions to construct the evaluator.
    The function will recieve an :class:`FData` as first argument, a numpy
    array with the eval_points and the ``aligned`` parameter.

    """

    def __init__(self, evaluate_function: EvaluateFunction) -> None:
        self.evaluate_function = evaluate_function

    def evaluate(  # noqa: D102
        self,
        fdata: Callable[[np.ndarray], np.ndarray],
        eval_points: np.ndarray,
        *,
        aligned: bool = True,
    ) -> np.ndarray:
        return self.evaluate_function(fdata, eval_points, aligned=aligned)

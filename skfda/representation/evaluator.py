"""This module contains the structure of the evaluator, the core of the FData
object for extrapolation and evaluation of FDataGrids"""

from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Structure of an evaluator.

    An evaluator defines how to evaluate points of a functional object, it
    can be used as extrapolator to evaluate points outside the domain range or
    as interpolation in a :class:`FDataGrid`. The corresponding examples of
    Interpolation and Extrapolation shows the basic usage of this class.

    The evaluator is called internally by :func:`evaluate`.

    Should implement the methods :func:`evaluate` and
    :func:`evaluate_composed`.


    """
    @abstractmethod
    def evaluate(self, fdata, eval_points, *, aligned=True):
        """Evaluation method.

        Evaluates the samples at evaluation points. The evaluation
        call will receive a 2-d array with the evaluation points, or
        a 3-d array with the evaluation points per sample if ``aligned``
        is ``False``.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                ``(number_eval_points, dim_domain)`` with the
                evaluation points.

        Returns:
            (numpy.darray): Numpy 3d array with shape
                ``(n_samples, number_eval_points, dim_codomain)`` with the
                result of the evaluation. The entry ``(i,j,k)`` will contain
                the value k-th image dimension of the i-th sample, at the
                j-th evaluation point.

        """
        pass

    def __repr__(self):
        return f"{type(self)}()"

    def __eq__(self, other):
        """Equality operator between evaluators."""
        return type(self) == type(other)


class GenericEvaluator(Evaluator):
    """Generic Evaluator.

    Generic evaluator that recibes a functions to construct the evaluator.
    The function will recieve an :class:`FData` as first argument, a numpy
    array with the eval_points and the ``aligned`` parameter.

    """

    def __init__(self, evaluate_function):
        self.evaluate_function = evaluate_function

    def evaluate(self, fdata, eval_points, *, aligned=True):
        return self.evaluate_function(fdata, eval_points, aligned=aligned)

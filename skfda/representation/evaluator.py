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
    def evaluate(self, fdata, eval_points):
        """Evaluation method.

        Evaluates the samples at the same evaluation points. The evaluation
        call will receive a 2-d array with the evaluation points.
        This method is called internally by :meth:`evaluate` when the
        argument ``aligned_evaluation`` is True.

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

    @abstractmethod
    def evaluate_composed(self, fdata, eval_points):
        """Evaluation method.

        Evaluates the samples at different evaluation points. The evaluation
        call will receive a 3-d array with the evaluation points for each
        sample. This method is called internally by :func:`evaluate` when
        the argument ``aligned_evaluation`` is False.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                ``(n_samples, number_eval_points, dim_domain)`` with the
                evaluation points for each sample.

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

    Generic evaluator that recibes two functions to construct the evaluator.
    The functions will recieve an :class:`FData` as first argument, a numpy
    array with the eval_points and a named argument derivative.

    """

    def __init__(self, evaluate_func, evaluate_composed_func=None):
        self.evaluate_func = evaluate_func

        if evaluate_composed_func is None:
            self.evaluate_composed_func = evaluate_func
        else:
            self.evaluate_composed_func = evaluate_composed_func

    def evaluate(self, fdata, eval_points):
        """Evaluation method.

        Evaluates the samples at the same evaluation points. The evaluation
        call will receive a 2-d array with the evaluation points.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is True.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(len(eval_points), dim_domain)` with the evaluation points.
                Each entry represents the coordinate of a point.

        Returns:
            (numpy.darray): Numpy 3-d array with shape `(n_samples,
                len(eval_points), dim_codomain)` with the result of the
                evaluation. The entry (i,j,k) will contain the value k-th
                image dimension of the i-th sample, at the j-th evaluation
                point.

        """
        return self.evaluate_func(fdata, eval_points)

    def evaluate_composed(self, fdata, eval_points):
        """Evaluation method.

        Evaluates the samples at different evaluation points. The evaluation
        call will receive a 3-d array with the evaluation points for each
        sample.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is False.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(n_samples, number_eval_points, dim_domain)` with the
                 evaluation points for each sample.

        Returns:
            (numpy.darray): Numpy 3d array with shape `(n_samples,
                number_eval_points, dim_codomain)` with the result of the
                evaluation. The entry (i,j,k) will contain the value k-th image
                dimension of the i-th sample, at the j-th evaluation point.

        """
        return self.evaluate_composed_func(fdata, eval_points)

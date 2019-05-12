"""This module contains the structure of the evaluator, the core of the FData
object for extrapolation and evaluation of FDataGrids"""

from abc import ABC, abstractmethod


class EvaluatorConstructor(ABC):
    """Constructor of an evaluator.

    A constructor builds an Evaluator from a :class:`FData`, which is
    used to the evaluation in the functional data object.

    The evaluator constructor should have a method :func:`evaluator` which
    receives an fdata object and returns an :class:`Evaluator`.

    """

    @abstractmethod
    def evaluator(self, fdata):
        """Construct an evaluator.

        Builds the evaluator from an functional data object.

        Args:
            fdata (:class:`FData`): Functional object where the evaluator will
                be used.

        Returns:
            (:class:`Evaluator`): Evaluator of the fdata.

        """
        pass

    def __eq__(self, other):
        """Equality operator between evaluators constructors"""
        return type(self) == type(other)


class Evaluator(ABC):
    """Structure of an evaluator.

    An evaluator defines how to evaluate points of a functional object, it
    can be used as extrapolator to evaluate points outside the domain range or
    as interpolator in a :class:`FDataGrid`. The corresponding examples of
    Interpolation and Extrapolation shows the basic usage of this class.

    The evaluator is called internally by :func:`evaluate`.

    Should implement the methods :func:`evaluate` and :func:`evaluate_composed`.


    """
    @abstractmethod
    def evaluate(self, eval_points, *, derivative=0):
        """Evaluation method.

        Evaluates the samples at the same evaluation points. The evaluation call
        will receive a 2-d array with the evaluation points.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is True.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(len(eval_points), ndim_domain)` with the evaluation points.
                Each entry represents the coordinate of a point.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Numpy 3-d array with shape `(n_samples,
                len(eval_points), ndim_image)` with the result of the
                evaluation. The entry (i,j,k) will contain the value k-th
                image dimension of the i-th sample, at the j-th evaluation
                point.

        """
        pass

    @abstractmethod
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
        pass


class GenericEvaluator(Evaluator):
    """Generic Evaluator.

    Generic evaluator that recibes two functions to construct the evaluator.
    The functions will recieve an :class:`FData` as first argument, a numpy
    array with the eval_points and a named argument derivative.

    """

    def __init__(self, fdata, evaluate_func, evaluate_composed_func=None):
        self.fdata = fdata
        self.evaluate_func = evaluate_func

        if evaluate_composed_func is None:
            self.evaluate_composed_func = evaluate_func
        else:
            self.evaluate_composed_func = evaluate_composed_func

    def evaluate(self, eval_points, *, derivative=0):
        """Evaluation method.

        Evaluates the samples at the same evaluation points. The evaluation call
        will receive a 2-d array with the evaluation points.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is True.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(len(eval_points), ndim_domain)` with the evaluation points.
                Each entry represents the coordinate of a point.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Numpy 3-d array with shape `(n_samples,
                len(eval_points), ndim_image)` with the result of the
                evaluation. The entry (i,j,k) will contain the value k-th
                image dimension of the i-th sample, at the j-th evaluation
                point.

        """
        return self.evaluate_func(self.fdata, eval_points,
                                  derivative=derivative)

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
        return self.evaluate_composed_func(self.fdata, eval_points,
                                           derivative=derivative)

from collections.abc import Iterable
import itertools
from skfda.misc.operators import gramian_matrix, Identity

import scipy.linalg
from sklearn.base import BaseEstimator

import numpy as np


class TikhonovRegularization(BaseEstimator):
    r"""
    Implements Tikhonov regularization.

    The penalization term in this type of regularization is the square of the
    :math:`L_2` (Euclidean) norm of a linear operator applied to the function
    or vector

    .. math::
            \lambda \| \Gamma x \|_2^2

    where :math:`\Gamma` is the so called Tikhonov operator
    (matrix for finite vectors) and :math:`\lambda` is a positive real number.

    This linear operator can be an arbitrary Python callable that correspond
    to a linear transformation. However, the
    :doc:`operators </modules/misc/operators>` module
    provides several common linear operators.

    Parameters:
        linear_operator: linear operator used for regularization.
        regularization_parameter: scaling parameter (:math:`\lambda`) of the
                                  penalization.

    Examples:

        Construct a regularization that penalizes the second derivative,
        which is a measure of the curvature of the function.

        >>> from skfda.misc.regularization import TikhonovRegularization
        >>> from skfda.misc.operators import LinearDifferentialOperator
        >>>
        >>> regularization = TikhonovRegularization(
        ...                     LinearDifferentialOperator(2))

        Construct a regularization that penalizes the identity operator,
        that is, completely equivalent to the :math:`L_2` regularization (
        :class:`L2Regularization`).

        >>> from skfda.misc.regularization import TikhonovRegularization
        >>> from skfda.misc.operators import Identity
        >>>
        >>> regularization = TikhonovRegularization(Identity())

        Construct a regularization that penalizes the difference between
        the points :math:`f(1)` and :math:`f(0)` of a function :math:`f`.

        >>> from skfda.misc.regularization import TikhonovRegularization
        >>>
        >>> regularization = TikhonovRegularization(lambda x: x(1) - x(0))

        Construct a regularization that penalizes the harmonic acceleration
        operator :math:`Lf = \omega^2 D f + D^3 f`, that, when the
        regularization parameter is large, forces the function to be
        :math:`f(t) = c_1 + c_2 \sin \omega t + c_3 \cos \omega t`, where
        :math:`\omega` is the angular frequency. This is useful for some
        periodic functions.

        >>> from skfda.misc.regularization import TikhonovRegularization
        >>> from skfda.misc.operators import LinearDifferentialOperator
        >>> import numpy as np
        >>>
        >>> period = 1
        >>> w = 2 * np.pi / period
        >>> regularization = TikhonovRegularization(
        ...                     LinearDifferentialOperator([0, w**2, 0, 1]))

    """

    def __init__(self, linear_operator,
                 *, regularization_parameter=1):
        self.linear_operator = linear_operator
        self.regularization_parameter = regularization_parameter

    def penalty_matrix(self, basis):
        r"""
        Return a penalty matrix for ordinary least squares.

        """
        return self.regularization_parameter * gramian_matrix(
            self.linear_operator, basis)


class L2Regularization(TikhonovRegularization):
    r"""
    Implements :math:`L_2` regularization.

    The penalization term in this type of regularization is the square of the
    :math:`L_2` (Euclidean) norm of the function or vector

    .. math::
            \lambda \| x \|_2^2

    where :math:`\lambda` is a positive real number.

    This is equivalent to Tikhonov regularization (
    :class:`TikhonovRegularization`) using the identity operator (
    :class:`Identity`).

    Parameters:
        regularization_parameter: scaling parameter (:math:`\lambda`) of the
                                  penalization.

    """

    def __init__(self, *, regularization_parameter=1):
        return super().__init__(
            linear_operator=Identity(),
            regularization_parameter=regularization_parameter)


def compute_penalty_matrix(basis_iterable, regularization_parameter,
                           regularization):
    """
    Computes the regularization matrix for a linear differential operator.

    X can be a list of mixed data.

    """
    # If there is no regularization, return 0 and rely on broadcasting
    if regularization_parameter == 0 or regularization is None:
        return 0

    # Compute penalty matrix if not provided
    if not isinstance(regularization, Iterable):
        regularization = (regularization,)

    if not isinstance(regularization_parameter, Iterable):
        regularization_parameter = itertools.repeat(
            regularization_parameter)

    penalty_blocks = [
        np.zeros((len(b), len(b))) if r is None else
        a * r.penalty_matrix(b)
        for b, r, a in zip(basis_iterable, regularization,
                           regularization_parameter)]
    penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

    return penalty_matrix

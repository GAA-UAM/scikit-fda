from collections.abc import Iterable
import itertools
from skfda.misc.operators import gramian_matrix, Identity

import scipy.linalg
from sklearn.base import BaseEstimator

import numpy as np

from ..operators._operators import get_n_basis


class TikhonovRegularization(BaseEstimator):
    r"""
    Implements Tikhonov regularization.

    The penalization term in this type of regularization is

    .. math::
            \| \Gamma x \|_2^2

    where :math:`\Gamma``is the so called Tikhonov operator
    (matrix for finite vectors).

    Parameters:
        linear_operator: linear operator used for regularization.
        regularization_parameter: scaling parameter of the penalization.

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
    Implements Tikhonov regularization.

    This is equivalent to Tikhonov regularization using the identity operator.

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
        np.zeros((get_n_basis(b), get_n_basis(b))) if r is None else
        a * r.penalty_matrix(b)
        for b, r, a in zip(basis_iterable, regularization,
                           regularization_parameter)]
    penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

    return penalty_matrix

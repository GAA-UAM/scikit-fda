from collections.abc import Iterable
import itertools
from skfda.misc.operators import gramian_matrix

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
        operator: linear operator used for regularization.

    """

    def __init__(self, linear_operator):
        self.linear_operator = linear_operator

    def penalty_matrix(self, basis):
        r"""
        Return a penalty matrix for ordinary least squares.

        """
        return gramian_matrix(self.linear_operator, basis)


def compute_penalty_matrix(basis_iterable, regularization_parameter,
                           regularization, penalty_matrix):
    """
    Computes the regularization matrix for a linear differential operator.

    X can be a list of mixed data.

    """
    # If there is no regularization, return 0 and rely on broadcasting
    if regularization_parameter == 0:
        return 0

    # Compute penalty matrix if not provided
    if penalty_matrix is None:

        if regularization is None:
            raise ValueError("The regularization parameter is "
                             f"{regularization_parameter} != 0 "
                             "and no regularization is specified")

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

import abc
from collections.abc import Iterable
import itertools

import scipy.linalg

from ..._utils._coefficients import CoefficientInfo


class Regularization(abc.ABC):
    """
    Abstract base class for different kinds of regularization.

    """

    @abc.abstractmethod
    def penalty_matrix(self, coef_info):
        r"""Return a penalty matrix given the coefficient information.

        """
        pass


def _convert_regularization(regularization):
    from ._linear_diff_op_regularization import (
        LinearDifferentialOperatorRegularization)

    # Convert to linear differential operator if necessary
    if regularization is None:
        regularization = LinearDifferentialOperatorRegularization(2)
    elif not isinstance(regularization, Regularization):
        regularization = LinearDifferentialOperatorRegularization(
            regularization)

    return regularization


def compute_penalty_matrix(coef_info, regularization_parameter,
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

        if isinstance(coef_info, Iterable):

            if not isinstance(regularization, Iterable):
                regularization = itertools.repeat(regularization)

            if not isinstance(regularization_parameter, Iterable):
                regularization_parameter = itertools.repeat(
                    regularization_parameter)

            penalty_blocks = [
                a * _convert_regularization(r).penalty_matrix(c)
                for c, r, a in zip(coef_info, regularization,
                                   regularization_parameter)]
            penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

        else:

            regularization = _convert_regularization(regularization)
            penalty_matrix = regularization.penalty_matrix(coef_info)
            penalty_matrix *= regularization_parameter

    return penalty_matrix

import abc

import scipy.linalg

import numpy as np
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


def compute_penalty_matrix(coef_info, regularization_parameter,
                           regularization, penalty_matrix):
    """
    Computes the regularization matrix for a linear differential operator.

    X can be a list of mixed data.
    """
    from ._linear_diff_op_regularization import (
        LinearDifferentialOperatorRegularization)

    # If there is no regularization, return 0 and rely on broadcasting
    if regularization_parameter == 0:
        return 0

    # Compute penalty matrix if not provided
    if penalty_matrix is None:

        # Convert the linear differential operator if necessary
        if regularization is None:
            regularization = LinearDifferentialOperatorRegularization(2)
        elif not isinstance(regularization, Regularization):
            regularization = LinearDifferentialOperatorRegularization(
                regularization)

        if isinstance(coef_info, CoefficientInfo):
            penalty_matrix = regularization.penalty_matrix(coef_info)
        else:
            # If X and basis are lists

            penalty_blocks = [regularization.penalty_matrix(c)
                              for c in coef_info]
            penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

    return regularization_parameter * penalty_matrix

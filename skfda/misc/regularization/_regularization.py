import abc

import scipy.linalg

import numpy as np
from ..._utils._coefficients import CoefficientInfo


class Regularization(abc.ABC):
    """
    Abstract base class for different kinds of regularization.

    """

    @abc.abstractmethod
    def penalty_matrix(self, basis):
        r"""Return a penalty matrix given a basis.

        """
        pass


def _apply_regularization(X, coef_info, regularization: Regularization):
    """
    Apply the lfd to a single data type.
    """

    if isinstance(X, np.ndarray):
        # Multivariate objects have no penalty
        return np.zeros((X.shape[1], X.shape[1]))

    else:
        return regularization.penalty_matrix(coef_info.basis)


def compute_penalty_matrix(X, coef_info, regularization_parameter,
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
            penalty_matrix = _apply_regularization(
                X, coef_info, regularization)
        else:
            # If X and basis are lists

            penalty_blocks = [_apply_regularization(x, c, regularization)
                              for x, c in zip(X, coef_info)]
            penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

    return regularization_parameter * penalty_matrix

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

        if regularization is None:
            raise ValueError("The regularization parameter is "
                             f"{regularization_parameter} != 0 "
                             "and no regularization is specified")

        if isinstance(coef_info, Iterable):

            if not isinstance(regularization, Iterable):
                regularization = (regularization,)

            if not isinstance(regularization_parameter, Iterable):
                regularization_parameter = itertools.repeat(
                    regularization_parameter)

            penalty_blocks = [
                0 if r is None else
                a * r.penalty_matrix(c)
                for c, r, a in zip(coef_info, regularization,
                                   regularization_parameter)]
            penalty_matrix = scipy.linalg.block_diag(*penalty_blocks)

        else:

            penalty_matrix = regularization.penalty_matrix(coef_info)
            penalty_matrix *= regularization_parameter

    return penalty_matrix

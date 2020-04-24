from functools import singledispatch

import numpy as np

from ..._utils._coefficients import CoefficientInfo, CoefficientInfoFDataBasis
from ._regularization import Regularization


@singledispatch
def penalty_matrix_coef_info(coef_info: CoefficientInfo,
                             regularization):
    """
    Return a penalty matrix given the coefficient information.

    This method is a singledispatch method that provides an
    implementation of the computation of the penalty matrix
    for a particular coefficient type.
    """
    return np.zeros((coef_info.shape[0], coef_info.shape[0]))


class EndpointsDifferenceRegularization(Regularization):
    """
    Regularization penalizing the difference of the functions
    endpoints.

    """

    penalty_matrix_coef_info = penalty_matrix_coef_info

    def penalty_matrix(self, coef_info):
        return penalty_matrix_coef_info(coef_info, self)


@EndpointsDifferenceRegularization.penalty_matrix_coef_info.register(
    CoefficientInfoFDataBasis)
def penalty_matrix_coef_info_fdatabasis(
        coef_info: CoefficientInfoFDataBasis,
        regularization: EndpointsDifferenceRegularization):

    evaluate_first = coef_info.basis(coef_info.basis.domain_range[0][0])
    evaluate_last = coef_info.basis(coef_info.basis.domain_range[0][1])
    evaluate_diff = evaluate_last - evaluate_first

    return evaluate_diff @ evaluate_diff.T

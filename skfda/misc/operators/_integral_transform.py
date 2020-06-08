import scipy.integrate

import numpy as np

from ...representation import FData
from ._operators import Operator, get_n_basis, gramian_matrix_optimization


class IntegralTransform(Operator):
    """Integral operator.



    Attributes:
        kernel_function (callable):  Kernel function corresponding to
                        the operator.

    """

    def __init__(self, kernel_function):
        self.kernel_function = kernel_function

    def __call__(self, f):

        def evaluate_covariance(points):

            def integral_body(integration_var):
                return (f(integration_var) *
                        self.kernel_function(integration_var, points))

            domain_range = f.domain_range[0]

            return scipy.integrate.quad_vec(
                integral_body, domain_range[0], domain_range[1])[0]

        return evaluate_covariance

from __future__ import annotations

from typing import Callable

import numpy as np

import scipy.integrate

from ...representation import FData
from ._operators import Operator


class IntegralTransform(Operator[FData, Callable[[np.ndarray], np.ndarray]]):
    """Integral operator.

    Parameters:
        kernel_function:  Kernel function corresponding to the operator.

    """

    def __init__(
        self,
        kernel_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.kernel_function = kernel_function

    def __call__(  # noqa: D102
        self,
        f: FData,
    ) -> Callable[[np.ndarray], np.ndarray]:

        def evaluate_covariance(  # noqa: WPS430
            points: np.ndarray,
        ) -> np.ndarray:

            def integral_body(  # noqa: WPS430
                integration_var: np.ndarray,
            ) -> np.ndarray:
                return (
                    f(integration_var)
                    * self.kernel_function(integration_var, points)
                )

            domain_range = f.domain_range[0]

            return scipy.integrate.quad_vec(
                integral_body,
                domain_range[0],
                domain_range[1],
            )[0]

        return evaluate_covariance

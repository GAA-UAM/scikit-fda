from __future__ import annotations

from typing import Callable

import scipy.integrate

from ...representation import FData
from ...typing._numpy import NDArrayFloat
from ._operators import Operator


class IntegralTransform(
    Operator[FData, Callable[[NDArrayFloat], NDArrayFloat]],
):
    """Integral operator.

    Parameters:
        kernel_function:  Kernel function corresponding to the operator.

    """

    def __init__(
        self,
        kernel_function: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
    ) -> None:
        self.kernel_function = kernel_function

    def __call__(  # noqa: D102
        self,
        f: FData,
    ) -> Callable[[NDArrayFloat], NDArrayFloat]:

        def evaluate_covariance(  # noqa: WPS430
            points: NDArrayFloat,
        ) -> NDArrayFloat:

            def integral_body(  # noqa: WPS430
                integration_var: NDArrayFloat,
            ) -> NDArrayFloat:
                return (
                    f(integration_var)
                    * self.kernel_function(integration_var, points)
                )

            domain_range = f.domain_range[0]

            return scipy.integrate.quad_vec(  # type: ignore[no-any-return]
                integral_body,
                domain_range[0],
                domain_range[1],
            )[0]

        return evaluate_covariance

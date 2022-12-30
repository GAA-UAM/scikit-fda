from __future__ import annotations

import itertools
import warnings
from typing import Any, Generic, Iterable, Optional, Union

import numpy as np
import scipy.linalg

from ..._utils._sklearn_adapter import BaseEstimator
from ...representation import FData
from ...representation.basis import Basis
from ...typing._numpy import NDArrayFloat
from ..operators import Identity, Operator, gram_matrix
from ..operators._operators import OperatorInput


class L2Regularization(
    BaseEstimator,
    Generic[OperatorInput],
):
    r"""
    Implements :math:`L_2` (Tikhonov) regularization.

    The penalization term in this type of regularization is the square of the
    :math:`L_2` (Euclidean) norm of a linear operator applied to the function
    or vector

    .. math::
            \lambda \| \Gamma x \|_2^2

    where :math:`\Gamma` is the so called Tikhonov operator
    (matrix for finite vectors) and :math:`\lambda` is a positive real number.

    This linear operator can be an arbitrary Python callable that correspond
    to a linear transformation. However, the
    :doc:`operators </modules/misc/operators>` module
    provides several common linear operators.

    Parameters:
        linear_operator: Linear operator used for regularization. By default
            the second derivative, which is related with the function
            curvature, is penalized.
        regularization_parameter: Scaling parameter (:math:`\lambda`) of the
            penalization.

    Examples:
        Construct a regularization that penalizes the second derivative,
        which is a measure of the curvature of the function.

        >>> from skfda.misc.regularization import L2Regularization
        >>> from skfda.misc.operators import LinearDifferentialOperator
        >>>
        >>> regularization = L2Regularization(
        ...                     LinearDifferentialOperator(2),
        ... )

        By default the regularization penalizes the identity operator:

        >>> regularization = L2Regularization()

        Construct a regularization that penalizes the difference between
        the points :math:`f(1)` and :math:`f(0)` of a function :math:`f`.

        >>> regularization = L2Regularization(lambda x: x(1) - x(0))

        Construct a regularization that penalizes the harmonic acceleration
        operator :math:`Lf = \omega^2 D f + D^3 f`, that, when the
        regularization parameter is large, forces the function to be
        :math:`f(t) = c_1 + c_2 \sin \omega t + c_3 \cos \omega t`, where
        :math:`\omega` is the angular frequency. This is useful for some
        periodic functions.

        >>> import numpy as np
        >>>
        >>> period = 1
        >>> w = 2 * np.pi / period
        >>> regularization = L2Regularization(
        ...                     LinearDifferentialOperator([0, w**2, 0, 1]),
        ... )

    """

    def __init__(
        self,
        linear_operator: Optional[Operator[OperatorInput, Any]] = None,
        *,
        regularization_parameter: float = 1,
    ) -> None:
        self.linear_operator = linear_operator
        self.regularization_parameter = regularization_parameter

    def penalty_matrix(
        self,
        basis: OperatorInput,
    ) -> NDArrayFloat:
        """Return a penalty matrix for ordinary least squares."""
        linear_operator = (
            Identity()
            if self.linear_operator is None
            else self.linear_operator
        )

        return self.regularization_parameter * gram_matrix(
            linear_operator,
            basis,
        )


class TikhonovRegularization(
    L2Regularization[OperatorInput],
):

    def __init__(
        self,
        linear_operator: Optional[Operator[OperatorInput, Any]] = None,
        *,
        regularization_parameter: float = 1,
    ) -> None:

        warnings.warn(
            "Class TikhonovRegularization is deprecated. Use class "
            "L2Regularization instead.",
            DeprecationWarning,
        )

        return super().__init__(
            linear_operator=linear_operator,
            regularization_parameter=regularization_parameter,
        )


BasisTypes = Union[np.ndarray, FData, Basis]
Regularization = L2Regularization[Any]
RegularizationLike = Union[
    None,
    Regularization,
    Iterable[Optional[Regularization]],
]


def compute_penalty_matrix(
    basis_iterable: Iterable[BasisTypes],
    regularization_parameter: Union[float, Iterable[float]],
    regularization: RegularizationLike,
) -> Optional[NDArrayFloat]:
    """
    Compute the regularization matrix for a linear differential operator.

    X can be a list of mixed data.

    """
    # If there is no regularization, return 0 and rely on broadcasting
    if regularization_parameter == 0 or regularization is None:
        return None

    # Compute penalty matrix if not provided
    if not isinstance(regularization, Iterable):
        regularization = (regularization,)

    if not isinstance(regularization_parameter, Iterable):
        regularization_parameter = itertools.repeat(
            regularization_parameter,
        )

    penalty_blocks = [
        np.zeros((len(b), len(b))) if r is None else
        a * r.penalty_matrix(b)
        for b, r, a in zip(
            basis_iterable,
            regularization,
            regularization_parameter,
        )]

    return scipy.linalg.block_diag(  # type: ignore[no-any-return]
        *penalty_blocks,
    )

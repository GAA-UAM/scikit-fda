import warnings
from typing import Optional, Tuple, TypeVar

import numpy as np

from ...typing._base import DomainRangeLike
from ...typing._numpy import NDArrayFloat
from ._basis import Basis

T = TypeVar("T", bound="ConstantBasis")


class ConstantBasis(Basis):
    """Constant basis.

    Basis for constant functions

    Parameters:
        domain_range: The :term:`domain range` over which the basis can be
            evaluated.

    Examples:
        Defines a contant base over the interval :math:`[0, 5]` consisting
        on the constant function 1 on :math:`[0, 5]`.

        >>> bs_cons = ConstantBasis((0,5))

    """

    def __init__(self, domain_range: Optional[DomainRangeLike] = None) -> None:
        """Constant basis constructor."""
        super().__init__(domain_range=domain_range, n_basis=1)

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:
        return np.ones((1, len(eval_points)))

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        return (
            (self.copy(), coefs.copy())
            if order == 0
            else (self.copy(), np.zeros(coefs.shape))
        )

    def _gram_matrix(self) -> NDArrayFloat:
        return np.array(
            [[self.domain_range[0][1] - self.domain_range[0][0]]],
        )

    def _to_R(self) -> str:  # noqa: N802
        drange = self.domain_range[0]
        drange_str = f"c({str(drange[0])}, {str(drange[1])})"
        return f"create.constant.basis(rangeval = {drange_str})"


class Constant(ConstantBasis):
    """Constant basis.

    Basis for constant functions

    .. deprecated:: 0.8
        Use :class:`~skfda.representation.basis.ConstantBasis` instead.

    Parameters:
        domain_range: The :term:`domain range` over which the basis can be
            evaluated.

    Examples:
        Defines a contant base over the interval :math:`[0, 5]` consisting
        on the constant function 1 on :math:`[0, 5]`.

        >>> bs_cons = Constant((0,5))

    """

    def __init__(self, domain_range: Optional[DomainRangeLike] = None) -> None:
        """Constant basis constructor."""
        warnings.warn(
            "The Constant class is deprecated. Use ConstantBasis instead.",
            DeprecationWarning,
        )
        super().__init__(domain_range=domain_range)

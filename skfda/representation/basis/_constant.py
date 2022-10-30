import warnings
from typing import Optional

from ...typing._base import DomainRangeLike
from ._constant_basis import ConstantBasis


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

        >>> bs_cons = ConstantBasis((0,5))

    """

    def __init__(self, domain_range: Optional[DomainRangeLike] = None) -> None:
        """Constant basis constructor."""
        warnings.warn(
            "The Constant class is deprecated. Use "
            "ConstantBasis instead.",
            DeprecationWarning,
        )
        super().__init__(domain_range=domain_range)

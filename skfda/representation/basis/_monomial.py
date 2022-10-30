import warnings
from typing import Optional

from ...typing._base import DomainRangeLike
from ._monomial_basis import MonomialBasis


class Monomial(MonomialBasis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

    .. deprecated:: 0.8
        Use :class:`~skfda.representation.basis.MonomialBasis` instead.

    Attributes:
        domain_range: a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis: number of functions in the basis.

    Examples:
        Defines a monomial base over the interval :math:`[0, 5]` consisting
        on the first 3 powers of :math:`t`: :math:`1, t, t^2`.

        >>> bs_mon = MonomialBasis(domain_range=(0,5), n_basis=3)

        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> bs_mon([0., 1., 2.])
        array([[[ 1.],
                [ 1.],
                [ 1.]],
               [[ 0.],
                [ 1.],
                [ 2.]],
               [[ 0.],
                [ 1.],
                [ 4.]]])

        And also evaluates its derivatives

        >>> deriv = bs_mon.derivative()
        >>> deriv([0, 1, 2])
        array([[[ 0.],
                [ 0.],
                [ 0.]],
               [[ 1.],
                [ 1.],
                [ 1.]],
               [[ 0.],
                [ 2.],
                [ 4.]]])
        >>> deriv2 = bs_mon.derivative(order=2)
        >>> deriv2([0, 1, 2])
        array([[[ 0.],
                [ 0.],
                [ 0.]],
               [[ 0.],
                [ 0.],
                [ 0.]],
               [[ 2.],
                [ 2.],
                [ 2.]]])
    """

    def __init__(
        self,
        *,
        domain_range: Optional[DomainRangeLike] = None,
        n_basis: int = 1,
    ):
        super().__init__(
            domain_range=domain_range,
            n_basis=n_basis,
        )
        warnings.warn(
            "The BSplines class is deprecated. Use "
            "BSplineBasis instead.",
            DeprecationWarning,
        )

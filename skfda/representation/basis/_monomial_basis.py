import warnings
from typing import Optional, Tuple, TypeVar

import numpy as np
import scipy.linalg

from ...typing._base import DomainRangeLike
from ...typing._numpy import NDArrayFloat
from ._basis import Basis

T = TypeVar("T", bound="MonomialBasis")


class MonomialBasis(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

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

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:

        # Input is scalar
        eval_points = eval_points[..., 0]

        exps = np.arange(self.n_basis)
        raised = np.power.outer(eval_points, exps)

        return raised.T

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:
        if order >= self.n_basis:
            return (
                type(self)(domain_range=self.domain_range, n_basis=1),
                np.zeros((len(coefs), 1)),
            )

        return (
            type(self)(
                domain_range=self.domain_range,
                n_basis=self.n_basis - order,
            ),
            np.array([np.polyder(x[::-1], order)[::-1] for x in coefs]),
        )

    def _gram_matrix(self) -> NDArrayFloat:
        integral_coefs = np.polyint(np.ones(2 * self.n_basis - 1))

        # We obtain the powers of both extremes in the domain range
        power_domain_limits = np.vander(
            self.domain_range[0],
            2 * self.n_basis,
        )

        # Subtract the powers (Barrow's rule)
        power_domain_limits_diff = (
            power_domain_limits[1] - power_domain_limits[0]
        )

        # Multiply the constants that appear in the integration
        evaluated_points = integral_coefs * power_domain_limits_diff

        # Order the powers, lower to higher, discarding the constant
        # (it does not appear in the integral)
        ordered_evaluated_points = evaluated_points[-2::-1]

        # Build the matrix
        return scipy.linalg.hankel(  # type: ignore[no-any-return]
            ordered_evaluated_points[: self.n_basis],
            ordered_evaluated_points[self.n_basis - 1:],
        )

    def _to_R(self) -> str:  # noqa: N802
        drange = self.domain_range[0]
        rangeval = f"c({drange[0]}, {drange[1]})"
        return (
            f"create.monomial.basis("
            f"rangeval = {rangeval}, "
            f"nbasis = {self.n_basis})"
        )


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

        >>> bs_mon = Monomial(domain_range=(0,5), n_basis=3)

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
            "The Monomial class is deprecated. Use MonomialBasis instead.",
            DeprecationWarning,
        )

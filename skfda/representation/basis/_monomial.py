import scipy.linalg

import numpy as np

from ..._utils import _same_domain
from ._basis import Basis


class Monomial(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis (int): number of functions in the basis.

    Examples:
        Defines a monomial base over the interval :math:`[0, 5]` consisting
        on the first 3 powers of :math:`t`: :math:`1, t, t^2`.

        >>> bs_mon = Monomial((0,5), n_basis=3)

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

    def _evaluate(self, eval_points):

        # Input is scalar
        eval_points = eval_points[..., 0]

        exps = np.arange(self.n_basis)
        raised = np.power.outer(eval_points, exps)

        return raised.T

    def _derivative_basis_and_coefs(self, coefs, order=1):
        if order >= self.n_basis:
            return (Monomial(self.domain_range, 1),
                    np.zeros((len(coefs), 1)))
        else:
            return (Monomial(self.domain_range, self.n_basis - order),
                    np.array([np.polyder(x[::-1], order)[::-1]
                              for x in coefs]))

    def _gram_matrix(self):
        integral_coefs = np.polyint(np.ones(2 * self.n_basis - 1))

        # We obtain the powers of both extremes in the domain range
        power_domain_limits = np.vander(
            self.domain_range[0], 2 * self.n_basis)

        # Subtract the powers (Barrow's rule)
        power_domain_limits_diff = (
            power_domain_limits[1] - power_domain_limits[0])

        # Multiply the constants that appear in the integration
        evaluated_points = integral_coefs * power_domain_limits_diff

        # Order the powers, lower to higher, discarding the constant
        # (it does not appear in the integral)
        ordered_evaluated_points = evaluated_points[-2::-1]

        # Build the matrix
        return scipy.linalg.hankel(
            ordered_evaluated_points[:self.n_basis],
            ordered_evaluated_points[self.n_basis - 1:])

    def basis_of_product(self, other):
        """Multiplication of a Monomial Basis with other Basis"""
        if not _same_domain(self, other):
            raise ValueError("Ranges are not equal.")

        if isinstance(other, Monomial):
            return Monomial(self.domain_range, self.n_basis + other.n_basis)

        return other.rbasis_of_product(self)

    def rbasis_of_product(self, other):
        """Multiplication of a Monomial Basis with other Basis"""
        return Basis.default_basis_of_product(self, other)

    def _to_R(self):
        drange = self.domain_range[0]
        return "create.monomial.basis(rangeval = c(" + str(drange[0]) + "," +\
               str(drange[1]) + "), nbasis = " + str(self.n_basis) + ")"

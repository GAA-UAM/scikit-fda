import numpy as np

from ..._utils import _same_domain
from ._basis import Basis


class Constant(Basis):
    """Constant basis.

    Basis for constant functions

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.

    Examples:
        Defines a contant base over the interval :math:`[0, 5]` consisting
        on the constant function 1 on :math:`[0, 5]`.

        >>> bs_cons = Constant((0,5))

    """

    def __init__(self, domain_range=None):
        """Constant basis constructor.

        Args:
            domain_range (tuple): Tuple defining the domain over which the
            function is defined.

        """
        super().__init__(domain_range=domain_range, n_basis=1)

    def _evaluate(self, eval_points):
        return np.ones((1, len(eval_points)))

    def _derivative_basis_and_coefs(self, coefs, order=1):
        return ((self.copy(), coefs.copy()) if order == 0
                else (self.copy(), np.zeros(coefs.shape)))

    def _gram_matrix(self):
        return np.array([[self.domain_range[0][1] -
                          self.domain_range[0][0]]])

    def _to_R(self):
        drange = self.domain_range[0]
        return "create.constant.basis(rangeval = c(" + str(drange[0]) + "," +\
               str(drange[1]) + "))"


import warnings
from typing import Iterable

from ._basis import Basis
from ._tensor_basis import TensorBasis


class Tensor(TensorBasis):
    r"""Tensor basis.

    Basis for multivariate functions constructed as a tensor product of
    :math:`\mathbb{R} \to \mathbb{R}` bases.

    .. deprecated:: 0.8
        Use :class:`~skfda.representation.basis.TensorBasis` instead.

    Attributes:
        domain_range (tuple): a tuple of length ``dim_domain`` containing
            the range of input values for each dimension.
        n_basis (int): number of functions in the basis.

    Examples:
        Defines a tensor basis over the interval :math:`[0, 5] \times [0, 3]`
        consisting on the functions

        .. math::

            1, v, u, uv, u^2, u^2v

        >>> from skfda.representation.basis import TensorBasis, MonomialBasis
        >>>
        >>> basis_x = MonomialBasis(domain_range=(0,5), n_basis=3)
        >>> basis_y = MonomialBasis(domain_range=(0,3), n_basis=2)
        >>>
        >>> basis = TensorBasis([basis_x, basis_y])


        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> basis([(0., 2.), (3., 0), (2., 3.)])
        array([[[  1.],
                [  1.],
                [  1.]],
               [[  2.],
                [  0.],
                [  3.]],
               [[  0.],
                [  3.],
                [  2.]],
               [[  0.],
                [  0.],
                [  6.]],
               [[  0.],
                [  9.],
                [  4.]],
               [[  0.],
                [  0.],
                [ 12.]]])

    """

    def __init__(self, basis_list: Iterable[Basis]):

        super().__init__(
            basis_list=basis_list,
        )
        warnings.warn(
            "The Tensor class is deprecated. Use "
            "TensorBasis instead.",
            DeprecationWarning,
        )

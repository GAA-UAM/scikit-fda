from __future__ import annotations

import warnings
from typing import Iterable

from ._basis import Basis
from ._vector_basis import VectorValuedBasis


class VectorValued(VectorValuedBasis):
    r"""Vector-valued basis.

    Basis for :term:`vector-valued functions <vector-valued function>`
    constructed from scalar-valued bases.

    For each dimension in the :term:`codomain`, it uses a scalar-valued basis
    multiplying each basis by the corresponding unitary vector.

    .. deprecated:: 0.8
        Use :class:`~skfda.representation.basis.VectorValuedBasis` instead.

    Attributes:
        domain_range (tuple): a tuple of length ``dim_domain`` containing
            the range of input values for each dimension.
        n_basis (int): number of functions in the basis.

    Examples:
        Defines a vector-valued base over the interval :math:`[0, 5]`
        consisting on the functions

        .. math::

            1 \vec{i}, t \vec{i}, t^2 \vec{i}, 1 \vec{j}, t \vec{j}

        >>> from skfda.representation.basis import VectorValuedBasis
        >>> from skfda.representation.basis import MonomialBasis
        >>>
        >>> basis_x = MonomialBasis(domain_range=(0,5), n_basis=3)
        >>> basis_y = MonomialBasis(domain_range=(0,5), n_basis=2)
        >>>
        >>> basis = VectorValuedBasis([basis_x, basis_y])


        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> basis([0., 1., 2.])
        array([[[ 1.,  0.],
                [ 1.,  0.],
                [ 1.,  0.]],
               [[ 0.,  0.],
                [ 1.,  0.],
                [ 2.,  0.]],
               [[ 0.,  0.],
                [ 1.,  0.],
                [ 4.,  0.]],
               [[ 0.,  1.],
                [ 0.,  1.],
                [ 0.,  1.]],
               [[ 0.,  0.],
                [ 0.,  1.],
                [ 0.,  2.]]])

    """

    def __init__(self, basis_list: Iterable[Basis]) -> None:
        super().__init__(
            basis_list=basis_list,
        )

        warnings.warn(
            "The VectorValued class is deprecated. Use "
            "VectorValuedBasis instead.",
            DeprecationWarning,
        )

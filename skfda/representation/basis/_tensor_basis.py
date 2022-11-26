import itertools
import math
import warnings
from typing import Any, Iterable, Tuple

import numpy as np

from ...typing._numpy import NDArrayFloat
from ._basis import Basis


class TensorBasis(Basis):
    r"""Tensor basis.

    Basis for multivariate functions constructed as a tensor product of
    :math:`\mathbb{R} \to \mathbb{R}` bases.


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

        self._basis_list = tuple(basis_list)

        if not all(
            b.dim_domain == 1 and b.dim_codomain == 1 for b in self._basis_list
        ):
            raise ValueError(
                "The basis functions must be univariate and scalar valued",
            )

        super().__init__(
            domain_range=[b.domain_range[0] for b in basis_list],
            n_basis=math.prod([b.n_basis for b in basis_list]),
        )

    @property
    def basis_list(self) -> Tuple[Basis, ...]:
        return self._basis_list

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:

        matrix = np.zeros((self.n_basis, len(eval_points), self.dim_codomain))

        basis_evaluations = [
            b(eval_points[:, i:i + 1]) for i, b in enumerate(self.basis_list)
        ]

        for i, ev in enumerate(itertools.product(*basis_evaluations)):

            matrix[i, :, :] = np.prod(ev, axis=0)

        return matrix

    def _gram_matrix(self) -> NDArrayFloat:

        gram_matrices = [b.gram_matrix() for b in self.basis_list]

        gram = gram_matrices[0]

        for g in gram_matrices[1:]:
            n_rows = len(gram) * len(g)
            gram = np.multiply.outer(gram, g)
            gram = np.moveaxis(gram, [1, 2], [2, 1])
            gram = gram.reshape(n_rows, n_rows)

        return gram

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and self.basis_list == other.basis_list

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.basis_list))


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

        >>> from skfda.representation.basis import Tensor, Monomial
        >>>
        >>> basis_x = Monomial(domain_range=(0,5), n_basis=3)
        >>> basis_y = Monomial(domain_range=(0,3), n_basis=2)
        >>>
        >>> basis = Tensor([basis_x, basis_y])


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
            "The Tensor class is deprecated. Use TensorBasis instead.",
            DeprecationWarning,
        )

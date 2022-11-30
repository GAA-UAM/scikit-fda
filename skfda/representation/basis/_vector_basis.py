from __future__ import annotations

import warnings
from typing import Any, Iterable, Tuple, TypeVar, Union

import numpy as np
import scipy.linalg

from ...typing._numpy import NDArrayFloat
from ._basis import Basis

T = TypeVar("T", bound="VectorValuedBasis")


class VectorValuedBasis(Basis):
    r"""Vector-valued basis.

    Basis for :term:`vector-valued functions <vector-valued function>`
    constructed from scalar-valued bases.

    For each dimension in the :term:`codomain`, it uses a scalar-valued basis
    multiplying each basis by the corresponding unitary vector.

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
        from ..._utils import _same_domain

        basis_list = tuple(basis_list)

        if not all(b.dim_codomain == 1 for b in basis_list):
            raise ValueError(
                "The basis functions must be scalar valued",
            )

        if any(
            b.dim_domain != basis_list[0].dim_domain
            or not _same_domain(b, basis_list[0])
            for b in basis_list
        ):
            raise ValueError(
                "The basis must all have the same domain "
                "dimension and range",
            )

        self._basis_list = basis_list

        super().__init__(
            domain_range=basis_list[0].domain_range,
            n_basis=sum(b.n_basis for b in basis_list),
        )

    @property
    def basis_list(self) -> Tuple[Basis, ...]:
        return self._basis_list

    @property
    def dim_domain(self) -> int:
        return self.basis_list[0].dim_domain

    @property
    def dim_codomain(self) -> int:
        return len(self.basis_list)

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:
        matrix = np.zeros((self.n_basis, len(eval_points), self.dim_codomain))

        n_basis_eval = 0

        basis_evaluations = [b(eval_points) for b in self.basis_list]

        for i, ev in enumerate(basis_evaluations):

            matrix[n_basis_eval:n_basis_eval + len(ev), :, i] = ev[..., 0]
            n_basis_eval += len(ev)

        return matrix

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:

        n_basis_list = [b.n_basis for b in self.basis_list]
        indexes = np.cumsum(n_basis_list)

        coefs_per_basis = np.hsplit(coefs, indexes[:-1])

        basis_and_coefs = [
            b._derivative_basis_and_coefs(c, order=order)  # noqa: WPS437
            for b, c in zip(self.basis_list, coefs_per_basis)
        ]

        new_basis_list, new_coefs_list = zip(*basis_and_coefs)

        new_basis = type(self)(new_basis_list)
        new_coefs = np.hstack(new_coefs_list)

        return new_basis, new_coefs

    def _gram_matrix(self) -> NDArrayFloat:

        gram_matrices = [b.gram_matrix() for b in self.basis_list]

        return scipy.linalg.block_diag(  # type: ignore[no-any-return]
            *gram_matrices,
        )

    def _coordinate_nonfull(
        self,
        coefs: NDArrayFloat,
        key: Union[int, slice],
    ) -> Tuple[Basis, NDArrayFloat]:

        basis_sizes = [b.n_basis for b in self.basis_list]
        basis_indexes = np.cumsum(basis_sizes)
        coef_splits = np.split(coefs, basis_indexes[:-1], axis=1)

        new_basis = self.basis_list[key]
        if not isinstance(new_basis, Basis):
            new_basis = VectorValuedBasis(new_basis)

        new_coefs = coef_splits[key]
        if not isinstance(new_coefs, np.ndarray):
            new_coefs = np.concatenate(coef_splits[key], axis=1)

        return new_basis, new_coefs

    def __repr__(self) -> str:
        """Representation of a Basis object."""
        return f"{self.__class__.__name__}(" f"basis_list={self.basis_list})"

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and self.basis_list == other.basis_list

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.basis_list))


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

        >>> from skfda.representation.basis import VectorValued
        >>> from skfda.representation.basis import Monomial
        >>>
        >>> basis_x = Monomial(domain_range=(0,5), n_basis=3)
        >>> basis_y = Monomial(domain_range=(0,5), n_basis=2)
        >>>
        >>> basis = VectorValued([basis_x, basis_y])


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
            "The VectorValued class is deprecated. "
            "Use VectorValuedBasis instead.",
            DeprecationWarning,
        )

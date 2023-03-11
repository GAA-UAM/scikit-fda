from __future__ import annotations

import abc
from typing import Any, Callable, TypeVar, Union

import multimethod
from typing_extensions import Protocol

from ...representation import FData
from ...representation.basis import Basis
from ...typing._numpy import NDArrayFloat

InputType = Union[NDArrayFloat, FData, Basis]

OperatorInput = TypeVar(
    "OperatorInput",
    bound=InputType,
    contravariant=True,
)

OutputType = Union[NDArrayFloat, Callable[[NDArrayFloat], NDArrayFloat]]

OperatorOutput = TypeVar(
    "OperatorOutput",
    bound=OutputType,
    covariant=True,
)


class Operator(Protocol[OperatorInput, OperatorOutput]):
    """Abstract class for :term:`operators`."""

    @abc.abstractmethod
    def __call__(self, vector: OperatorInput) -> OperatorOutput:
        """Evaluate the operator."""
        pass


@multimethod.multidispatch
def gram_matrix_optimization(
    linear_operator: Any,
    basis: OperatorInput,
) -> NDArrayFloat:
    """
    Efficient implementation of gram_matrix.

    Generic function that can be subclassed for different combinations of
    operator and basis in order to provide a more efficient implementation
    for the gram matrix.
    """
    return NotImplemented


def gram_matrix_numerical(
    linear_operator: Operator[OperatorInput, OutputType],
    basis: OperatorInput,
) -> NDArrayFloat:
    """
    Return the gram matrix given a basis, computed numerically.

    This method should work for every linear operator.

    """
    from .. import inner_product_matrix

    evaluated_basis = linear_operator(basis)

    domain_range = getattr(basis, "domain_range", None)

    return inner_product_matrix(evaluated_basis, _domain_range=domain_range)


def gram_matrix(
    linear_operator: Operator[OperatorInput, OutputType],
    basis: OperatorInput,
) -> NDArrayFloat:
    r"""
    Return the gram matrix given a basis.

    The gram operator of a linear operator :math:`\Gamma` is

    .. math::
        G = \Gamma*\Gamma

    This method evaluates that gram operator in a given basis,
    which is necessary for performing Tikhonov regularization,
    among other things.

    It tries to use an optimized implementation if one is available,
    falling back to a numerical computation otherwise.

    """
    # Try to use a more efficient implementation
    matrix = gram_matrix_optimization(linear_operator, basis)
    if matrix is not NotImplemented:
        return matrix

    return gram_matrix_numerical(linear_operator, basis)


class MatrixOperator(Operator[NDArrayFloat, NDArrayFloat]):
    """Linear operator for finite spaces.

    Between finite dimensional spaces, every linear operator can be expressed
    as a product by a matrix.

    Attributes:
        matrix:  The matrix containing the linear transformation.

    """

    def __init__(self, matrix: NDArrayFloat) -> None:
        self.matrix = matrix

    def __call__(self, f: NDArrayFloat) -> NDArrayFloat:  # noqa: D102
        return self.matrix @ f

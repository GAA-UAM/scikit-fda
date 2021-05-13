from __future__ import annotations

import abc
from typing import Any, Callable, TypeVar, Union

import multimethod
import numpy as np
from typing_extensions import Protocol

from ...representation import FData
from ...representation.basis import Basis

OperatorInput = TypeVar(
    "OperatorInput",
    bound=Union[np.ndarray, FData, Basis],
    contravariant=True,
)

OutputType = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]

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
def gramian_matrix_optimization(
    linear_operator: Any,
    basis: OperatorInput,
) -> np.ndarray:
    """
    Efficient implementation of gramian_matrix.

    Generic function that can be subclassed for different combinations of
    operator and basis in order to provide a more efficient implementation
    for the gramian matrix.
    """
    return NotImplemented


def gramian_matrix_numerical(
    linear_operator: Operator[OperatorInput, OutputType],
    basis: OperatorInput,
) -> np.ndarray:
    """
    Return the gramian matrix given a basis, computed numerically.

    This method should work for every linear operator.

    """
    from .. import inner_product_matrix

    evaluated_basis = linear_operator(basis)

    domain_range = getattr(basis, "domain_range", None)

    return inner_product_matrix(evaluated_basis, _domain_range=domain_range)


def gramian_matrix(
    linear_operator: Operator[OperatorInput, OutputType],
    basis: OperatorInput,
) -> np.ndarray:
    r"""
    Return the gramian matrix given a basis.

    The gramian operator of a linear operator :math:`\Gamma` is

    .. math::
        G = \Gamma*\Gamma

    This method evaluates that gramian operator in a given basis,
    which is necessary for performing Tikhonov regularization,
    among other things.

    It tries to use an optimized implementation if one is available,
    falling back to a numerical computation otherwise.

    """
    # Try to use a more efficient implementation
    matrix = gramian_matrix_optimization(linear_operator, basis)
    if matrix is not NotImplemented:
        return matrix

    return gramian_matrix_numerical(linear_operator, basis)


class MatrixOperator(Operator[np.ndarray, np.ndarray]):
    """Linear operator for finite spaces.

    Between finite dimensional spaces, every linear operator can be expressed
    as a product by a matrix.

    Attributes:
        matrix:  The matrix containing the linear transformation.

    """

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def __call__(self, f: np.ndarray) -> np.ndarray:  # noqa: D102
        return self.matrix @ f

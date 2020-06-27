import abc

import multimethod
import scipy.integrate

import numpy as np


class Operator(abc.ABC):
    """
    Abstract class for operators (functions whose domain are functions).

    """

    @abc.abstractmethod
    def __call__(self, vector):
        pass


@multimethod.multidispatch
def gramian_matrix_optimization(linear_operator, basis):
    r"""
    Generic function that can be subclassed for different combinations of
    operator and basis in order to provide a more efficient implementation
    for the gramian matrix.
    """
    return NotImplemented


def get_n_basis(basis):
    n_basis = getattr(basis, "n_basis", None)
    if n_basis is None:
        n_basis = len(basis)

    return n_basis


def compute_triang_functional(evaluated_basis,
                              indices,
                              basis):
    def cross_product(x):
        """Multiply the two evaluations."""
        res = evaluated_basis([x])[:, 0]

        return res[indices[0]] * res[indices[1]]

    # Range of first dimension
    domain_range = basis.domain_range[0]

    # Obtain the integrals for the upper matrix
    integral = scipy.integrate.quad_vec(
        cross_product, domain_range[0], domain_range[1])[0]

    return integral[..., 0]


def compute_triang_multivariate(evaluated_basis,
                                indices,
                                basis):

    cross_product = evaluated_basis[indices[0]] * evaluated_basis[indices[1]]

    # Obtain the integrals for the upper matrix
    return np.sum(cross_product, axis=-1)


def gramian_matrix_numerical(linear_operator, basis):
    r"""
    Return the gramian matrix given a basis, computed numerically.

    This method should work for every linear operator.

    """
    n_basis = get_n_basis(basis)

    indices = np.triu_indices(n_basis)

    evaluated_basis = linear_operator(basis)
    compute_triang = (compute_triang_functional if callable(
        evaluated_basis) else compute_triang_multivariate)
    triang_vec = compute_triang(evaluated_basis, indices, basis)

    matrix = np.empty((n_basis, n_basis))

    # Set upper matrix
    matrix[indices] = triang_vec

    # Set lower matrix
    matrix[(indices[1], indices[0])] = triang_vec

    return matrix


def gramian_matrix(linear_operator, basis):
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


class MatrixOperator(Operator):
    """Linear operator for finite spaces.

    Between finite dimensional spaces, every linear operator can be expressed
    as a product by a matrix.

    Attributes:
        matrix (array-like object):  The matrix containing the linear
               transformation.

    """

    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, f):
        return self.matrix @ f

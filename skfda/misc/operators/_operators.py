import abc

import multimethod


class Operator(abc.ABC):
    """
    Abstract class for :term:`operators`.

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


def gramian_matrix_numerical(linear_operator, basis):
    r"""
    Return the gramian matrix given a basis, computed numerically.

    This method should work for every linear operator.

    """
    from .. import inner_product_matrix

    evaluated_basis = linear_operator(basis)

    return inner_product_matrix(evaluated_basis)


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

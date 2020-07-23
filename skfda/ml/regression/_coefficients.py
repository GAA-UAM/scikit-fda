from functools import singledispatch

import numpy as np

from ...misc._math import inner_product
from ...representation.basis import Basis, FDataBasis


class CoefficientInfo():
    """
    Information about an estimated coefficient.

    Parameters:
        basis: Basis of the coefficient.

    """

    def __init__(self, basis):
        self.basis = basis

    def regression_matrix(self, X, y):
        """
        Return the constant coefficients matrix for regression.

        Parameters:
            X: covariate data for regression.
            y: target data for regression.

        """
        return np.atleast_2d(X)

    def convert_from_constant_coefs(self, coefs):
        """
        Return the coefficients object from the constant coefs.

        Parameters:
            coefs: estimated constant coefficients.

        """
        return coefs

    def inner_product(self, coefs, X):
        """
        Compute the inner product between the coefficient and
        the covariate.

        """
        return inner_product(coefs, X)


class CoefficientInfoFDataBasis(CoefficientInfo):
    """
    Information about a FDataBasis coefficient.

    Parameters:
        basis: Basis of the coefficient.

    """

    def regression_matrix(self, X, y):
        # The matrix is the matrix of coefficients multiplied by
        # the matrix of inner products.

        xcoef = X.coefficients
        self.inner_basis = X.basis.inner_product_matrix(self.basis)
        return xcoef @ self.inner_basis

    def convert_from_constant_coefs(self, coefs):
        return FDataBasis(self.basis, coefs.T)

    def inner_product(self, coefs, X):
        # Efficient implementation of the inner product using the
        # inner product matrix previously computed
        return inner_product(coefs, X, inner_product_matrix=self.inner_basis.T)


@singledispatch
def coefficient_info_from_covariate(X, y, **kwargs) -> CoefficientInfo:
    """
    Make a coefficient info object from a covariate.

    """
    return CoefficientInfo(basis=np.identity(X.shape[1], dtype=X.dtype))


@coefficient_info_from_covariate.register(FDataBasis)
def coefficient_info_from_covariate_fdatabasis(
        X: FDataBasis, y, **kwargs) -> CoefficientInfoFDataBasis:
    basis = kwargs['basis']
    if basis is None:
        basis = X.basis

    if not isinstance(basis, Basis):
        raise TypeError(f"basis must be a Basis object, not {type(basis)}")

    return CoefficientInfoFDataBasis(basis=basis)

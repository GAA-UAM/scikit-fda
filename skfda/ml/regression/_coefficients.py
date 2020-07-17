from functools import singledispatch

import numpy as np

from ...representation.basis import Basis, FDataBasis


class CoefficientInfo():
    """
    Information about an estimated coefficient.

    At the very least it should have a type and a shape, but it may have
    additional information depending on its type.

    Parameters:
        coef_type: Class of the coefficient.
        shape: Shape of the constant coefficients form.

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


@singledispatch
def coefficient_info_from_covariate(X, y, **kwargs) -> CoefficientInfo:
    """
    Make a coefficient info object from a covariate.

    """
    return CoefficientInfo(basis=np.identity(X.shape[1], dtype=X.dtype))


class CoefficientInfoFDataBasis(CoefficientInfo):

    def regression_matrix(self, X, y):
        xcoef = X.coefficients
        inner_basis = X.basis.inner_product_matrix(self.basis)
        return xcoef @ inner_basis

    def convert_from_constant_coefs(self, coefs):
        return FDataBasis(self.basis, coefs.T)


@coefficient_info_from_covariate.register(FDataBasis)
def coefficient_info_from_covariate_fdatabasis(
        X: FDataBasis, y, **kwargs) -> CoefficientInfoFDataBasis:
    basis = kwargs['basis']
    if basis is None:
        basis = X.basis

    if not isinstance(basis, Basis):
        raise TypeError(f"basis must be a Basis object, not {type(basis)}")

    return CoefficientInfoFDataBasis(basis=basis)

from functools import singledispatch

import numpy as np

from ..representation.basis import Basis, FDataBasis


class CoefficientInfo():
    """
    Information about an estimated coefficient.

    At the very least it should have a type and a shape, but it may have
    additional information depending on its type.

    Parameters:
        coef_type: Class of the coefficient.
        shape: Shape of the coefficient.

    """

    def __init__(self, coef_type, shape):
        self.coef_type = coef_type
        self.shape = shape

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
    return CoefficientInfo(type(X), shape=X.shape)


class CoefficientInfoFDataBasis(CoefficientInfo):

    def __init__(self, shape, basis):
        super().__init__(coef_type=FDataBasis, shape=shape)

        self.basis = basis

    def regression_matrix(self, X, y):
        xcoef = X.coefficients
        inner_basis = X.basis.inner_product(self.basis)
        return xcoef @ inner_basis

    def convert_from_constant_coefs(self, coefs):
        return FDataBasis(self.basis, coefs.T)


@coefficient_info_from_covariate.register
def coefficient_info_from_covariate_fdatabasis(
        X: FDataBasis, y, **kwargs) -> CoefficientInfoFDataBasis:
    basis = kwargs['basis']
    if basis is None:
        basis = X.basis

    if not isinstance(basis, Basis):
        raise TypeError(f"basis must be a Basis object, not {type(basis)}")

    return CoefficientInfoFDataBasis(shape=(len(X),), basis=basis)

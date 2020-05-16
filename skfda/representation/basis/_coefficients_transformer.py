from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._fdatabasis import FDataBasis


class CoefficientsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer returning the coefficients of FDataBasis objects as a matrix.

    Attributes:
        shape_ (tuple): original shape of coefficients per sample.

    Examples:
        >>> from skfda.representation.basis import (FDataBasis, Monomial,
        ...                                         CoefficientsTransformer)
        >>>
        >>> basis = Monomial(n_basis=4)
        >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
        >>> fd = FDataBasis(basis, coefficients)
        >>>
        >>> transformer = CoefficientsTransformer()
        >>> transformer.fit_transform(fd)
        array([[ 0.5,  1. ,  2. ,  0.5],
               [ 1.5,  1. ,  4. ,  0.5]])

    """

    def fit(self, X: FDataBasis, y=None):

        self.shape_ = X.coefficients.shape[1:]

        return self

    def transform(self, X, y=None):

        check_is_fitted(self)

        assert X.coefficients.shape[1:] == self.shape_

        coefficients = X.coefficients.copy()
        coefficients = coefficients.reshape((X.n_samples, -1))

        return coefficients

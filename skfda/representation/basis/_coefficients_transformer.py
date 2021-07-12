from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._fdatabasis import FDataBasis


class CoefficientsTransformer(
    BaseEstimator,  # type:ignore
    TransformerMixin,  # type:ignore
):
    r"""
    Transformer returning the coefficients of FDataBasis objects as a matrix.

    Attributes:
        basis\_ (tuple): Basis used.

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

    def fit(  # noqa: D102
        self,
        X: FDataBasis,
        y: None = None,
    ) -> CoefficientsTransformer:

        self.basis_ = X.basis

        return self

    def transform(  # noqa: D102
        self,
        X: FDataBasis,
        y: None = None,
    ) -> np.ndarray:

        check_is_fitted(self)

        assert X.basis == self.basis_

        return X.coefficients.copy()

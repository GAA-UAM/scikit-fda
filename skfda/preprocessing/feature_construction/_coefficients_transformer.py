from __future__ import annotations

from typing import Optional

from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from ...representation import FDataBasis
from ...typing._numpy import NDArrayFloat


class CoefficientsTransformer(
    BaseEstimator,
    TransformerMixin[FDataBasis, NDArrayFloat, object],
):
    r"""
    Transformer returning the coefficients of FDataBasis objects as a matrix.

    Attributes:
        basis\_ (tuple): Basis used.

    Examples:
        >>> from skfda.preprocessing.feature_construction import (
        ...     CoefficientsTransformer,
        ... )
        >>> from skfda.representation.basis import (
        ...     FDataBasis,
        ...     MonomialBasis,
        ... )
        >>> basis = MonomialBasis(n_basis=4)
        >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
        >>> fd = FDataBasis(basis, coefficients)
        >>>
        >>> transformer = CoefficientsTransformer()
        >>> transformer.fit_transform(fd)
        array([[ 0.5,  1. ,  2. ,  0.5],
               [ 1.5,  1. ,  4. ,  0.5]])
                >>> transformer = CoefficientsTransformer(n_components=2)
        >>> transformer.fit_transform(fd)
        array([[ 0.5,  1. ],
               [ 1.5,  1. ]])

    """

    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components = n_components

    def fit(  # noqa: D102
        self,
        X: FDataBasis,
        y: object = None,
    ) -> CoefficientsTransformer:

        self.basis_ = X.basis

        return self

    def transform(  # noqa: D102
        self,
        X: FDataBasis,
        y: object = None,
    ) -> NDArrayFloat:

        check_is_fitted(self)

        assert X.basis == self.basis_

        return X.coefficients[:, :self.n_components].copy()

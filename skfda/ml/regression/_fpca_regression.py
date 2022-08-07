from typing import Callable, Optional, TypeVar, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ...misc.regularization import L2Regularization
from ._linear_regression import LinearRegression
from ...preprocessing.dim_reduction import FPCA
from ...representation import FData
from ...representation._typing import ArrayLike
from ...representation.basis import Basis

Function = TypeVar("Function", bound=FData)
WeightsCallable = Callable[[np.ndarray], np.ndarray]


class FPCARegression(BaseEstimator, RegressorMixin):
    r"""Regression using Functional Principal Components Analysis.

    Performs Functional Principal Components Analysis to reduce the dimension
    of the functional data, and then uses a linear regression model to
    relate the transformed data to a scalar value.

    Args:
        n_components: Number of principal components to keep.
        centering: If True then the data is centered before applying the
            principal components analysis. Defaults to True.
        regularization: Regularization parameter for the principal component
            analysis. If None then no regularization is applied. Defaults to
            None.
        weights: The weights used for discrete integration in the principal
            component analysis. If None then the weights are calculated using
            the trapezoidal rule. Defaults to None.
        components_basis: Basis used for the principal components. If None
            then the basis of the input data is used. Defaults to None.
            It is only used if the input data is a FDataBasis object.

    Attributes:
        n\_components\_: :class:`int`
            Number of components actually used.
    """

    def __init__(
        self,
        n_components: int = 2,
        centering: bool = True,
        regularization: Optional[L2Regularization[FData]] = None,
        weights: Optional[Union[ArrayLike, WeightsCallable]] = None,
        components_basis: Optional[Basis] = None,
    ) -> None:
        self.n_components_ = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = components_basis

    def fit(
        self,
        X: FData,
        y: np.ndarray,
    ) -> "FPCARegression":
        """Fit the model according to the given training data.

        Args:
            X: Functional data.
            y: Target values.

        Returns:
            self: object

        """
        self._fpca = FPCA(
            n_components=self.n_components_,
            centering=self.centering,
            regularization=self.regularization,
            weights=self.weights,
            components_basis=self.components_basis,
        )

        self._fpca.fit(X)

        # If the variables are not centered, an intercept term has to be
        # calculated
        self._linear_model = LinearRegression(fit_intercept=not self.centering)
        self._linear_model.fit(self._fpca.transform(X), y)

        return self

    def predict(
        self,
        X: FData,
    ) -> np.ndarray:
        """Predict using the linear model.

        Args:
            X: Functional data.

        Returns:
            Target values.

        """
        check_is_fitted(self, ["_fpca", "_linear_model"])

        return self._linear_model.predict(self._fpca.transform(X))

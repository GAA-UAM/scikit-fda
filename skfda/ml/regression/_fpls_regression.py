from __future__ import annotations

from typing import Any, TypeVar, Union

from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...misc.regularization import L2Regularization
from ...preprocessing.dim_reduction import FPLS
from ...representation import FDataGrid
from ...representation.basis import Basis, FDataBasis
from ...typing._numpy import NDArrayFloat

InputType = TypeVar(
    "InputType",
    bound=Union[FDataGrid, FDataBasis, NDArrayFloat],
)

OutputType = TypeVar(
    "OutputType",
    bound=Union[FDataGrid, FDataBasis, NDArrayFloat],
)


class FPLSRegression(
    RegressorMixin[InputType, OutputType],
    BaseEstimator,
):
    r"""
    Regression using Functional Partial Least Squares.

    Parameters:
        n_components: Number of components to keep.
            By default all available components are utilized.
        regularization_X: Regularization for the calculation of the X weights.
        weight_basis_X: Basis to use for the X block. Only
            applicable if X is a FDataBasis. Otherwise it must be None.
        weight_basis_Y: Basis to use for the Y block. Only
            applicable if Y is a FDataBasis. Otherwise it must be None.

    Attributes:
        coef\_: Coefficients of the linear model.
        fpls\_: FPLS object used to fit the model.

    Examples:
        Fit a FPLS regression model with two components.

        >>> from skfda.ml.regression import FPLSRegression
        >>> from skfda.datasets import fetch_tecator
        >>> from skfda.representation import FDataGrid
        >>> from skfda.typing._numpy import NDArrayFloat

        >>> X, y = fetch_tecator(return_X_y=True)
        >>> fpls = FPLSRegression[FDataGrid, NDArrayFloat](n_components=2)
        >>> fpls = fpls.fit(X, y)

    """

    def __init__(
        self,
        n_components: int | None = None,
        regularization_X: L2Regularization[Any] | None = None,
        weight_basis_X: Basis | None = None,
        weight_basis_Y: Basis | None = None,
        _integration_weights_X: NDArrayFloat | None = None,
        _integration_weights_Y: NDArrayFloat | None = None,
    ) -> None:
        self.n_components = n_components
        self._integration_weights_X = _integration_weights_X
        self._integration_weights_Y = _integration_weights_Y
        self.regularization_X = regularization_X
        self.weight_basis_X = weight_basis_X
        self.weight_basis_Y = weight_basis_Y

    def fit(
        self,
        X: InputType,
        y: OutputType,
    ) -> FPLSRegression[InputType, OutputType]:
        """
        Fit the model using the data for both blocks.

        Args:
            X: Data of the X block
            y: Data of the Y block

        Returns:
            self
        """
        self.fpls_ = FPLS[InputType, OutputType](
            n_components=self.n_components,
            regularization_X=self.regularization_X,
            component_basis_X=self.weight_basis_X,
            component_basis_Y=self.weight_basis_Y,
            _integration_weights_X=self._integration_weights_X,
            _integration_weights_Y=self._integration_weights_Y,
            _deflation_mode="reg",
        )

        self.fpls_.fit(X, y)

        self.coef_ = (
            self.fpls_.x_rotations_matrix_
            @ self.fpls_.y_loadings_matrix_.T
        )
        return self

    def predict(self, X: InputType) -> OutputType:
        """Predict using the model.

        Args:
            X: Data to predict.

        Returns:
            Predicted values.
        """
        check_is_fitted(self)
        return self.fpls_.inverse_transform_y(self.fpls_.transform_x(X))

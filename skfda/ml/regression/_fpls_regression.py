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
    BaseEstimator,
    RegressorMixin[InputType, OutputType],
):
    """
    Regression using Functional Partial Least Squares.

    Args:
        n_components: Number of components to keep. Defaults to 5.
    ...
    """

    def __init__(
        self,
        n_components: int = 5,
        regularization_X: L2Regularization[Any] | None = None,
        component_basis_X: Basis | None = None,
        component_basis_Y: Basis | None = None,
        _integration_weights_X: NDArrayFloat | None = None,
        _integration_weights_Y: NDArrayFloat | None = None,
    ) -> None:
        self.n_components = n_components
        self._integration_weights_X = _integration_weights_X
        self._integration_weights_Y = _integration_weights_Y
        self.regularization_X = regularization_X
        self.weight_basis_X = component_basis_X
        self.weight_basis_Y = component_basis_Y

    def fit(
        self,
        X: InputType,
        y: OutputType,
    ) -> FPLSRegression[InputType, OutputType]:
        """Fit the model using X as training data and y as target values."""
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

        return self

    def predict(self, X: InputType) -> OutputType:
        """Predict using the model.

        Args:
            X: FData to predict.

        Returns:
            Predicted values.

        """
        check_is_fitted(self)
        return self.fpls_.inverse_transform_y(self.fpls_.transform_x(X))

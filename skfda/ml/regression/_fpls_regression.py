from __future__ import annotations

from typing import TypeVar, Union

import multimethod
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, RegressorMixin
from ...misc.regularization import L2Regularization
from ...preprocessing.dim_reduction import FPLS
from ...representation import FData, FDataGrid
from ...representation.basis import Basis, FDataBasis
from ...typing._numpy import NDArrayFloat

FPLSRegressionSelf = TypeVar("FPLSRegressionSelf", bound="FPLSRegression")

Input = Union[FData, NDArrayFloat]

class FPLSRegression(
    BaseEstimator,
    RegressorMixin[Input, Input],
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
        scale: bool = False,
        integration_weights_X: NDArrayFloat | None = None,
        integration_weights_Y: NDArrayFloat | None = None,
        regularization_X: L2Regularization | None = None,
        weight_basis_X: Basis | None = None,
        weight_basis_Y: Basis | None = None,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.integration_weights_X = integration_weights_X
        self.integration_weights_Y = integration_weights_Y
        self.regularization_X = regularization_X
        self.weight_basis_X = weight_basis_X
        self.weight_basis_Y = weight_basis_Y

    def _predict_y(self, X: FData) -> FData:
        if isinstance(X, FDataGrid):
            return X.data_matrix[..., 0] @ self.fpls_.x_block.G_data_weights @ self.coef_
        elif isinstance(X, FDataBasis):
            return X.coefficients @ self.fpls_.x_block.G_data_weights @ self.coef_
        else:
            return X @ self.fpls_.x_block.G_data_weights @ self.coef_
    
    def _postprocess_response(self, y_data: NDArrayFloat) -> FData:
        if isinstance(self.train_y, FDataGrid):
            return self.train_y.copy(
                data_matrix=y_data,
                sample_names=(None,) * y_data.shape[0],
            )
        elif isinstance(self.train_y, FDataBasis):
            return self.train_y.copy(
                coefficients=y_data,
                sample_names=(None,) * y_data.shape[0],
            )
        else:
            return y_data

    def fit(
        self,
        X: FData,
        y: NDArrayFloat,
    ) -> FPLSRegressionSelf:
        """Fit the model using X as training data and y as target values."""
        # Center and scale data
        self.x_mean = X.mean(axis=0)
        self.y_mean = y.mean(axis=0)
        if self.scale:
            self.x_std = X.std()
            self.y_std = y.std()
        else:
            self.x_std = 1
            self.y_std = 1

        X = (X - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        self.fpls_ = FPLS(
            n_components=self.n_components,
            scale=False,
            integration_weights_X=self.integration_weights_X,
            integration_weights_Y=self.integration_weights_Y,
            regularization_X=self.regularization_X,
            weight_basis_X=self.weight_basis_X,
            weight_basis_Y=self.weight_basis_Y,
            deflation_mode="reg",
        )

        self.fpls_.fit(X, y)

        self.coef_ = self.fpls_.x_rotations_ @ self.fpls_.y_loadings_.T

        self.train_X = X
        self.train_y = y

        return self

    def predict(self, X: FData) -> NDArrayFloat:
        """Predict using the model.

        Args:
            X: FData to predict.

        Returns:
            Predicted values.

        """
        check_is_fitted(self)

        X = (X - self.x_mean) / self.x_std

        y_scaled = self._predict_y(X)
        y_scaled = self._postprocess_response(y_scaled)

        return y_scaled * self.y_std + self.y_mean

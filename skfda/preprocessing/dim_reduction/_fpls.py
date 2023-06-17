from __future__ import annotations

from typing import Optional, Tuple, Union

import multimethod
import numpy as np
import scipy
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData, FDataGrid
from ...representation.basis import Basis, FDataBasis, _GridBasis
from ...typing._numpy import NDArrayFloat

POWER_SOLVER_EPS = 1e-15


def _power_solver(X):
    t = X[:, 0]
    t_prev = np.ones(t.shape) * np.max(t) * 2
    iter_count = 0
    while np.linalg.norm(t - t_prev) > POWER_SOLVER_EPS:
        t_prev = t
        t = X @ t
        t /= np.linalg.norm(t)
        iter_count += 1
        if iter_count > 1000:
            break
    return t


def _calculate_weights(
    X,
    Y,
    G_ww,
    G_xw,
    G_cc,
    G_yc,
    L_X_inv,
    L_Y_inv,
):
    X = X @ G_xw @ L_X_inv.T
    Y = Y @ G_yc @ L_Y_inv.T
    S = X.T @ Y
    w = _power_solver(S @ S.T)

    # Calculate the other weight
    c = np.dot(Y.T, np.dot(X, w))

    # Undo the transformation
    w = L_X_inv.T @ w

    # Normalize
    w /= np.sqrt(np.dot(w.T, G_ww @ w))

    # Undo the transformation
    c = L_Y_inv.T @ c

    # Normalize the other weight
    c /= np.sqrt(np.dot(c.T, G_cc @ c))

    return w, c


def _pls_nipals(
    X,
    Y,
    n_components,
    G_ww,
    G_xw,
    G_cc,
    G_yc,
    L_X_inv,
    L_Y_inv,
    deflation="reg",
):
    X = X.copy()
    Y = Y.copy()
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]

    W, C = [], []
    T, U = [], []
    P, Q = [], []
    for _ in range(n_components):
        w, c = _calculate_weights(
            X,
            Y,
            G_ww=G_ww,
            G_xw=G_xw,
            G_cc=G_cc,
            G_yc=G_yc,
            L_X_inv=L_X_inv,
            L_Y_inv=L_Y_inv,
        )

        t = np.dot(X @ G_xw, w)
        u = np.dot(Y @ G_yc, c)

        p = np.dot(X.T, t) / np.dot(t.T, t)

        y_proyection = t if deflation == "reg" else u

        q = np.dot(Y.T, y_proyection) / np.dot(y_proyection, y_proyection)

        X = X - np.outer(t, p)
        Y = Y - np.outer(y_proyection, q)

        W.append(w)
        C.append(c)
        T.append(t)
        U.append(u)
        P.append(p)
        Q.append(q)

    W = np.array(W).T
    C = np.array(C).T
    T = np.array(T).T
    U = np.array(U).T
    P = np.array(P).T
    Q = np.array(Q).T

    # Ignore flake8 waring of too long output tuple
    return W, C, T, U, P, Q  # noqa: WPS227


InputType = Union[FData, NDArrayFloat]


class FPLS(
    BaseEstimator,
    InductiveTransformerMixin[InputType, InputType, Optional[InputType]],
):
    """
    Functional Partial Least Squares Regression.

    Attributes:
        n_components: Number of components to keep.
    ...
    """

    def __init__(
        self,
        n_components: int = 5,
        scale: bool = False,
        integration_weights_X: NDArrayFloat | None = None,
        integration_weights_Y: NDArrayFloat | None = None,
        regularization_X: L2Regularization | None = None,
        regularization_Y: L2Regularization | None = None,
        weight_basis_X: Basis | None = None,
        weight_basis_Y: Basis | None = None,
        deflation_mode: str = "can",
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.integration_weights_X = integration_weights_X
        self.integration_weights_Y = integration_weights_Y
        self.regularization_X = regularization_X
        self.regularization_Y = regularization_Y
        self.weight_basis_X = weight_basis_X
        self.weight_basis_Y = weight_basis_Y
        self.deflation_mode = deflation_mode

    @multimethod.multidispatch
    def _process_input_x(self, X):
        """
        Process the input data of the X block.

        This method is called by the fit method and
        it is implemented for each type of input data.
        """
        self.G_xw = np.identity(X.shape[1])
        self.G_ww = self.G_xw
        self._y_predictor = lambda X: X @ self.G_xw @ self.coef_
        self._make_component_x = lambda: self.x_rotations_
        self._regularization_matrix_X = 0

        self._transform_x = lambda X: X @ self.G_xw @ self.x_rotations_
        self._inv_transform_x = lambda T: T @ self.x_loadings_.T

        return X

    @_process_input_x.register
    def _process_input_x_grid(self, X: FDataGrid):
        x_mat = X.data_matrix[..., 0]
        if self.integration_weights_X is None:
            identity = np.eye(x_mat.shape[1])
            self.integration_weights_X = scipy.integrate.simps(identity, X.grid_points[0])

        self.G_xw = np.diag(self.integration_weights_X)
        self.G_ww = self.G_xw

        if self.regularization_X is not None:
            self._regularization_matrix_X = compute_penalty_matrix(
                basis_iterable=(_GridBasis(grid_points=X.grid_points),),
                regularization_parameter=1,
                regularization=self.regularization_X,
            )
        else:
            self._regularization_matrix_X = 0

        self._y_predictor = (
            lambda X: X.data_matrix[..., 0] @ self.G_xw @ self.coef_
        )

        self._make_component_x = lambda: X.copy(
            data_matrix=np.transpose(self.x_rotations_),
            sample_names=(None,) * self.n_components,
            dataset_name="FPLS X components",
        )

        self._transform_x = (
            lambda X: X.data_matrix[..., 0] @ self.G_xw @ self.x_rotations_
        )

        self._inv_transform_x = lambda T: X.copy(
            data_matrix=T @ self.x_loadings_.T,
            sample_names=(None,) * T.shape[0],
            dataset_name="FPLS X components",
        )

        return x_mat

    @_process_input_x.register
    def _process_input_x_basis(self, X: FDataBasis):
        x_mat = X.coefficients

        if self.weight_basis_Y is None:
            self.weight_basis_Y = X.basis

        self.G_xw = X.basis.inner_product_matrix(self.weight_basis_Y)
        self.G_ww = self.weight_basis_Y.gram_matrix()

        if self.regularization_X is not None:
            self._regularization_matrix_X = compute_penalty_matrix(
                basis_iterable=(self.weight_basis_Y,),
                regularization_parameter=1,
                regularization=self.regularization_X,
            )
        else:
            self._regularization_matrix_X = 0

        self._y_predictor = lambda X: X.coefficients @ self.G_xw @ self.coef_

        self._make_component_x = lambda: X.copy(
            coefficients=np.transpose(self.x_rotations_),
            sample_names=(None,) * self.n_components,
            dataset_name="FPLS X components",
        )

        self._transform_x = (
            lambda X: X.coefficients @ self.G_xw @ self.x_rotations_
        )

        self._inv_transform_x = lambda T: X.copy(
            coefficients=T @ self.x_loadings_.T,
            sample_names=(None,) * T.shape[0],
            dataset_name="FPLS X reconstructions",
        )

        return x_mat

    @multimethod.multidispatch
    def _process_input_y(self, Y):
        """
        Process the input data of the Y block.

        This method is called by the fit method and
        it is implemented for each type of input data.
        """
        self.G_yc = np.identity(Y.shape[1])
        self.G_cc = np.identity(Y.shape[1])
        self._regularization_matrix_Y = 0
        self._make_component_y = lambda: self.y_rotations_

        self._transform_y = lambda Y: Y @ self.G_yc @ self.y_rotations_
        self._inv_transform_y = lambda T: T @ self.y_loadings_.T
        return Y

    @_process_input_y.register
    def _process_input_y_grid(self, Y: FDataGrid):
        y_mat = Y.data_matrix[..., 0]
        if self.integration_weights_Y is None:
            identity = np.eye(y_mat.shape[1])
            self.integration_weights_Y = scipy.integrate.simps(identity, Y.grid_points[0])

        self.G_yc = np.diag(self.integration_weights_Y)
        self.G_cc = self.G_yc

        if self.regularization_Y is not None:
            self._regularization_matrix_Y = compute_penalty_matrix(
                basis_iterable=(_GridBasis(grid_points=Y.grid_points),),
                regularization_parameter=1,
                regularization=self.regularization_Y,
            )
        else:
            self._regularization_matrix_Y = 0

        self._make_component_y = lambda: Y.copy(
            data_matrix=np.transpose(self.y_rotations_),
            sample_names=(None,) * self.n_components,
            dataset_name="FPLS Y components",
        )

        self._transform_y = (
            lambda Y: Y.data_matrix[..., 0] @ self.G_yc @ self.y_rotations_
        )

        self._inv_transform_y = lambda T: Y.copy(
            data_matrix=T @ self.y_loadings_.T,
            sample_names=(None,) * T.shape[0],
            dataset_name="FPLS Y reconstructins",
        )

        return y_mat

    @_process_input_y.register
    def _process_input_y_basis(self, Y: FDataBasis):
        y_mat = Y.coefficients
        self.G_yc = Y.basis.gram_matrix()
        self.G_cc = self.G_yc

        if self.weight_basis_Y is None:
            self.weight_basis_Y = Y.basis

        if self.regularization_Y is not None:
            self._regularization_matrix_Y = compute_penalty_matrix(
                basis_iterable=(self.weight_basis_Y,),
                regularization_parameter=1,
                regularization=self.regularization_Y,
            )
        else:
            self._regularization_matrix_Y = 0

        self._make_component_y = lambda: Y.copy(
            coefficients=np.transpose(self.y_rotations_),
            sample_names=(None,) * self.n_components,
            dataset_name="FPLS Y components",
        )

        self._transform_y = (
            lambda Y: Y.coefficients @ self.G_yc @ self.y_rotations_
        )

        self._inv_transform_y = lambda T: Y.copy(
            coefficients=T @ self.y_loadings_.T,
            sample_names=(None,) * T.shape[0],
            dataset_name="FPLS Y reconstructions",
        )

        return y_mat

    def _fit_data(
        self,
        X: InputType,
        Y: InputType,
    ) -> None:
        """Fit the model using X and Y as already centered data."""
        x_mat = self._process_input_x(X)

        penalization_matrix = self.G_ww + self._regularization_matrix_X
        L_X_inv = np.linalg.inv(np.linalg.cholesky(penalization_matrix))

        y_mat = self._process_input_y(Y)

        penalization_matrix = self.G_cc + self._regularization_matrix_Y
        L_Y_inv = np.linalg.inv(np.linalg.cholesky(penalization_matrix))

        # Supress flake8 warning about too many values to unpack
        W, C, T, U, P, Q = _pls_nipals(  # noqa: WPS236
            x_mat,
            y_mat,
            self.n_components,
            G_ww=self.G_ww,
            G_xw=self.G_xw,
            G_cc=self.G_cc,
            G_yc=self.G_yc,
            L_X_inv=L_X_inv,
            L_Y_inv=L_Y_inv,
            deflation=self.deflation_mode,
        )

        self.x_weights_ = W
        self.y_weights_ = C
        self.x_scores_ = T
        self.y_scores_ = U
        self.x_loadings_ = P
        self.y_loadings_ = Q

        self.x_rotations_ = W @ np.linalg.pinv(P.T @ self.G_xw @ W)

        self.y_rotations_ = C @ np.linalg.pinv(Q.T @ self.G_yc @ C)

        self.components_x_ = self._make_component_x()
        self.components_y_ = self._make_component_y()

        self.coef_ = self.x_rotations_ @ Q.T

    def fit(
        self,
        X: InputType,
        y: InputType,
    ) -> FPLS:
        """
        Fit the model using the data for both blocks.

        Any of the parameters can be a FDataGrid, FDataBasis or numpy array.

        Args:
            X: Data of the X block
            y: Data of the Y block

        Returns:
            self
        """
        if isinstance(y, np.ndarray) and len(y.shape) == 1:
             y = y[:, np.newaxis]

        calculate_mean = (
            lambda x: x.mean() if isinstance(x, FData) else x.mean(axis=0)
        )
        # Center and scale data
        self.x_mean = calculate_mean(X)
        self.y_mean = calculate_mean(y)
        if self.scale:
            self.x_std = X.std()
            self.y_std = y.std()
        else:
            self.x_std = 1
            self.y_std = 1

        X = (X - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        self._fit_data(X, y)
        return self

    def transform(
        self,
        X: InputType,
        y: InputType | None = None,
    ) -> NDArrayFloat | Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Apply the dimension reduction learned on the train data.

        Args:
            X: Data to transform. Must have the same number of features and
                type as the data used to train the model.
            y : Data to transform. Must have the same number of features and
                type as the data used to train the model.
                If None, only X is transformed.

        Returns:
            x_scores: Data transformed.
            y_scores: Data transformed (if y is not None)
        """
        check_is_fitted(self)

        X = (X - self.x_mean) / self.x_std
        x_scores = self._transform_x(X)

        if y is not None:
            y = (y - self.y_mean) / self.y_std
            y_scores = self._transform_y(y)
            return x_scores, y_scores
        return x_scores

    def inverse_transform(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat | None = None,
    ) -> InputType | Tuple[InputType, InputType]:
        """
        Transform data back to its original space.

        Args:
            X: Data to transform back. Must have the same number of columns
                as the number of components of the model.
            Y: Data to transform back. Must have the same number of columns
                as the number of components of the model.

        Returns:
            X: Data reconstructed from the transformed data.
            Y: Data reconstructed from the transformed data
                (if Y is not None)
        """
        check_is_fitted(self)

        X = X * self.x_std + self.x_mean
        x_recon = self._inv_transform_x(X)

        if Y is not None:
            Y = Y * self.y_std + self.y_mean
            y_recon = self._inv_transform_y(Y)
            return x_recon, y_recon
        return x_recon

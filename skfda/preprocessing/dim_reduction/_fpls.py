from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Tuple, Union, overload
from typing_extensions import override

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


class FPLSBlock:
    def __init__(
        self,
        n_components: int,
        label: str,
        G_data_weights: NDArrayFloat,
        G_weights: NDArrayFloat,
        data_matrix: NDArrayFloat,
        regularization_matrix: NDArrayFloat | None = None,
    ) -> None:
        self.n_components = n_components
        self.label = label
        self.G_data_weights = G_data_weights
        self.G_weights = G_weights
        self.data_matrix = data_matrix

        if regularization_matrix is None:
            regularization_matrix = np.zeros(
                (data_matrix.shape[1], data_matrix.shape[1]),
            )
        self.regularization_matrix = regularization_matrix

    def set_values(
        self,
        rotations: NDArrayFloat,
        loadings: NDArrayFloat,
    ):
        self.rotations = rotations
        self.loadings = loadings

    @abstractmethod
    def make_component(self) -> NDArrayFloat | FData:
        pass

    @abstractmethod
    def transform(
        self,
        X: NDArrayFloat | FData,
    ) -> NDArrayFloat:
        pass

    @abstractmethod
    def inverse_transform(
        self,
        components: NDArrayFloat,
    ) -> NDArrayFloat | FData:
        pass

    @abstractmethod
    def y_predict(
        self,
        X: NDArrayFloat | FData,
        coef: NDArrayFloat | FData,
    ) -> NDArrayFloat | FData:
        pass


class FPLSBlockDataMultivariate(FPLSBlock):
    def __init__(
        self,
        data: NDArrayFloat,
        n_components: int,
        label: str,
    ) -> None:
        self.data = data

        super().__init__(
            n_components=n_components,
            label=label,
            G_data_weights=np.identity(data.shape[1]),
            G_weights=np.identity(data.shape[1]),
            data_matrix=data,
        )

    def make_component(self):
        return self.rotations

    def transform(self, data: NDArrayFloat | FData) -> NDArrayFloat:
        if not isinstance(data, NDArrayFloat):
            raise TypeError(
                f"Data in block {self.label} must be a numpy array",
            )
        return data @ self.rotations

    def inverse_transform(self, components: NDArrayFloat) -> NDArrayFloat:
        return components @ self.loadings.T

    def y_predict(
        self, X: NDArrayFloat | FData, coef: NDArrayFloat | FData,
    ) -> NDArrayFloat | FData:
        if not isinstance(X, NDArrayFloat):
            raise TypeError(
                f"X data in block {self.label} must be a numpy array",
            )
        return X @ coef


class FPLSBlockDataGrid(FPLSBlock):
    def __init__(
        self,
        data: FDataGrid,
        n_components: int,
        label: str,
        integration_weights: NDArrayFloat | None,
        regularization: L2Regularization | None,
    ) -> None:
        self.data = data
        data_mat = data.data_matrix[..., 0]

        if integration_weights is None:
            identity = np.identity(data_mat.shape[1])
            integration_weights = scipy.integrate.simps(
                identity,
                data.grid_points[0],
            )
        self.integration_weights = integration_weights

        self.G_data_weights = np.diag(self.integration_weights)
        self.G_weights = np.diag(self.integration_weights)

        regularization_matrix = None
        if regularization is not None:
            regularization_matrix = compute_penalty_matrix(
                basis_iterable=(_GridBasis(grid_points=data.grid_points),),
                regularization_parameter=1,
                regularization=regularization,
            )

        super().__init__(
            n_components=n_components,
            label=label,
            G_data_weights=np.diag(self.integration_weights),
            G_weights=np.diag(self.integration_weights),
            regularization_matrix=regularization_matrix,
            data_matrix=data_mat,
        )

    def make_component(self) -> FDataGrid:
        return self.data.copy(
            data_matrix=np.transpose(self.rotations),
            sample_names=(None,) * self.n_components,
            dataset_name=f"FPLS {self.label} components",
        )

    def transform(self, data: FDataGrid | NDArrayFloat) -> NDArrayFloat:
        if not isinstance(data, FDataGrid):
            raise TypeError(
                f"Data in block {self.label} must be an FDataGrid",
            )
        return data.data_matrix[..., 0] @ self.G_data_weights @ self.rotations

    def inverse_transform(self, components: NDArrayFloat) -> NDArrayFloat:
        return self.data.copy(
            data_matrix=components @ self.loadings.T,
            sample_names=(None,) * components.shape[0],
            dataset_name=f"FPLS {self.label} components",
        )

    def y_predict(
        self, X: NDArrayFloat | FData, coef: NDArrayFloat,
    ) -> NDArrayFloat:
        if not isinstance(X, FDataGrid):
            raise TypeError(
                f"X data in block {self.label} must be an FDataGrid",
            )
        return X.data_matrix[..., 0] @ self.G_data_weights @ coef


class FPLSBlockDataBasis(FPLSBlock):
    def __init__(
        self,
        data: FDataBasis,
        n_components: int,
        label: str,
        weight_basis: Basis | None,
        regularization: L2Regularization | None,
    ) -> None:
        self.data = data

        if weight_basis is None:
            self.weight_basis = data.basis
        else:
            self.weight_basis = weight_basis

        regularization_matrix = None
        if regularization is not None:
            regularization_matrix = compute_penalty_matrix(
                basis_iterable=(self.weight_basis,),
                regularization_parameter=1,
                regularization=regularization,
            )

        super().__init__(
            n_components=n_components,
            label=label,
            G_data_weights=self.weight_basis.gram_matrix(),
            G_weights=data.basis.inner_product_matrix(
                self.weight_basis,
            ),
            regularization_matrix=regularization_matrix,
            data_matrix=data.coefficients,
        )


    def make_component(self):
        return self.data.copy(
            coefficients=np.transpose(self.rotations),
            sample_names=(None,) * self.n_components,
            dataset_name=f"FPLS {self.label} components",
        )

    def transform(self, data: NDArrayFloat | FData) -> NDArrayFloat:
        if not isinstance(data, FDataBasis):
            raise TypeError(
                f"Data in block {self.label} must be an FDataBasis",
            )
        return data.coefficients @ self.G_weights @ self.rotations

    def inverse_transform(self, components: NDArrayFloat) -> NDArrayFloat:
        return self.data.copy(
            coefficients=components @ self.loadings.T,
            sample_names=(None,) * components.shape[0],
            dataset_name=f"FPLS {self.label} reconstructions",
        )

    def y_predict(
        self, X: NDArrayFloat | FData, coef: NDArrayFloat
    ) -> NDArrayFloat:
        if not isinstance(X, FDataBasis):
            raise TypeError(
                f"X data in block {self.label} must be an FDataBasis",
            )
        return X.coefficients @ self.G_data_weights @ coef


def block_factory(
    data,
    n_components: int,
    label: str,
    integration_weights: NDArrayFloat | None,
    regularization: L2Regularization | None,
    weight_basis: Basis | None,
) -> FPLSBlock:
    if isinstance(data, FDataGrid):
        return FPLSBlockDataGrid(
            data=data,
            n_components=n_components,
            label=label,
            integration_weights=integration_weights,
            regularization=regularization,
        )
    elif isinstance(data, FDataBasis):
        return FPLSBlockDataBasis(
            data=data,
            n_components=n_components,
            label=label,
            weight_basis=weight_basis,
            regularization=regularization,
        )
    return FPLSBlockDataMultivariate(
        data=data,
        n_components=n_components,
        label=label,
    )


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

    def _fit_data(
        self,
        X: InputType,
        Y: InputType,
    ) -> None:
        """Fit the model using X and Y as already centered data."""
        self._x_block = block_factory(
            data=X,
            n_components=self.n_components,
            label="X",
            integration_weights=self.integration_weights_X,
            regularization=self.regularization_X,
            weight_basis=self.weight_basis_X,
        )

        penalization_matrix = (
            self._x_block.G_weights + self._x_block.regularization_matrix
        )
        L_X_inv = np.linalg.inv(np.linalg.cholesky(penalization_matrix))

        self._y_block = block_factory(
            data=Y,
            n_components=self.n_components,
            label="X",
            integration_weights=self.integration_weights_Y,
            regularization=self.regularization_Y,
            weight_basis=self.weight_basis_Y,
        )

        penalization_matrix = (
            self._y_block.G_weights + self._y_block.regularization_matrix
        )
        L_Y_inv = np.linalg.inv(np.linalg.cholesky(penalization_matrix))

        # Supress flake8 warning about too many values to unpack
        W, C, T, U, P, Q = _pls_nipals(  # noqa: WPS236
            X=self._x_block.data_matrix,
            Y=self._y_block.data_matrix,
            n_components=self.n_components,
            G_ww=self._x_block.G_weights,
            G_xw=self._x_block.G_data_weights,
            G_cc=self._y_block.G_weights,
            G_yc=self._y_block.G_data_weights,
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
        self.x_rotations_ = W @ np.linalg.pinv(
            P.T @ self._x_block.G_data_weights @ W,
        )

        self.y_rotations_ = C @ np.linalg.pinv(
            Q.T @ self._y_block.G_data_weights @ C,
        )

        self._x_block.set_values(
            rotations=self.x_rotations_,
            loadings=self.x_loadings_,
        )
        self._y_block.set_values(
            rotations=self.y_rotations_,
            loadings=self.y_loadings_,
        )

        self.components_x_ = self._x_block.make_component()
        self.components_y_ = self._y_block.make_component()

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
        self._x_mean = calculate_mean(X)
        self._y_mean = calculate_mean(y)
        if self.scale:
            self._x_std = X.std()
            self._y_std = y.std()
        else:
            self._x_std = 1
            self._y_std = 1

        X = (X - self._x_mean) / self._x_std
        y = (y - self._y_mean) / self._y_std

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

        X = (X - self._x_mean) / self._x_std
        x_scores = self._x_block.transform(X)

        if y is not None:
            y = (y - self._y_mean) / self._y_std
            y_scores = self._y_block.transform(y)
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

        X = X * self._x_std + self._x_mean
        x_recon = self._x_block.inverse_transform(X)

        if Y is not None:
            Y = Y * self._y_std + self._y_mean
            y_recon = self._y_block.inverse_transform(Y)
            return x_recon, y_recon
        return x_recon

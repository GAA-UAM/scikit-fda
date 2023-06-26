from __future__ import annotations

from typing import Any, Generic, Optional, Tuple, TypeVar, Union, cast

import multimethod
import numpy as np
import scipy
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData, FDataGrid
from ...representation.basis import Basis, FDataBasis, _GridBasis
from ...typing._numpy import NDArrayFloat

POWER_SOLVER_EPS = 1e-15
INV_EPS = 1e-15


def _power_solver(X: NDArrayFloat) -> NDArrayFloat:
    """Return the dominant eigenvector of a matrix using the power method."""
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
    X: NDArrayFloat,
    Y: NDArrayFloat,
    G_ww: NDArrayFloat,
    G_xw: NDArrayFloat,
    G_cc: NDArrayFloat,
    G_yc: NDArrayFloat,
    L_X_inv: NDArrayFloat,
    L_Y_inv: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Calculate the weights for the PLS algorithm.

    Parameters as in _pls_nipals.
    Returns:
        - w: (n_features, 1)
            The X block weights.
        - c: (n_targets, 1)
            The Y block weights.
    """
    X = X @ G_xw @ L_X_inv.T
    Y = Y @ G_yc @ L_Y_inv.T
    S = X.T @ Y
    w = _power_solver(S @ S.T)

    # Calculate the other weight
    c = np.dot(Y.T, np.dot(X, w))

    # Undo the transformation
    w = L_X_inv.T @ w

    # Normalize
    w /= np.sqrt(np.dot(w.T, G_ww @ w)) + INV_EPS

    # Undo the transformation
    c = L_Y_inv.T @ c

    # Normalize the other weight
    c /= np.sqrt(np.dot(c.T, G_cc @ c)) + INV_EPS

    return w, c


# ignore flake8 multi-line function annotation
def _pls_nipals(  # noqa: WPS320
    X: NDArrayFloat,
    Y: NDArrayFloat,
    n_components: int,
    G_ww: NDArrayFloat,
    G_xw: NDArrayFloat,
    G_cc: NDArrayFloat,
    G_yc: NDArrayFloat,
    L_X_inv: NDArrayFloat,
    L_Y_inv: NDArrayFloat,
    deflation: str = "reg",
) -> Tuple[
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
]:
    """
    Perform the NIPALS algorithm for PLS.

    Parameters:
        - X: (n_samples, n_features)
            The X block data matrix.
        - Y: (n_samples, n_targets)
            The Y block data matrix.
        - n_components: number of components to extract.

        - G_ww: (n_features, n_features)
            The inner product matrix for the X block weights
            (The discretization weights matrix in the case of FDataGrid).
        - G_xw: (n_features, n_features)
            The inner product matrix for the X block data and weights
            (The discretization weights matrix in the case of FDataGrid).

        - G_cc: (n_targets, n_targets)
            The inner product matrix for the Y block weights
            (The discretization weights matrix in the case of FDataGrid).
        - G_yc: (n_targets, n_targets)
            The inner product matrix for the Y block data and weights
            (The discretization weights matrix in the case of FDataGrid).

        - L_X_inv: (n_features, n_features)
            The inverse of the Cholesky decomposition:
            L_X @ L_X.T = G_ww + P_x,
            where P_x is a the penalty matrix for the X block.
        - L_Y_inv: (n_targets, n_targets)
            The inverse of the Cholesky decomposition:
            L_Y @ L_Y.T = G_cc + P_y,
            where P_y is a the penalty matrix for the Y block.

        - deflation: The deflation method to use.
            Can be "reg" for regression or "can" for dimensionality reduction.
    Returns:
        - W: (n_features, n_components)
            The X block weights.
        - C: (n_targets, n_components)
            The Y block weights.
        - T: (n_samples, n_components)
            The X block scores.
        - U: (n_samples, n_components)
            The Y block scores.
        - P: (n_features, n_components)
            The X block loadings.
        - Q: (n_targets, n_components)
            The Y block loadings.
    """
    X = X.copy()
    Y = Y.copy()
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]

    # Store the matrices as list of columns
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

        p = np.dot(X.T, t) / (np.dot(t.T, t) + INV_EPS)

        y_proyection = t if deflation == "reg" else u

        q = np.dot(Y.T, y_proyection) / (
            np.dot(y_proyection, y_proyection) + INV_EPS
        )

        X = X - np.outer(t, p)
        Y = Y - np.outer(y_proyection, q)

        W.append(w)
        C.append(c)
        T.append(t)
        U.append(u)
        P.append(p)
        Q.append(q)

    # Convert each least of columns to a matrix
    return (  # noqa: WPS227 (too long output tuple)
        np.array(W).T,
        np.array(C).T,
        np.array(T).T,
        np.array(U).T,
        np.array(P).T,
        np.array(Q).T,
    )


BlockType = TypeVar(
    "BlockType",
    bound=Union[FDataGrid, FDataBasis, NDArrayFloat],
)


# Ignore too many public instance attributes
class _FPLSBlock(Generic[BlockType]):  # noqa: WPS230
    """
    Class to store the data of a block of a FPLS model.

    Attributes:
        - n_components: number of components to extract.
        - label: label of the block (X or Y).
        - G_data_weights: (n_samples, n_samples)
            The inner product matrix for the data and weights
            (The discretization weights matrix in the case of FDataGrid).
        - G_weights: (n_samples, n_samples)
            The inner product matrix for the weights
            (The discretization weights matrix in the case of FDataGrid).
        - data_matrix: (n_samples, n_features)
            The data matrix of the block.
    """

    mean: BlockType

    def __init__(
        self,
        data: BlockType,
        n_components: int,
        label: str,
        integration_weights: NDArrayFloat | None,
        regularization: L2Regularization[Any] | None,
        weights_basis: Basis | None,
    ) -> None:
        self.n_components = n_components
        self.label = label
        self._initialize_data(
            data=data,
            integration_weights=integration_weights,
            regularization=regularization,
            weights_basis=weights_basis,
        )

    @multimethod.multidispatch
    def _initialize_data(
        self,
        data: Union[FData, NDArrayFloat],
        integration_weights: Optional[NDArrayFloat],
        regularization: Optional[L2Regularization[Any]],
        weights_basis: Optional[Basis],
    ) -> None:
        """Initialize the data of the block."""
        raise NotImplementedError(
            f"Data type {type(data)} not supported",
        )

    @_initialize_data.register
    def _initialize_data_multivariate(
        self,
        data: np.ndarray,  # type: ignore[type-arg]
        integration_weights: None,
        regularization: None,
        weights_basis: None,
    ) -> None:
        """Initialize the data of the block."""
        self.G_data_weights = np.identity(data.shape[1])
        self.G_weights = np.identity(data.shape[1])

        self.mean = np.mean(data, axis=0)
        self.data_matrix = data - self.mean
        if len(self.data_matrix.shape) == 1:
            self.data_matrix = self.data_matrix[:, np.newaxis]
        self.data = data - self.mean

        self.regularization_matrix = np.zeros(
            (data.shape[1], data.shape[1]),
        )

    @_initialize_data.register
    def _initialize_data_fdatagrid(
        self,
        data: FDataGrid,
        integration_weights: Optional[np.ndarray],  # type: ignore[type-arg]
        regularization: Optional[L2Regularization[Any]],
        weights_basis: None,
    ) -> None:
        self.mean = data.mean()
        data = data - self.mean
        self.data = data

        data_mat = data.data_matrix[..., 0]
        if integration_weights is None:
            identity = np.identity(data_mat.shape[1])
            integration_weights = scipy.integrate.simps(
                identity,
                data.grid_points[0],
            )
        self.integration_weights = np.diag(integration_weights)

        regularization_matrix = None
        if regularization is not None:
            regularization_matrix = compute_penalty_matrix(
                basis_iterable=(_GridBasis(grid_points=data.grid_points),),
                regularization_parameter=1,
                regularization=regularization,
            )
        if regularization_matrix is None:
            regularization_matrix = np.zeros(
                (data_mat.shape[1], data_mat.shape[1]),
            )

        self.regularization_matrix = regularization_matrix
        self.G_data_weights = self.integration_weights
        self.G_weights = self.integration_weights
        self.data_matrix = data_mat

    @_initialize_data.register
    def _initialize_data_fdatabasis(
        self,
        data: FDataBasis,
        integration_weights: None,
        regularization: Optional[L2Regularization[Any]],
        weights_basis: Optional[Basis],
    ) -> None:
        self.mean = data.mean()
        data = data - self.mean
        self.data = data

        data_mat = data.coefficients
        if weights_basis is None:
            self.weights_basis = data.basis
        else:
            self.weights_basis = weights_basis

        regularization_matrix = None
        if regularization is not None:
            regularization_matrix = compute_penalty_matrix(
                basis_iterable=(self.weights_basis,),
                regularization_parameter=1,
                regularization=regularization,
            )
        if regularization_matrix is None:
            regularization_matrix = np.zeros(
                (data_mat.shape[1], data_mat.shape[1]),
            )

        self.regularization_matrix = regularization_matrix
        self.G_weights = self.weights_basis.gram_matrix()
        self.G_data_weights = data.basis.inner_product_matrix(
            self.weights_basis,
        )
        self.data_matrix = data_mat

    def calculate_components(
        self,
        rotations: NDArrayFloat,
        loadings: NDArrayFloat,
    ) -> BlockType:
        """Set the results of NIPALS."""
        self.rotations = rotations
        self.loadings = loadings
        self.components = self._calculate_component()
        return self.components

    def _calculate_component(self) -> BlockType:
        if isinstance(self.data, FDataGrid):
            return self.data.copy(
                data_matrix=np.transpose(self.rotations),
                sample_names=(None,) * self.n_components,
                dataset_name=f"FPLS {self.label} components",
            )
        elif isinstance(self.data, FDataBasis):
            return self.data.copy(
                coefficients=np.transpose(self.rotations),
                sample_names=(None,) * self.n_components,
                dataset_name=f"FPLS {self.label} components",
            )
        elif isinstance(self.data, np.ndarray):
            return cast(BlockType, self.rotations)

        raise NotImplementedError(
            f"Data type {type(self.data)} not supported",
        )

    def transform(
        self,
        data: BlockType,
    ) -> NDArrayFloat:
        """Transform from the data space to the component space."""
        if isinstance(data, FDataGrid):
            data_grid = data - cast(FDataGrid, self.mean)
            return (
                data_grid.data_matrix[..., 0]
                @ self.G_data_weights
                @ self.rotations
            )
        elif isinstance(data, FDataBasis):
            data_basis = data - cast(FDataBasis, self.mean)
            return (
                data_basis.coefficients @ self.G_data_weights @ self.rotations
            )
        elif isinstance(data, np.ndarray):
            data_array = data - cast(NDArrayFloat, self.mean)
            return data_array @ self.rotations

        raise NotImplementedError(
            f"Data type {type(data)} not supported",
        )

    def inverse_transform(
        self,
        components: NDArrayFloat,
    ) -> BlockType:
        """Transform from the component space to the data space."""
        if isinstance(self.data, FDataGrid):
            reconstructed_grid = self.data.copy(
                data_matrix=components @ self.loadings.T,
                sample_names=(None,) * components.shape[0],
                dataset_name=f"FPLS {self.label} components",
            )
            return reconstructed_grid + cast(FDataGrid, self.mean)

        elif isinstance(self.data, FDataBasis):
            reconstructed_basis = self.data.copy(
                coefficients=components @ self.loadings.T,
                sample_names=(None,) * components.shape[0],
                dataset_name=f"FPLS {self.label} components",
            )
            return reconstructed_basis + cast(FDataBasis, self.mean)

        elif isinstance(self.data, np.ndarray):
            reconstructed = components @ self.loadings.T
            reconstructed += self.mean
            return cast(BlockType, reconstructed)

        raise NotImplementedError(
            f"Data type {type(self.data)} not supported",
        )

    def get_penalty_matrix(self) -> NDArrayFloat:
        """Return the penalty matrix."""
        return self.G_weights + self.regularization_matrix

    def get_cholesky_inv_penalty_matrix(self) -> NDArrayFloat:
        """Return the Cholesky decomposition of the penalty matrix."""
        return np.linalg.inv(np.linalg.cholesky(self.get_penalty_matrix()))


InputTypeX = TypeVar(
    "InputTypeX",
    bound=Union[FDataGrid, FDataBasis, NDArrayFloat],
)
InputTypeY = TypeVar(
    "InputTypeY",
    bound=Union[FDataGrid, FDataBasis, NDArrayFloat],
)


# Ignore too many public instance attributes
class FPLS(  # noqa: WPS230
    BaseEstimator,
    Generic[InputTypeX, InputTypeY],
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
        regularization_X: L2Regularization[InputTypeX] | None = None,
        regularization_Y: L2Regularization[InputTypeY] | None = None,
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

    def _initialize_blocks(self, X: InputTypeX, Y: InputTypeY) -> None:
        self._x_block = _FPLSBlock(
            data=X,
            n_components=self.n_components,
            label="X",
            integration_weights=self.integration_weights_X,
            regularization=self.regularization_X,
            weights_basis=self.weight_basis_X,
        )
        self._y_block = _FPLSBlock(
            data=Y,
            n_components=self.n_components,
            label="X",
            integration_weights=self.integration_weights_Y,
            regularization=self.regularization_Y,
            weights_basis=self.weight_basis_Y,
        )

    def fit(
        self,
        X: InputTypeX,
        y: InputTypeY,
    ) -> FPLS[InputTypeX, InputTypeY]:
        """
        Fit the model using the data for both blocks.

        Any of the parameters can be a FDataGrid, FDataBasis or numpy array.

        Args:
            X: Data of the X block
            y: Data of the Y block

        Returns:
            self
        """
        self._initialize_blocks(X, y)

        # Supress flake8 warning about too many values to unpack
        W, C, T, U, P, Q = _pls_nipals(  # noqa: WPS236
            X=self._x_block.data_matrix,
            Y=self._y_block.data_matrix,
            n_components=self.n_components,
            G_ww=self._x_block.G_weights,
            G_xw=self._x_block.G_data_weights,
            G_cc=self._y_block.G_weights,
            G_yc=self._y_block.G_data_weights,
            L_X_inv=self._x_block.get_cholesky_inv_penalty_matrix(),
            L_Y_inv=self._y_block.get_cholesky_inv_penalty_matrix(),
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

        self.x_components_ = self._x_block.calculate_components(
            rotations=self.x_rotations_,
            loadings=P,
        )
        self.y_components_ = self._y_block.calculate_components(
            rotations=self.y_rotations_,
            loadings=Q,
        )

        self.coef_ = self.x_rotations_ @ Q.T
        return self

    def transform(
        self,
        X: InputTypeX,
        y: InputTypeY | None = None,
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

        x_scores = self._x_block.transform(X)

        if y is not None:
            y_scores = self._y_block.transform(y)
            return x_scores, y_scores
        return x_scores

    def transform_x(
        self,
        X: InputTypeX,
    ) -> NDArrayFloat:
        """
        Apply the dimension reduction learned on the train data.

        Args:
            X: Data to transform. Must have the same number of features and
                type as the data used to train the model.


        Returns:
            - Data transformed.
        """
        check_is_fitted(self)

        return self._x_block.transform(X)

    def transform_y(
        self,
        Y: InputTypeY,
    ) -> NDArrayFloat:
        """
        Apply the dimension reduction learned on the train data.

        Args:
            Y: Data to transform. Must have the same number of features and
                type as the data used to train the model.


        Returns:
            - Data transformed.
        """
        check_is_fitted(self)

        return self._y_block.transform(Y)

    def inverse_transform(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat | None = None,
    ) -> InputTypeX | Tuple[InputTypeX, InputTypeY]:
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

        x_reconstructed = self._x_block.inverse_transform(X)

        if Y is not None:
            y_reconstructed = self._y_block.inverse_transform(Y)
            return x_reconstructed, y_reconstructed
        return x_reconstructed

    def inverse_transform_x(
        self,
        X: NDArrayFloat,
    ) -> InputTypeX:
        """
        Transform X data back to its original space.

        Args:
            X: Data to transform back. Must have the same number of columns
                as the number of components of the model.

        Returns:
            - Data reconstructed from the transformed data.
        """
        check_is_fitted(self)

        return self._x_block.inverse_transform(X)

    def inverse_transform_y(
        self,
        Y: NDArrayFloat,
    ) -> InputTypeY:
        """
        Transform Y data back to its original space.

        Args:
            Y: Data to transform back. Must have the same number of columns
                as the number of components of the model.

        Returns:
            - Data reconstructed from the transformed data.
        """
        check_is_fitted(self)

        return self._y_block.inverse_transform(Y)

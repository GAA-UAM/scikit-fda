"""Functional Diffusion Maps Module."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy
from sklearn.utils.extmath import svd_flip
from typing_extensions import Self

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...representation import FData
from ...typing._numpy import NDArrayFloat

KernelFunction = Callable[[FData, FData | None], NDArrayFloat]


class DiffusionMap(
    BaseEstimator,
    InductiveTransformerMixin[FData, NDArrayFloat, object],
):
    r"""
    Functional diffusion maps.

    Class that implements functional diffusion maps
    :footcite:p:`barroso++_2023_fdm` for both basis and grid representations
    of the data.

    Note:
        Performing fit and transform actions is not equivalent to performing
        fit_transform. In the former case an approximation of the diffusion
        coordinates is computed via the Nyström method. In the latter, the
        true diffusion coordinates are computed.

    Parameters:
        n_components: Dimension of the space where the embedded
            functional data belongs to. For visualization of the
            data purposes, a value of 2 or 3 shall be used.
        kernel: kernel function used over the functional observations.
            It serves as a measure of connectivity or similitude between
            points, where higher value means greater connectivity.
        alpha: density parameter in the interval [0, 1] used in the
            normalization step. A value of 0 means the data distribution
            is not taken into account during the normalization step.
            The opposite holds for a higher value of alpha.
        n_steps: Number of steps in the random walk.

    Attributes:
        transition_matrix\_: trasition matrix computed from the data.
        eigenvalues\_: highest n_components eigenvalues of transition_matrix\_
            in descending order starting from the second highest.
        eigenvectors\_right\_: right eigenvectors of transition\_matrix\_
            corresponding to eigenvalues\_.
        d\_alpha\_: vector of densities of the weigthed graph.
        training\_dataset\_: dataset used for training the method. It is needed
        for the transform method.

    Examples:
        In this example we fetch the Canadian weather dataset and divide it
        into train and test sets. We then obtain the diffusion coordinates
        for the train set and predict these coordinates for the test set.

        >>> from skfda.datasets import fetch_weather
        >>> from skfda.representation import FDataGrid
        >>> from skfda.misc.covariances import Gaussian
        >>> X, y = fetch_weather(return_X_y=True, as_frame=True)
        >>> fd : FDataGrid = X.iloc[:, 0].values
        >>> fd_train = fd[:25]
        >>> fd_test = fd[25:]
        >>> fdm = DiffusionMap(
        ...     n_components=2,
        ...     kernel=Gaussian(variance=1, length_scale=1),
        ...     alpha=1,
        ...     n_steps=1
        ... )
        >>> embedding_train = fdm.fit_transform(X=fd_train)
        >>> embedding_test = fdm.transform(X=fd_test)

    References:

        .. footbibliography::

    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        kernel: KernelFunction,
        alpha: float = 0,
        n_steps: int = 1,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.alpha = alpha
        self.n_steps = n_steps

    def fit(  # noqa: WPS238
        self,
        X: FData,
        y: object = None,
    ) -> Self:
        """
        Compute the transition matrix and save it.

        Args:
            X: Functional data for which to obtain diffusion coordinates.
            y: Ignored.

        Returns:
            self

        """
        # Parameter validation
        if self.n_components < 1:
            raise ValueError(
                f'Embedding dimension ({self.n_components}) cannot '
                'be less than 1. ',
            )

        if self.n_components >= X.n_samples:
            raise ValueError(
                f'Embedding dimension ({self.n_components}) cannot be '
                f'greater or equal to the number of samples ({X.n_samples}).',
            )

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(
                f'Parameter alpha (= {self.alpha}) must '
                'be in the interval [0, 1].',
            )

        if self.n_steps < 0:
            raise ValueError(
                f'Parameter n_steps (= {self.n_steps}) must ',
                'be a greater than 0.',
            )

        # Construct the weighted graph
        weighted_graph = self.kernel(X, X)

        # Compute density of each vertex by adding the rows
        self.d_alpha_ = np.sum(weighted_graph, axis=1) ** self.alpha

        # Construct the normalized graph
        norm_w_graph = weighted_graph / np.outer(self.d_alpha_, self.d_alpha_)

        rows_sum = np.sum(norm_w_graph, axis=1)
        self.transition_matrix_ = norm_w_graph / rows_sum[:, np.newaxis]

        eigenvalues, eigenvectors_right_ = scipy.linalg.eig(
            self.transition_matrix_,
        )

        # Remove highest eigenvalue and take the n_components
        # highest eigenvalues in descending order
        index_order = eigenvalues.argsort()[-2:-self.n_components - 2:-1]
        eigenvalues = np.real(eigenvalues[index_order])
        eigenvectors_right, _ = svd_flip(
            eigenvectors_right_[:, index_order],
            np.zeros_like(eigenvectors_right_[:, index_order]).T,
        )
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_right_ = eigenvectors_right

        self.training_dataset_ = X

        return self

    def fit_transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the diffusion coordinates for the functional data X.

        The diffusion coordinate corresponding to the i-th curve is stored
        in the i-th row of the returning matrix.
        Note that fit_transform is not equivalent to applying fit and
        transform.

        Args:
            X: Functional data for which to obtain diffusion coordinates.
            y: Ignored.

        Returns:
            Diffusion coordinates for the functional data X.

        """
        self.fit(X, y)

        # Compute and return the diffusion map
        return np.real(
            self.eigenvectors_right_
            * self.eigenvalues_ ** self.n_steps,
        )

    def transform(
        self,
        X: FData,
    ) -> NDArrayFloat:
        """
        Compute the diffusion coordinates for the functional data X.

        Compute the diffusion coordinates of out-of-sample data using the
        eigenvectors and eigenvalues computed during the training.

        Args:
            X: Functional data for which to predict diffusion coordinates.

        Returns:
            Diffusion coordinates for the functional data X_out.

        """
        # Construct the weighted graph crossing the training
        # and out-of-sample data
        weighted_graph_out = self.kernel(X, self.training_dataset_)

        # Compute density of each vertex by adding the rows
        h_alpha = np.sum(weighted_graph_out, axis=1) ** self.alpha

        # Construct the normalized graph
        norm_w_graph_out = (
            weighted_graph_out / np.outer(h_alpha, self.d_alpha_)
        )

        # Define transition matrix by normalizing by rows
        rows_sum = np.sum(norm_w_graph_out, axis=1)
        transition_matrix_out = norm_w_graph_out / rows_sum[:, np.newaxis]

        return transition_matrix_out @ (  # type: ignore[no-any-return]
            self.eigenvectors_right_
            * self.eigenvalues_ ** (self.n_steps - 1)
        )

"""FPCA through Condictional Expectation Module."""

from __future__ import annotations
from typing import Callable, Optional, Union

import numpy as np
import scipy.integrate
from scipy.linalg import solve_triangular
from sklearn.decomposition import PCA

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData
from ...representation.basis import Basis, FDataBasis, _GridBasis
from ...representation.grid import FDataGrid
from ...typing._numpy import ArrayLike, NDArrayFloat

KernelFunction = Callable[[NDArrayFloat, float], float]


def gaussian_kernel(t: NDArrayFloat, h: float) -> float:
    """Default Gaussian kernel function."""
    d = len(t)
    norm_sq = np.sum((t / h) ** 2)
    coeff = 1 / ((2 * np.pi) ** (d / 2) * h**d)
    return float(coeff * np.exp(-0.5 * norm_sq))


class PACE(
    InductiveTransformerMixin[FData, NDArrayFloat, object],
    BaseEstimator,
):
    r"""
    FPCA through conditional expectation.

    Class that implements functional principal component analysis through
    conditional expectation. This method is native to FDataIrregular.

    For more information about the theoretical foundation for this algorithm,
    see: footcite:t:`yao+muller+wang_2005_pace`.

    Parameters:
        n_components: If parameter is an integer, it refers to the number of
            principal components to keep from functional principal component
            analysis. If parameter is a float in the range (0.0, 1. 0], it
            refers to the minimum proportion of variance explained by the
            selected principal components. Defaults to 1.0 (maximum number
            of principal components that can be extracted).
        centering: Set to ``False`` when the functional data is already known
            to be centered and there is no need to center it. Otherwise,
            the mean of the functional data object is calculated and the data
            centered before fitting . Defaults to ``True``.
        assume_noisy: Set to ``False`` when the data is assumed to be noiseless.
            Otherwise, when smoothing the covariance surface, the diagonal will
            be treated separately. Defaults to ``True``.
        kernel_mean: callable univariate smoothing kernel function for the mean,
            of the form :math:`K(t, h)`, where `t` are the n-dimensional time
            point, with n being the dimension of the domain, and `h` is the
            bandiwdth. Defaults to a Gaussian kernel.
        bandwidth_mean: bandwidth to use in the smoothing kernel for the mean.
            If no parameter is given, the bandwidth is calculated using the GCV
            method. Defaluts to ``None``.
        kernel_cov: callable univariate smoothing kernel function for the
            covariance and calculations regarding its diagonal. It should have
            the form :math:`K(t, h)`, where `t` are the n-dimensional time
            point, with n being the dimension of the domain, and `h` is the
            bandiwdth. To smooth the covariance, each value in the two
            directions will be calculated with the function and the two values
            will be multiplied, acting as an isotropic kernel. Defaults to a
            Gaussian kernel.
        bandwidth_cov: bandwidth to use in the smoothing kernel for the
            covariance. If no parameter is given, the bandwidth is calculated
            using the GCV method. Defaluts to ``None``.
        n_grid_points: number of grid points to calculate the covariance and,
            subsequently, the eigenfunctions for better approximations. The
            final FPC scores will be given in the original grid points.
        components_basis: The basis in which we want the principal
            components. We can use a different basis than the basis contained
            in the passed FDataBasis object. This parameter is only used when
            fitting an FDataBasis.
        variance_error_interval: A 2-element float vector in [0.0, 1.0]
            indicating the percent of data truncated during :math:`\\sigma^2`
            calculation. Defaults to (0.25, 0.75), as is suggested in
            footcite:t:`staniswalis+lee_1998_nonparametric_regression`

    Attributes:
        components\_: this contains the principal components.
        explained_variance\_ : The amount of variance explained by
            each of the selected components.
        explained_variance_ratio\_ : this contains the percentage
            of variance explained by each principal component.
        singular_values\_: The singular values corresponding to each of the
            selected components.
        mean\_: mean of the data.
        bandwidth_mean\_: calculated or user-given bandwidth used for the mean.
        covariance\_: covariance of the data.
        bandwidth_cov\_: calculated or user-given bandwidth used for the
            covariance.

    Examples:

    References:
        .. footbibliography::
    """

    def __init__(
        self,
        n_components: float = 1.0,
        *,
        centering: bool = True,
        assume_noisy: bool = True,
        kernel_mean: KernelFunction = gaussian_kernel,
        bandwidth_mean: float | None = None,
        kernel_cov: KernelFunction = gaussian_kernel,
        bandwidth_cov: float | None = None,
        n_grid_points: int = 51,
        components_basis: Basis | None = None,
        variance_error_interval: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        if (isinstance(n_components, int) and n_components <= 0) or not (
            isinstance(n_components, int) and not (0.0 < n_components <= 1.0)
        ):
            raise ValueError(
                "n_components must be an integer or a float in (0.0, 1.0].",
            )

        if (
            (bandwidth_mean is not None and bandwidth_mean <= 0)
            or (bandwidth_cov is not None and bandwidth_cov <= 0)
            or (n_grid_points <= 0)
        ):
            raise ValueError(
                "Bandwidth and grid point values must be positive.",
            )

        if (
            variance_error_interval[0] < 0
            or variance_error_interval[1] > 1
            or variance_error_interval[0] >= variance_error_interval[1]
        ):
            raise ValueError(
                "variance_error_interval must be an increasing tuple of two "
                "floats in [0.0, 1.0].",
            )
        
        self.n_components = n_components
        self.centering = centering
        self.assume_noisy = assume_noisy
        self.kernel_mean = kernel_mean
        self.bandwidth_mean = bandwidth_mean
        self.kernel_cov = kernel_cov
        self.bandwidth_cov = bandwidth_cov
        self.n_grid_points = n_grid_points
        self.components_basis = components_basis
        self.variance_error_interval = variance_error_interval

    def fit(
        self,
        X: FData,
        y: object = None,
    ) -> PACE:
        """
        Compute the ``n_components`` first principal components and saves them.

        Args:
            X: The functional data object to be analysed.
            y: Ignored. Only present because of fit function convention.

        Returns:
            self
        """
        pass

    def transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored. Only present because of fit function convention.

        Returns:
            Principal component scores. Data matrix of shape
            ``(n_samples, n_components)``.
        """
        pass

    def fit_transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the n_components first principal components and their scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored

        Returns:
            Principal component scores.

        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(
        self,
        pc_scores: NDArrayFloat,
    ) -> FData:
        """
        Compute the recovery from the fitted principal components scores.

        In other words, it maps ``pc_scores``, from the fitted functional
        PCs' space, back to the input functional space. ``pc_scores`` might be
        an array returned by ``transform`` method.

        Args:
            pc_scores: ndarray (n_samples, n_components).

        Returns:
            A FData object.
        """
        pass

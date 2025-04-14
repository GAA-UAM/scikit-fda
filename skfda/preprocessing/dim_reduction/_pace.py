"""FPCA through Condictional Expectation Module."""

from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData
from ...representation.basis import Basis, FDataBasis, _GridBasis
from ...representation.grid import FDataGrid
from ...representation.irregular import FDataIrregular
from ...typing._base import DomainRangeLike
from ...typing._numpy import ArrayLike, NDArrayAny, NDArrayFloat

KernelFunction = Callable[[NDArrayFloat, float], NDArrayFloat]


def gaussian_kernel(t: NDArrayFloat, h: float) -> NDArrayFloat:
    """
    Vectorized Gaussian kernel function.

    Args:
        t: Array of shape (n_samples, n_dims), where each row is a difference vector.
        h: Bandwidth (scalar)

    Returns:
        Kernel weights of shape (n_samples,)
    """
    if t.ndim == 1:
        t = t[None, :]  # Ensure 2D shape

    d = t.shape[1]
    norm_sq = np.sum((t / h) ** 2, axis=1)  # ||t/h||^2 for each row
    coeff = 1 / ((2 * np.pi) ** (d / 2) * h**d)
    return np.array(coeff * np.exp(-0.5 * norm_sq), dtype=np.float64)


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
        assume_noisy: Set to ``False`` when the data is assumed to be
            noiseless. Otherwise, when smoothing the covariance surface, the
            diagonal will be treated separately. Defaults to ``True``.
        kernel_mean: callable univariate smoothing kernel function for the
            mean, of the form :math:`K(t, h)`, where `t` are the n-dimensional
            time point, with n being the dimension of the domain, and `h` is
            the bandiwdth. Defaults to a Gaussian kernel.
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
            Defaults to 51.
        boundary_effect_interval: A 2-element float vector indicating the
            percentage of the time points to be considered as left and right
            boundary regions of the time window of observations. Defaults to
            (0.0, 1.0), that is, the whole time window.
        variance_error_interval: A 2-element float vector in [0.0, 1.0]
            indicating the percent of data truncated during :math:`\\sigma^2`
            calculation. Defaults to (0.25, 0.75), as is suggested in
            footcite:t:`staniswalis+lee_1998_nonparametric_regression`

    Attributes:
        components\_: this contains the principal components.
        explained_variance\_ : the amount of variance explained by
            each of the selected components.
        explained_variance_ratio\_ : this contains the percentage
            of variance explained by each principal component.
        singular_values\_: the singular values corresponding to each of the
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
        assume_noisy: bool = True,
        kernel_mean: KernelFunction = gaussian_kernel,
        bandwidth_mean: float | None = None,
        kernel_cov: KernelFunction = gaussian_kernel,
        bandwidth_cov: float | None = None,
        n_grid_points: int = 51,
        boundary_effect_interval: Sequence[float] = (0.0, 1.0),
        variance_error_interval: Sequence[float] = (0.25, 0.75),
    ) -> None:
        if (isinstance(n_components, int) and n_components <= 0) or not (
            isinstance(n_components, int)
            and (n_components <= 0.0 or n_components > 1.0)
        ):
            error_msg = (
                "n_components must be an integer or a float in (0.0, 1.0]."
            )
            raise ValueError(error_msg)

        if (
            (bandwidth_mean is not None and bandwidth_mean <= 0)
            or (bandwidth_cov is not None and bandwidth_cov <= 0)
            or (n_grid_points <= 0)
        ):
            error_msg = "Bandwidth and grid point values must be positive."
            raise ValueError(error_msg)

        interval_length = 2
        if (
            len(boundary_effect_interval) != interval_length
            or boundary_effect_interval[0] < 0
            or boundary_effect_interval[1] > 1
            or boundary_effect_interval[0] >= boundary_effect_interval[1]
            or len(variance_error_interval) != interval_length
            or variance_error_interval[0] < 0
            or variance_error_interval[1] > 1
            or variance_error_interval[0] >= variance_error_interval[1]
        ):
            error_msg = (
                "interval parameters must be an increasing sequence of two "
                "floats in [0.0, 1.0]."
            )
            raise ValueError(error_msg)

        self.n_components = n_components
        self.assume_noisy = assume_noisy
        self.kernel_mean = kernel_mean
        self.bandwidth_mean = bandwidth_mean
        self.kernel_cov = kernel_cov
        self.bandwidth_cov = bandwidth_cov
        self.n_grid_points = n_grid_points
        self.boundary_effect_interval = boundary_effect_interval
        self.variance_error_interval = variance_error_interval

    def _slice_fdata_irregular(
        self,
        data: FDataIrregular,
    ) -> FDataIrregular:
        """
        Slice the FDataIrregular object to the interval [a, b].

        Args:
            data: The FDataIrregular object to be sliced.

        Returns:
            A new FDataIrregular object sliced to the interval [a, b].
        """
        # Reduce time points and values based on the boundary effect
        domain_range = np.array(data.domain_range)
        domain_diff = domain_range[:, 1] - domain_range[:, 0]

        # Apply boundary effect intervals: one global pair (e.g. (0.1, 0.9))
        # applied to all dims
        start_cut = (
            domain_range[:, 0] + domain_diff * self.boundary_effect_interval[0]
        )
        end_cut = domain_range[:, 1] - domain_diff * (
            1 - self.boundary_effect_interval[1]
        )

        cut_domain_range = list(zip(start_cut, end_cut, strict=True))

        a_bounds = np.array([interval[0] for interval in cut_domain_range])
        b_bounds = np.array([interval[1] for interval in cut_domain_range])

        all_points = data.points
        all_values = data.values
        start_indices = data.start_indices
        end_indices = np.append(start_indices[1:], len(all_points))

        new_points = []
        new_values = []
        new_start_indices = [0]

        for start, end in zip(start_indices, end_indices, strict=True):
            pts = all_points[start:end, :]  # (m_i, n_dims)
            vals = all_values[start:end, :]  # (m_i, output_dims)

            # Build boolean mask: keep rows where all coords are inside their
            # bounds
            mask = np.all((pts >= a_bounds) & (pts <= b_bounds), axis=1)

            filtered_pts = pts[mask]
            filtered_vals = vals[mask]

            new_points.append(filtered_pts)
            new_values.append(filtered_vals)
            new_start_indices.append(new_start_indices[-1] + len(filtered_pts))

        filtered_points = np.concatenate(new_points, axis=0)
        filtered_values = np.concatenate(new_values, axis=0)
        filtered_start_indices = np.array(
            new_start_indices[:-1], dtype=np.uint32
        )

        return FDataIrregular(
            points=filtered_points,
            values=filtered_values,
            start_indices=filtered_start_indices,
            domain_range=cut_domain_range,
            argument_names=data.argument_names,
            coordinate_names=data.coordinate_names,
            sample_names=data.sample_names,
            dataset_name=data.dataset_name,
        )

    def _mean_lls(
        self,
        h: float,
        t_eval: NDArrayFloat,
        t_obs: NDArrayFloat,
        y_obs: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Local linear smoothing to estimate the mean function at given points.

        Args:
            h: Bandwidth for the kernel function.
            t_eval: Points to evaluate, shape (n_eval, d).
            t_obs: Observed time points, shape (n_obs, d).
            y_obs: Observed values, shape (n_obs, n_outputs).

        Returns:
            Smoothed values at t_eval points, shape (n_eval, n_outputs).
        """
        epsilon = 1e-8

        # Ensure all are 2D
        t_eval = np.atleast_2d(t_eval)
        t_obs = np.atleast_2d(t_obs)
        y_obs = np.atleast_2d(y_obs)

        n_eval, d = t_eval.shape
        n_obs, _ = t_obs.shape

        # Compute pairwise differences (n_eval, n_obs, d)
        diffs = t_eval[:, None, :] - t_obs[None, :, :]

        # Evaluate kernel per (n_eval, n_obs)
        flat_diffs = diffs.reshape(-1, d)
        flat_weights = self.kernel_mean(flat_diffs, h)

        # Reshape back to (n_eval, n_obs)
        weights = flat_weights.reshape(n_eval, n_obs)

        # Reshape weights for broadcasting over outputs
        weights_exp = weights[:, :, None]
        scalar_diffs = diffs[..., 0]
        scalar_diffs_exp = scalar_diffs[:, :, None]

        y_obs_exp = y_obs[None, :, :]

        # Compute smoother components
        k0 = np.sum(weights, axis=1)[:, None]
        k1 = np.sum(weights_exp * scalar_diffs_exp, axis=1)
        k2 = np.sum(weights_exp * scalar_diffs_exp**2, axis=1)

        s0 = np.sum(weights_exp * y_obs_exp, axis=1)
        s1 = np.sum(
            weights_exp * scalar_diffs_exp * y_obs_exp, axis=1
        )

        det = k0 * k2 - k1**2 + epsilon

        return np.where(
            det != 0,
            (s0 * k2 - s1 * k1) / det,
            np.mean(y_obs, axis=0, keepdims=True),
        )

    def _mean_gcv_score(
        self,
        h: float,
        t_obs: NDArrayFloat,
        y_obs: NDArrayFloat,
    ) -> float:
        """
        Compute the Generalized Cross-Validation (GCV) score for a given bandwidth.

        Args:
            h: Bandwidth to evaluate
            t_obs: Observed time points
            y_obs: Observed function values
            t_eval: Query points where smoother is evaluated.

        Returns:
            GCV score
        """
        if h <= 0:  # Bandwidth must be positive
            return np.inf

        # Compute smoothed estimates for each observed point
        y_hat = self._mean_lls(h, t_obs, t_obs, y_obs)

        # Compute residual sum of squares (RSS)
        rss = np.sum((y_obs - y_hat) ** 2)

        # Approximate trace of smoother matrix
        domain_diff = r = np.max(pdist(t_obs))

        k0 = self.kernel_mean(np.zeros((1, t_obs.shape[1])), 1.0)[0]

        n_obs = t_obs.shape[0]

        denom = (1 - (domain_diff * k0) / (n_obs * h)) ** 2
        return float(rss / denom) if denom > 0 else np.inf

    def fit(
        self,
        X: FDataIrregular,
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
        # Check that the number of components is smaller than the sample size
        if self.n_components > len(X.start_indices):
            error_msg = (
                "The sample size must be bigger than the number of components",
            )
            raise AttributeError(error_msg)

        if self.boundary_effect_interval != (0.0, 1.0):
            # Slice the data to remove the boundary effect
            x_work = self._slice_fdata_irregular(X)
        else:
            x_work = X

        time_points = np.sort(np.unique(x_work.points, axis=0), axis=0)

        optimal_bandwidth = minimize_scalar(
            self._mean_gcv_score,
            args=(x_work.points, x_work.values),
            bounds=(0.1, 10.0),
            method="bounded",
        ).x

        print(f"Optimal bandwidth for mean: {optimal_bandwidth}")

        self.mean_ = self._mean_lls(
            optimal_bandwidth,
            time_points,
            x_work.points,
            x_work.values,
        )

        # print(f"Mean: {self.mean_}")

        # plt.scatter(
        #     x_work.points[:, 0],
        #     x_work.values[:, 0],
        #     color="red",
        #     label="Observed Data",
        # )
        # plt.plot(
        #     time_points[:, 0],
        #     self.mean_[:, 0],
        #     label="Smoothed Mean",
        #     color="blue",
        # )
        # plt.xlabel("Time")
        # plt.ylabel("Mean Estimate")
        # plt.legend()
        # plt.title("Local Linear Smoother for Irregular Data (Vectorized)")
        # plt.show()

        # Generate 1D grids per dimension
        # axes = [
        #     np.linspace(start, end, self.n_grid_points)
        #     for start, end in x_work.domain_range
        # ]

        # # Create full meshgrid and flatten to shape (n_points, n_dimensions)
        # mesh = np.meshgrid(*axes, indexing="ij")
        # work_grid = np.stack([m.flatten() for m in mesh], axis=-1)

        return self

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

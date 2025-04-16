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
from scipy.spatial import cKDTree
from scipy.sparse import diags
from scipy.linalg import pinv
import warnings



KernelFunction = Callable[[NDArrayFloat], NDArrayFloat]


def gaussian_kernel(t: NDArrayFloat) -> NDArrayFloat:
    """
    Vectorized Gaussian kernel function.

    Args:
        t: Array of shape (n_samples, n_dims), where each row is a difference vector.

    Returns:
        Kernel weights of shape (n_samples,)
    """
    d = t.shape[1]

    norm_sq = np.sum((t) ** 2, axis=1)  # ||t/h||^2 for each row
    coeff = 1 / ((2 * np.pi) ** (d / 2))
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
        kernel_mean: callable vectorized univariate smoothing kernel function
            for the mean, of the form :math:`K(t, h)`, where `t` are the
            n-dimensional time point, with n being the dimension of the domain,
            and `h` is the bandiwdth. Defaults to a Gaussian kernel.
        bandwidth_mean: bandwidth to use in the smoothing kernel for the mean.
            If no parameter is given, the bandwidth is calculated using the GCV
            method. If a float is given, it is used as the bandwidth. If a
            tuple is given, it is used as the bandwidth search range. This last
            option is useful in some kernel functions, where the search result
            is a boundary of the default search range (0.1, 10.0). Defaluts to
            ``None``.
        kernel_cov: callable vectorized univariate smoothing kernel function
            for the covariance and calculations regarding its diagonal. It
            should have the form :math:`K(t, h)`, where `t` are the
            n-dimensional time point, with n being the dimension of the domain,
            and `h` is the bandiwdth. To smooth the covariance, each value in
            the two directions will be calculated with the function and the two
            values will be multiplied, acting as an isotropic kernel. Defaults
            to a Gaussian kernel.
        bandwidth_cov: bandwidth to use in the smoothing kernel for the
            covariance. If no parameter is given, the bandwidth is calculated
            using the GCV method. If a float is given, it is used as the
            bandwidth. If a tuple is given, it is used as the bandwidth search
            range. This last option is useful in some kernel functions, where
            the search result is a boundary of the default search range (0.1,
            10.0). Defaluts to ``None``.
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
        bandwidth_mean_interval\_: Interval of the bandwidth search for the
            mean.
        covariance\_: covariance of the data.
        bandwidth_cov\_: calculated or user-given bandwidth used for the
            covariance.
        bandwidth_cov_interval\_: Interval of the bandwidth search for the
            covariance.

    Examples:

    References:
        .. footbibliography::
    """

    def _check_bandwidth(
        self,
        bandwidth: float | NDArrayFloat | None,
    ) -> tuple[float | None, tuple[float, float] | None]:
        """
        Check if the bandwidth has the correct form.

        Args:
            bandwidth: Bandwidth to check.

        Returns:
            A 2-element tuple with the value if the bandwidth is a float, or
            None otherwise, and the search range if the bandwidth is a tuple,
            or None otherwise. In the case that the bandwidth is None, the
            function returns None, (0.1, 10.0) as the default search range.
        """
        if bandwidth is None:
            return None, (0.1, 10.0)

        if isinstance(bandwidth, float) and bandwidth <= 0:
            error_msg = "Given bandwidth values must be positive."
            raise ValueError(error_msg)

        tuple_length = 2

        if isinstance(bandwidth, Sequence) and (
            len(bandwidth) != tuple_length
            or not all(isinstance(b, (float, int)) for b in bandwidth)
            or bandwidth[0] <= 0
            or bandwidth[1] <= bandwidth[0]
        ):
            error_msg = (
                "Bandwidth search ranges must be a non-decreasing 2-sequence "
                "of floats."
            )
            raise ValueError(error_msg)

        if isinstance(bandwidth, float):
            return bandwidth, None

        return None, (bandwidth[0], bandwidth[1])

    def __init__(
        self,
        n_components: float = 1.0,
        *,
        assume_noisy: bool = True,
        kernel_mean: KernelFunction = gaussian_kernel,
        bandwidth_mean: float | NDArrayFloat | None = None,
        kernel_cov: KernelFunction = gaussian_kernel,
        bandwidth_cov: float | NDArrayFloat | None = None,
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

        self.bandwidth_mean_, self.bandwidth_mean_interval_ = (
            self._check_bandwidth(bandwidth_mean)
        )

        self.bandwidth_cov_, self.bandwidth_cov_interval_ = (
            self._check_bandwidth(bandwidth_cov)
        )

        if n_grid_points <= 0:
            error_msg = "n_grid_points must be positive or None."
            raise ValueError(error_msg)

        tuple_length = 2
        if (
            len(boundary_effect_interval) != tuple_length
            or boundary_effect_interval[0] < 0
            or boundary_effect_interval[1] > 1
            or boundary_effect_interval[0] >= boundary_effect_interval[1]
            or len(variance_error_interval) != tuple_length
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
        self.kernel_cov = kernel_cov
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
            GCV score for the given bandwidth.
        """
        if h <= 0:  # Bandwidth must be positive
            return np.inf

        # Compute smoothed estimates for each observed point
        y_hat = self._mean_lls(h, t_obs, t_obs, y_obs)

        # Compute residual sum of squares (RSS)
        rss = np.sum((y_obs - y_hat) ** 2)

        # Approximate trace of smoother matrix
        domain_diff = np.max(pdist(t_obs))
        k0 = self.kernel_mean(np.zeros((1, t_obs.shape[1])))[0]
        n_obs = t_obs.shape[0]

        denom = (1 - (domain_diff * k0) / (n_obs * h)) ** 2
        return float(rss / denom) if denom > 0 else np.inf

    def _mean_lls(
        self,
        h: float,
        t_eval: NDArrayFloat,
        t_obs: NDArrayFloat,
        y_obs: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Local linear smoother for mean estimation.

        Args:
            h: Bandwidth for the kernel.
            t_eval: Query points where smoother is evaluated.
            t_obs: Observed time points.
            y_obs: Observed function values.

        Returns:
            Smoothed estimates for each query point.
        """
        epsilon = 1e-8

        t_eval = np.atleast_2d(t_eval)
        t_obs = np.atleast_2d(t_obs)
        y_obs = np.atleast_2d(y_obs)

        n_eval, d = t_eval.shape
        n_obs, q = y_obs.shape

        # (n_eval, n_obs, d): differences for each eval-obs pair
        diffs = t_eval[:, None, :] - t_obs[None, :, :]

        # Compute kernel weights
        flat_diffs = diffs.reshape(-1, d)
        flat_weights = self.kernel_mean(flat_diffs/h)
        weights = flat_weights.reshape(n_eval, n_obs)

        estimates = np.empty((n_eval, q))

        for i in range(n_eval):
            wi = weights[i]
            xi = diffs[i]
            yi = y_obs

            k0 = np.sum(wi)
            k1 = np.sum(wi[:, None] * xi, axis=0)
            k2 = np.einsum("ni,nj->ij", wi[:, None] * xi, xi)

            s0 = np.sum(wi[:, None] * yi, axis=0)
            s1 = np.einsum("ni,nj->ij", wi[:, None] * xi, yi)

            # Solve linear system for beta0 (intercept)
            # Build left-hand matrix and right-hand side
            xtwx = np.block(
                [[np.array([[k0]]), k1[None, :]], [k1[:, None], k2]]
            ) + epsilon * np.eye(d + 1)
            xtwy = np.vstack([s0[None, :], s1])

            beta = np.linalg.solve(xtwx, xtwy)
            estimates[i] = beta[0]  # intercept term

        return estimates

    def _compute_raw_covariances(
        self,
        x_work: FDataIrregular,
        mean: NDArrayFloat,
        time_points: NDArrayFloat,
    ) -> tuple[
        NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat,
    ]:
        """
        Compute raw covariances for irregular data.

        Compute raw covariances for irregular data, filtering duplicates based
        on the ``assume_noisy`` parameter.

        Args:
            x_work: FDataIrregular object containing the data.
            mean: Mean function values.
            time_points: Time points for the mean.

        Returns:
            Array of time point pairs.
            Array of raw covariance values.
            Array of indices for the data points.
            Array of weights for the covariance.
            Array of raw covariance values without removing duplicates.
        """
        points = x_work.points
        values = x_work.values
        start_indices = x_work.start_indices
        end_indices = np.append(start_indices[1:], len(points))

        # Create lists to store the results
        t_1, t_2, x_1, x_2, subj_idx, raw_cov = [], [], [], [], [], []

        # Vectorize the time points and corresponding values
        for i in range(len(start_indices)):
            p_i = points[start_indices[i] : end_indices[i]]
            v_i = values[start_indices[i] : end_indices[i]]

            # Find the mean projection for each point
            tree = cKDTree(time_points)
            _, indices = tree.query(p_i)
            mean_proj = mean[indices]

            # Center the values by subtracting the mean
            r_i = v_i - mean_proj

            for j in range(len(p_i)):
                for k in range(len(p_i)):
                    # Store all pairs of time points (including duplicates)
                    t_1.append([p_i[j]])
                    t_2.append([p_i[k]])
                    x_1.append(r_i[j])
                    x_2.append(r_i[k])
                    subj_idx.append(i)  # Subject index
                    raw_cov.append(r_i[j] * r_i[k])  # Raw covariance

        # Convert lists to numpy arrays
        t_pairs = np.array([t_1, t_2]).squeeze().T
        t_pairs = t_pairs.reshape(t_pairs.shape[0], 2, -1)
        f_raw_cov = np.array(raw_cov)

        if self.assume_noisy:
            t_neq = np.where(t_pairs[:, 0] != t_pairs[:, 1])[0]
            t_pairs = t_pairs[t_neq]
            f_raw_cov = f_raw_cov[t_neq]

        win = np.ones(len(f_raw_cov))

        return t_pairs, f_raw_cov, np.array(subj_idx), win, np.array(raw_cov)

    def _cov_gcv_score(
        self,
        h: float,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
    ) -> float:
        """
        Compute GCV score for bandwidth h for covariance smoothing.

        Returns:
            Scalar GCV score.
        """
        print(f"Evaluating GCV for h={h:.4f}")
        if h <= 0:
            return np.inf

        # Evaluate smoothed covariance at same locations
        r_eval = cov_coords[:, 0, :]
        s_eval = cov_coords[:, 1, :]
        print(f"r_eval: {r_eval}")
        print(f"s_eval: {s_eval}")
        print(f"cov_values: {cov_values}")
        print(r_eval.shape, s_eval.shape, cov_values.shape)
        G_hat = self._cov_lls(h, r_eval, s_eval, cov_coords, cov_values)

        rss = np.sum((cov_values - G_hat) ** 2)

        domain_diff = np.max(pdist(t_pairs))
        k0 = self.kernel_cov(np.zeros((1, t_pairs.shape[1])), 1.0)[0]
        k0 *= k0
        n_obs = len(cov_values)

        denom = (1 - (domain_diff * k0) / (n_obs * h)) ** 2
        return float(rss / denom) if denom > 0 else np.inf

    def _cov_lls(
        self,
        h: float,
        r_eval: NDArrayFloat,
        s_eval: NDArrayFloat,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        win: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Local linear smoother for covariance estimation.

        Args:
            h: Bandwidth for the kernel.
            r_eval: First array of query points where smoother is evaluated.
            s_eval: Second array of query points where smoother is evaluated.
            cov_coords: Coordinates of the covariance.
            cov_values: Values of the covariance.
            win: Weights for the covariance.

        Returns:
            n_grid_points x n_grid_points array of smoothed covariance values.
        """
        # Active indices based on non-zero weights
        active = np.nonzero(win)[0]
        t_pairs = cov_coords[active, :]
        cov_values = cov_values[active]
        win = win[active]

        # Initialize the mu matrix (output)
        mu = np.zeros((len(s_eval), len(r_eval)))

        # Loop through the grid points (r_eval, s_eval)
        for i in range(len(s_eval)):
            for j in range(i, len(r_eval)):  # Only upper triangle
                # Locate the local window based on the kernel type
                if self.kernel_cov is not gaussian_kernel:
                    # Get the indices where the conditions are true
                    ind_r = np.where(np.abs(t_pairs[:, 0] - r_eval[j]) <= h + 1e-6)[0]
                    ind_s = np.where(np.abs(t_pairs[:, 1] - s_eval[i]) <= h + 1e-6)[0]

                    # Combine the indices by checking both conditions
                    ind = np.intersect1d(ind_r, ind_s)

                else:  # Gaussian kernel, use all points
                    ind = np.arange(len(t_pairs))

                # Local points
                l_t = t_pairs[ind, :]
                l_x = cov_values[ind]
                l_w = win[ind]

                min_unique = 6
                min_unique_pairs = 3
                # Compute the weight matrix based on the kernel, checking that
                # there are sufficient distinct points and time point pairs.
                if len(np.unique(l_t[:])) >= min_unique or len(np.unique(l_t, axis=0)) >= min_unique_pairs:
                    l_diff_t = (l_t[:, 0] - r_eval[j]) / h
                    l_diff_s = (l_t[:, 1] - s_eval[i]) / h

                    kernel_r = self.kernel_cov(l_diff_t)
                    kernel_s = self.kernel_cov(l_diff_s)

                    w = kernel_r * kernel_s
                    w *= l_w

                    x = np.ones((len(l_x), 3))
                    x[:, 1] = (l_t[:, 0] - r_eval[j]).flatten()
                    x[:, 2] = (l_t[:, 1] - s_eval[i]).flatten()

                    # Weighted least squares: Solve for beta
                    w_diag = np.diag(w)  # Create a diagonal weight matrix
                    xtwx = x.T @ w_diag @ x  # X^T W X
                    xtxy = x.T @ w_diag @ l_x  # X^T W y
                    beta = np.linalg.pinv(xtwx) @ xtxy  # Solving for beta

                    mu[i, j] = beta[0]  # Intercept term

                    print(f"mu[{i}, {j}] = {mu[i, j]}")

                else:
                    warnings.warn(
                        "Not enough points in local window for "
                        f"({r_eval[j]}, {s_eval[i]}), increase bandwidth."
                        "This may lead to inaccurate results.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return np.array([])

        return mu

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

        # === MATLAB-style candidate search ===
        # domain_diff = np.max(x_work.points, axis=0) - np.min(x_work.points, axis=0)
        # r = float(np.max(domain_diff))

        # dists = np.sort(np.diff(np.unique(x_work.points[:, 0])))
        # if len(dists) >= 3:
        #     dstar = np.min([np.sum(dists[:i+1]) for i in range(2, len(dists))])
        # else:
        #     dstar = r / 10  # fallback

        # h0 = 2.5 * dstar

        # # Create q and the 10 candidate bandwidths
        # q = (r / (4 * h0)) ** (1 / 9)
        # candidates = np.array([h0 * q**i for i in range(10)])

        # # Evaluate GCV at each candidate
        # gcv_scores = np.array([
        #     self._mean_gcv_score(h, x_work.points, x_work.values)
        #     for h in candidates
        # ])
        # optimal_bandwidth = candidates[np.argmin(gcv_scores)]

        if self.bandwidth_mean_ is None:
            self.bandwidth_mean_ = minimize_scalar(
                self._mean_gcv_score,
                args=(x_work.points, x_work.values),
                bounds=self.bandwidth_mean_interval_,
                method="bounded",
            ).x

            # The following correction is a practical empirical correction. This is
            # inspired by the fact that, although the Gaussian kernel gives good
            # results for irregular data, the fact that it has infinite support
            # (nonzero weights for all points) can lead to over-smoothing. The
            # following term has the objective of slightly correcting this effect.
            # This can also be seen in the PACE package in Matlab.
            if self.kernel_mean == gaussian_kernel:
                self.bandwidth_mean_ *= 1.1

        print(f"Selected bandwidth for mean: {self.bandwidth_mean_}. ")

        self.mean_ = self._mean_lls(
            self.bandwidth_mean_,
            time_points,
            x_work.points,
            x_work.values,
        )

        # print(f"Mean: {self.mean_}")

        raw_cov_coords, raw_cov_values, cov_subj_idx, win, unf_cov_values = (
            self._compute_raw_covariances(
                x_work,
                self.mean_,
                time_points,
            )
        )

        if self.bandwidth_cov_ is None:
            # self.bandwidth_cov_ = minimize_scalar(
            #     self._cov_gcv_score,
            #     args=(raw_cov_coords, raw_cov_values),
            #     bounds=self.bandwidth_cov_interval_,
            #     method="bounded",
            # ).x
            bandwidth_candidates = np.linspace(0.1, 10, 3)

            # Now you can evaluate GCV for each candidate
            gcv_scores = np.array(
                [
                    self._cov_gcv_score(h, raw_cov_coords, raw_cov_values)
                    for h in bandwidth_candidates
                ]
            )
            print(f"Bandwidth candidates: {bandwidth_candidates}")

            # Select the bandwidth with the minimum GCV score
            self.bandwidth_cov_ = bandwidth_candidates[np.argmin(gcv_scores)]
            print(f"Selected bandwidth for covariance: {self.bandwidth_cov_}.")

        print(f"Selected bandwidth for covariance: {self.bandwidth_cov_}. ")

        # Create n-dimensional work grid to calculate covariance surface in
        # Create grid of domain points for covariance evaluation
        axes = [
            np.linspace(start, end, self.n_grid_points)
            for start, end in x_work.domain_range
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        t_eval = np.stack([m.ravel() for m in mesh], axis=-1)  # (n_eval, d)

        # print(f"Shape of t_eval: {t_eval.shape}")

        self.covariance_ = self._cov_lls(
            self.bandwidth_cov_,
            t_eval,
            t_eval,
            raw_cov_coords,
            raw_cov_values,
            win,
        )

        print(f"Covariance: {self.covariance_.shape}")

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
        return [1.0]

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
        return [1.0]

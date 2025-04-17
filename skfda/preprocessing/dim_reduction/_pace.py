"""FPCA through Condictional Expectation Module."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...representation import FData
from ...representation.irregular import FDataIrregular
from ...typing._numpy import NDArrayFloat

KernelFunction = Callable[[NDArrayFloat], NDArrayFloat]


def gaussian_kernel(t: NDArrayFloat) -> NDArrayFloat:
    """
    Vectorized Gaussian kernel function.

    Args:
        t: Array of shape (n_samples, n_dims), where each row is a different
        vector.

    Returns:
        Kernel weights of shape (n_samples,)
    """
    n_obs, n_eval, n_dims = t.shape

    norm_sq = np.sum(t**2, axis=2)  # Shape: (n_obs, n_eval)

    # Apply the Gaussian kernel formula
    coeff = 1 / ((2 * np.pi) ** (n_dims / 2))
    return np.array(coeff * np.exp(-0.5 * norm_sq))


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
        bw_cov_n_grid_points: number of grid points to calculate the bandwidth
            for the covariance. This parameter's main purpose is to reduce the
            computational cost of the GCV method. Defaults to 30.
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
        bw_cov_n_grid_points: int = 30,
        n_grid_points: int = 51,
        boundary_effect_interval: Sequence[float] = (0.0, 1.0),
        variance_error_interval: Sequence[float] = (0.25, 0.75),
    ) -> None:
        if (isinstance(n_components, int) and n_components <= 0) or (
            not isinstance(n_components, int)
            and (n_components <= 0.0 or n_components > 1.0)
        ):
            error_msg = (
                "n_components must be an integer or a float in (0.0, 1.0]."
            )
            raise ValueError(error_msg)

        bandwidth_mean_, bandwidth_mean_interval_ = (
            self._check_bandwidth(bandwidth_mean)
        )

        bandwidth_cov_, bandwidth_cov_interval_ = (
            self._check_bandwidth(bandwidth_cov)
        )

        if n_grid_points <= 0 or bw_cov_n_grid_points <= 0:
            error_msg = "Grid points must be positive or None."
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
        self.bandwidth_mean_ = bandwidth_mean_
        self.bandwidth_mean_interval_ = bandwidth_mean_interval_
        self.kernel_cov = kernel_cov
        self.bandwidth_cov_ = bandwidth_cov_
        self.bandwidth_cov_interval_ = bandwidth_cov_interval_
        self.bw_cov_n_grid_points = bw_cov_n_grid_points
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
            new_start_indices[:-1], dtype=np.uint32,
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
        Compute the Generalized Cross-Validation (GCV) score.

        Compute the Generalized Cross-Validation (GCV) score for a given
        bandwidth.

        Args:
            h: Bandwidth to evaluate
            t_obs: Observed time points
            y_obs: Observed function values

        Returns:
            GCV score for the given bandwidth.
        """
        if h <= 0:  # Bandwidth must be positive
            return np.inf

        # Compute smoothed estimates for each observed point
        y_hat = self._mean_lls(h, t_obs, t_obs, y_obs, self.kernel_mean)

        # Compute residual sum of squares (RSS)
        rss = np.sum((y_obs - y_hat) ** 2)

        # Approximate trace of smoother matrix
        domain_diff = np.max(pdist(t_obs))
        k0 = self.kernel_mean(np.zeros((1, 1, t_obs.shape[1])))[0]
        n_obs = t_obs.shape[0]

        denom = (1 - (domain_diff * k0) / (n_obs * h)) ** 2
        return float(rss / denom) if denom > 0 else np.inf

    def _mean_lls(
        self,
        h: float,
        t_eval: NDArrayFloat,
        t_obs: NDArrayFloat,
        y_obs: NDArrayFloat,
        kernel: KernelFunction,
    ) -> NDArrayFloat:
        """
        Local linear smoother for mean estimation.

        Args:
            h: Bandwidth for the kernel.
            t_eval: Query points where smoother is evaluated.
            t_obs: Observed time points.
            y_obs: Observed function values.
            kernel: Kernel function to use for smoothing.

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
        weights = kernel(diffs / h)

        estimates = np.empty((n_eval, q))

        for i in range(n_eval):
            wi = weights[i]
            xi = diffs[i]
            yi = y_obs
            win = wi[:, None]

            k0 = np.sum(wi)
            k1 = np.sum(win * xi, axis=0)
            k2 = np.einsum("ni,nj->ij", win * xi, xi)

            s0 = np.sum(win * yi, axis=0)
            s1 = np.einsum("ni,nj->ij", win * xi, yi)

            # Solve linear system for beta0 (intercept)
            # Build left-hand matrix and right-hand side
            xtwx = np.block(
                [[np.array([[k0]]), k1[None, :]], [k1[:, None], k2]],
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
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
        NDArrayFloat,
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
            Array of time point pairs for equal time points.
            Array of diagonal of raw covariance values.
        """
        points = x_work.points
        values = x_work.values
        start_indices = x_work.start_indices
        end_indices = np.append(start_indices[1:], len(points))

        # Create lists to store the results
        t_1, t_2, x_1, x_2 = [], [], [], []
        subj_idx, raw_cov = [], []

        # Vectorize the time points and corresponding values
        for i in range(len(start_indices)):
            p_i = points[start_indices[i]:end_indices[i]]
            v_i = values[start_indices[i]:end_indices[i]]

            # Find the mean projection for each point
            tree = cKDTree(time_points)
            _, indices = tree.query(p_i)
            mean_proj = mean[indices]

            # Center the values by subtracting the mean
            r_i = v_i - mean_proj

            for j, p_ij in enumerate(p_i):
                for k, p_ik in enumerate(p_i):
                    # Store all pairs of time points (including duplicates)
                    t_1.append([p_ij])
                    t_2.append([p_ik])
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
            t_eq = np.where(t_pairs[:, 0] == t_pairs[:, 1])[0]
            t_pairs_neq = t_pairs[t_neq]
            f_raw_cov_neq = f_raw_cov[t_neq]
            t_pairs_eq = t_pairs[t_eq][:, 0]
            f_raw_cov_eq = f_raw_cov[t_eq]

        win = np.ones(len(f_raw_cov_neq))

        return t_pairs_neq, f_raw_cov_neq, np.array(subj_idx), win, t_pairs_eq, f_raw_cov_eq

    def _cov_gcv_score(
        self,
        h: float,
        t_eval: NDArrayFloat,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        win: NDArrayFloat,
        time_points: NDArrayFloat,
    ) -> float:
        """
        Compute GCV score for bandwidth h for covariance smoothing.

        Args:
            h: Bandwidth to evaluate
            t_eval: Query points where smoother is evaluated.
            cov_coords: Coordinates of the covariance.
            cov_values: Values of the covariance.
            win: Weights for the covariance.
            time_points: Time points for the mean (used to obtain range).

        Returns:
            Scalar GCV score.
        """
        print(f"Evaluating GCV for h={h:.4f}")
        if h <= 0:
            return np.inf

        # Evaluate smoothed covariance at same locations
        g_hat = self._cov_lls(
            h,
            t_eval,
            t_eval,
            cov_coords,
            cov_values,
            win,
        ).squeeze()

        # Interpolation grid points
        x, y = np.meshgrid(t_eval, t_eval)
        grid_points = np.c_[x.ravel(), y.ravel()]

        # Interpolate
        # g_hat_int = griddata(
        #     grid_points,
        #     g_hat.ravel(),
        #     cov_coords.squeeze(),
        #     method="cubic",
        # )

        # Interpolate at the grid points
        interpolator = CloughTocher2DInterpolator(grid_points, g_hat.ravel())
        g_hat_int = interpolator(cov_coords.squeeze())

        # Calculate residual sum of squares (RSS)
        rss = np.sum(
            (cov_values.squeeze() - g_hat_int)
            * (cov_values.squeeze() - g_hat_int).T,
        )

        # Calculate pairwise distances between points in cov_coords
        domain_diff = np.max(pdist(time_points))
        k0 = self.kernel_cov(np.zeros((1, 1, cov_coords.shape[2])))[0]
        n_obs = len(cov_values)
        # Normalize by number of observations and bandwidth
        denom = 1 - (1 / n_obs) * ((domain_diff * k0) / h) ** 2

        return float(rss / denom**2) if denom > 0 else np.inf

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

        n_eval, d = r_eval.shape
        n_obs, q = cov_values.shape

        # Correct the broadcasting shape of r_eval and s_eval
        diff_r = (t_pairs[:, 0, None] - r_eval[None, :]) / h
        diff_s = (t_pairs[:, 1, None] - s_eval[None, :]) / h

        kernel_r = self.kernel_cov(diff_r).T
        kernel_s = self.kernel_cov(diff_s).T

        weights = np.einsum("ik,jk->ijk", kernel_r, kernel_s)
        w_diag = weights * win

        # Build the design matrix for weighted least squares
        x = np.ones((n_eval, n_eval, n_obs, 3))
        for i in range(n_eval):
            for j in range(n_eval):
                x[:, j, :, 1, None] = t_pairs[None, :, 0] - r_eval[:, None]
                x[i, :, :, 2, None] = t_pairs[None, :, 1] - s_eval[:, None]

        x_t = np.transpose(x, (0, 1, 3, 2))
        xtw = x_t * w_diag[:, :, None, :]
        xtwx = xtw @ x
        xtwy = xtw @ cov_values

        beta = np.linalg.pinv(xtwx) @ xtwy

        cov = beta[:, :, 0]
        cov_t = np.transpose(cov, (1, 0, 2))

        return np.array((cov + cov_t) / 2.0)

    def _get_pc(
        self,
        cov_matrix: NDArrayFloat,
        n_components: float,
    ) -> tuple[int, NDArrayFloat, NDArrayFloat]:
        """
        Select the number of principal components.

        Select the best number of principal components based on fraction of
        variance explained or number of components.

        Args:
            cov_matrix: The smoothed covariance matrix.
            n_components: The threshold for the fraction of variance explained,
                or the number of components to keep.

        Returns:
            The chosen number of principal components.
            Cumulative fraction of variance explained.
            Eigenvalues.
        """
        cov = cov_matrix.squeeze()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Remove negative or complex eigenvalues and sort in decreasing order
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = np.sort(eigenvalues)[::-1]

        fve = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        if isinstance(n_components, int):
            if n_components > len(eigenvalues):
                error_msg = (
                    "The number of components must be smaller than the sample size",
                )
                raise AttributeError(error_msg)
            n_selected_components = n_components
        else:
            # Find the optimal number of components
            n_selected_components = np.where(fve >= n_components)[0][0]+1

        return n_selected_components, fve, eigenvalues

    def _get_sigma2(
        self,
        h: float,
        t_eval: NDArrayFloat,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        t_diag: NDArrayFloat,
        cov_diag: NDArrayFloat,
        win: NDArrayFloat,
    ) -> None:
        """
        Estimate the variance of the covariance matrix.

        Args:
            cov_matrix: The smoothed covariance matrix.

        Returns:
            The estimated variance.
            The estimated covariance.
        """
        smooth_diag = self._mean_lls(
            h,
            t_eval,
            t_diag,
            cov_diag,
            self.kernel_cov,
        )

        rotated_cov = self._rotated_cov_lls(
            h,
            t_eval,
            t_eval,
            cov_coords,
            cov_values,
            win,
        )

        # print(f"Rotated covariance: {rotated_cov}")

    def _rotated_cov_lls(
        self,
        h: float,
        r_eval: NDArrayFloat,
        s_eval: NDArrayFloat,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        win: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Local linear smoother for rotated covariance estimation.

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
        r_mat = np.sqrt(2)/2 * np.array([[1, 1],[-1, 1]])

        # Rotate coordinates of covariance points and evaluation points
        r_cov_coords = np.einsum("ijk,jk->ik", cov_coords, r_mat)
        r_cov_coords = r_cov_coords[:, :, np.newaxis]

        t_eval = np.stack((r_eval, s_eval), axis=1).squeeze()
        r_t_eval = t_eval @ r_mat
        r_t_eval = r_t_eval[:, :, np.newaxis]

        active = np.nonzero(win)[0]
        t_pairs = r_cov_coords[active, :]
        cov_values = cov_values[active]
        win = win[active]

        n_eval, d = r_eval.shape
        n_obs, q = cov_values.shape

        # Correct the broadcasting shape of r_eval and s_eval
        diff_r = (t_pairs[:, 0, None] - r_t_eval[None,:,0]) / h
        diff_s = (t_pairs[:, 1, None] - r_t_eval[None,:,1]) / h

        kernel_r = self.kernel_cov(diff_r).T
        kernel_s = self.kernel_cov(diff_s).T

        weights = np.einsum("ik,jk->ijk", kernel_r, kernel_s)
        w_diag = weights * win

        # Code correct up to here

        x = np.ones((n_eval, n_eval, n_obs, 3))
        for i in range(n_eval):
            for j in range(n_eval):
                x[:, j, :, 1, None] = (t_pairs[None, :, 0] - r_t_eval[None,:,0])**2
                x[i, :, :, 2, None] = t_pairs[None, :, 1] - r_t_eval[None,:,0]

        print(f"Shape of x: {x[0,0]}")

        x_t = np.transpose(x, (0, 1, 3, 2))
        xtw = x_t * w_diag[:, :, None, :]
        xtwx = xtw @ x
        xtwy = xtw @ cov_values

        beta = np.linalg.pinv(xtwx) @ xtwy

        cov = beta[:, :, 0]
        cov_t = np.transpose(cov, (1, 0, 2))

        return np.array((cov + cov_t) / 2.0)







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

            # The following correction is a practical empirical correction.
            # This is inspired by the fact that, although the Gaussian kernel
            # gives good results for irregular data, the fact that it has
            # infinite support (nonzero weights for all points) can lead to
            # over-smoothing. The following term has the objective of slightly
            # correcting this effect. This can also be seen in the PACE package
            # in Matlab.
            if self.kernel_mean == gaussian_kernel:
                self.bandwidth_mean_ *= 1.1

        print(f"Selected bandwidth for mean: {self.bandwidth_mean_}. ")

        self.mean_ = self._mean_lls(
            self.bandwidth_mean_,
            time_points,
            x_work.points,
            x_work.values,
            self.kernel_mean,
        )

        # print(f"Mean: {self.mean_}")

        raw_cov_coords, raw_cov_values, cov_subj_idx, win, raw_diag_coords, raw_diag_values = (
            self._compute_raw_covariances(
                x_work,
                self.mean_,
                time_points,
            )
        )

        # Create n-dimensional work grid to calculate covariance surface in
        # Create grid of domain points for covariance evaluation
        axes = [
            np.linspace(start, end, self.n_grid_points)
            for start, end in x_work.domain_range
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        t_eval = np.stack([m.ravel() for m in mesh], axis=-1)

        if self.bandwidth_cov_ is None:
            cov_grid = np.linspace(
                t_eval[0],
                t_eval[-1],
                self.bw_cov_n_grid_points,
            )

            # bw_candidates = np.array([
            #     1.7000,2.1653,2.7580,3.5129,4.4744,
            #     5.6991,7.2590,9.2459,11.7766,15.000,
            # ])
            # print(f"Bandwidth candidates: {bw_candidates}")

            # gcv_scores = []
            # for bw in bw_candidates:
            #     gcv_score = self._cov_gcv_score(
            #         bw,
            #         t_eval,
            #         raw_cov_coords,
            #         raw_cov_values,
            #         win,
            #         time_points,
            #     )
            #     gcv_scores.append(gcv_score)
            # print(f"Bandwidth candidates: {bw_candidates}")
            # print(f"GCV scores: {gcv_scores}")

            # self.bandwidth_cov_ = bw_candidates[np.argmin(gcv_scores)]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.bandwidth_cov_ = minimize_scalar(
                    self._cov_gcv_score,
                    args=(
                        cov_grid,
                        raw_cov_coords,
                        raw_cov_values,
                        win,
                        time_points,
                    ),
                    bounds=self.bandwidth_cov_interval_,
                    method="bounded",
                    tol=1e-1,
                ).x

            # The following correction is a practical empirical correction.
            # This is inspired by the fact that, although the Gaussian kernel
            # gives good results for irregular data, the fact that it has
            # infinite support (nonzero weights for all points) can lead to
            # over-smoothing. The following term has the objective of slightly
            # correcting this effect. This can also be seen in the PACE package
            # in Matlab.
            if self.kernel_cov == gaussian_kernel:
                self.bandwidth_cov_ *= 1.1

        print(f"Selected bandwidth for covariance: {self.bandwidth_cov_}. ")

        self.covariance_ = self._cov_lls(
            self.bandwidth_cov_,
            t_eval,
            t_eval,
            raw_cov_coords,
            raw_cov_values,
            win,
        )

        no_opt, fve, eigenvalues = self._get_pc(self.covariance_, self.n_components)
        self.n_components = no_opt
        self.explained_variance_ration = fve
        self.explained_variance_ = eigenvalues

        print(f"Optimal number of components: {no_opt}")

        if self.assume_noisy:
            self._get_sigma2(
                self.bandwidth_cov_,
                t_eval,
                raw_cov_coords,
                raw_cov_values,
                raw_diag_coords,
                raw_diag_values,
                win,
            )







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

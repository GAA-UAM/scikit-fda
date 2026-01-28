"""FPCA through Condictional Expectation Module."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

import numpy as np
from numpy import trapezoid
from scipy.interpolate import CloughTocher2DInterpolator, make_interp_spline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...representation import FData
from ...representation.irregular import FDataGrid, FDataIrregular
from ...typing._numpy import NDArrayFloat, NDArrayInt

KernelFunction = Callable[[NDArrayFloat], NDArrayFloat]


class RawCovarianceLists(NamedTuple):
    t_1: list[list[float]]
    t_2: list[list[float]]
    raw_cov: list[NDArrayFloat]
    subj_idx: list[int]


class RawCovarianceResult(NamedTuple):
    t_pairs_neq: NDArrayFloat
    f_raw_cov_neq: NDArrayFloat
    subj_idx: NDArrayInt
    weights: NDArrayFloat
    t_pairs_eq: NDArrayFloat
    f_raw_cov_eq: NDArrayFloat


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


class PACE(  # noqa: WPS230
    InductiveTransformerMixin[FData, NDArrayFloat, object],
    BaseEstimator,
):
    r"""
    FPCA through conditional expectation.

    Class that implements functional principal component analysis through
    conditional expectation. This method is native to FDataIrregular.

    For more information about the theoretical foundation for this algorithm,
    see :footcite:t:`yao+muller+wang_2005_pace`.

    Parameters:
        n_components: If parameter is an integer, it refers to the number of
            principal components to keep from functional principal component
            analysis. If parameter is a float in the range (0.0, 1.0), it
            refers to the minimum proportion of variance explained by the
            selected principal components. Defaults to ``None`` (maximum number
            of principal components that can be extracted).
        assume_noisy: Set to ``False`` when the data is assumed to be
            noiseless. Otherwise, when smoothing the covariance surface, the
            diagonal will be treated separately. Defaults to ``True``.
        kernel_mean: Callable vectorized univariate smoothing kernel function
            for the mean, of the form :math:`K(t)`, where :math:`t` are the
            n-dimensional time point, with n being the dimension of the domain.
            Defaults to a Gaussian kernel.
        bandwidth_mean: Bandwidth to use in the smoothing kernel for the mean.
            If a float is given, it is used as the bandwidth. If a tuple is
            given, it is used as the bandwidth search range, and the bandwidth
            is calculated using the GCV method.
        kernel_cov: Callable vectorized univariate smoothing kernel function
            for the covariance and calculations regarding its diagonal. It
            should have the form :math:`K(t)`, where :math:`t` are the
            n-dimensional time point, with n being the dimension of the domain.
            To smooth the covariance, each value in the two directions will be
            calculated with the function and the two values will be multiplied,
            acting as an isotropic kernel. Defaults to a Gaussian kernel.
        bandwidth_cov: Bandwidth to use in the smoothing kernel for the
            covariance. If a float is given, it is used as the bandwidth. If a
            tuple is given, it is used as the bandwidth search range, and the
            bandwidth is calculated using the GCV method.
        bw_cov_n_grid_points: Number of grid points to calculate the bandwidth
            for the covariance. This parameter's main purpose is to reduce the
            computational cost of the GCV method. If the parameter
            ``bandwidth_cov`` is provided as a float, this parameter is
            ignored. Defaults to ``30``.
        n_grid_points: Number of grid points to calculate the covariance and,
            subsequently, the eigenfunctions for better approximations. The
            final FPC scores will be given in the original grid points.
            Defaults to ``51``.
        boundary_effect_interval: A 2-element float vector indicating the
            percentage of the time points to be considered as left and right
            boundary regions of the time window of observations. Defaults to
            ``(0.0, 1.0)``, that is, the whole time window.
        variance_error_interval: A 2-element float vector in :math:`[0.0, 1.0]`
            indicating the percent of data truncated during :math:`\sigma^2`
            calculation. Defaults to ``(0.25, 0.75)``, as is suggested in
            footcite:t:`staniswalis+lee_1998_nonparametric_regression`.

    Attributes:
        components\_: FDataGrid that contains the principal components.
        explained_variance\_ : Array that contains the amount of variance
            explained by each of the selected components.
        explained_variance_ratio\_ : Array that contains the percentage
            of variance explained by each principal component.
        mean\_: FDataGrid that contains the smoothed mean of the data.
        bandwidth_mean\_: Calculated or user-given bandwidth used for the mean.
        covariance\_: Matrix of shape (``n_grid_points``, ``n_grid_points``,
            codomain dimension) that contains the covariance of the data.
        t_covariance\_: Matrix of shape (``n_grid_points``,
            domain dimension) that contains the time points of the covariance.
        bandwidth_cov\_: Calculated or user-given bandwidth used for the
            covariance.
        sigma2\_: Calculated error of the covariance.

    Examples:
        >>> import numpy as np
        >>> from skfda.representation import FDataIrregular
        >>> from skfda.preprocessing.dim_reduction import PACE
        >>>
        >>> points = np.array([0.0, 1.0, 0.0, 1.0])
        >>> values = np.array([1.0, 0.0, 0.0, 2.0])
        >>> start_indices = np.array([0, 2])
        >>>
        >>> fd = FDataIrregular(
        ...     points=points,
        ...     values=values,
        ...     start_indices=start_indices,
        ... )
        >>> pace = PACE(
        ...     n_components=2,
        ...     bandwidth_mean=np.array([0.1, 10]),
        ...     bandwidth_cov=np.array([0.1, 10]),
        ... )
        >>> scores = pace.fit_transform(fd)
        >>> expected = np.array([
        ...     [-0.04886943, -0.00023432],
        ...     [ 0.04886943,  0.00023432],
        ... ])
        >>> np.allclose(scores, expected)
        True
        >>> round(float(pace.sigma2_), 3)
        0.973

    References:
        .. footbibliography::
    """

    def _check_bandwidth(
        self,
        bandwidth: float | NDArrayFloat,
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
        if isinstance(bandwidth, float) and bandwidth <= 0:
            error_msg = "Given bandwidth values must be positive."
            raise ValueError(error_msg)

        tuple_length = 2

        if isinstance(bandwidth, Sequence) and (
            len(bandwidth) != tuple_length
            or (not all(isinstance(b, (float, int)) for b in bandwidth))
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

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_components: float | None = None,
        assume_noisy: bool = True,
        kernel_mean: KernelFunction = gaussian_kernel,
        bandwidth_mean: float | NDArrayFloat,
        kernel_cov: KernelFunction = gaussian_kernel,
        bandwidth_cov: float | NDArrayFloat,
        bw_cov_n_grid_points: int = 30,
        n_grid_points: int = 51,
        boundary_effect_interval: Sequence[float] = (0.0, 1.0),
        variance_error_interval: Sequence[float] = (0.25, 0.75),
    ) -> None:
        if n_components is None:
            n_components = 1.0
        else:
            int_leq_0 = isinstance(n_components, int) and n_components <= 0
            float_out_range = not isinstance(n_components, int) and (
                n_components <= 0.0 or n_components >= 1.0
            )
            if int_leq_0 or float_out_range:
                error_msg = (
                    "n_components must be an integer or a float in (0.0, 1.0)."
                )
                raise ValueError(error_msg)

        bandwidth_mean_, bandwidth_mean_interval_ = self._check_bandwidth(
            bandwidth_mean,
        )

        bandwidth_cov_, bandwidth_cov_interval_ = self._check_bandwidth(
            bandwidth_cov,
        )

        if n_grid_points <= 0 or bw_cov_n_grid_points <= 0:
            error_msg = "Grid points must be positive or None."
            raise ValueError(error_msg)

        tuple_length = 2
        boundary_incorrect = (
            len(boundary_effect_interval) != tuple_length
            or (boundary_effect_interval[0] < 0)
            or (boundary_effect_interval[1] > 1)
            or (boundary_effect_interval[0] >= boundary_effect_interval[1])
        )
        variance_interval_incorrect = (
            len(variance_error_interval) != tuple_length
            or (variance_error_interval[0] < 0)
            or (variance_error_interval[1] > 1)
            or (variance_error_interval[0] >= variance_error_interval[1])
        )
        if boundary_incorrect or variance_interval_incorrect:
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

    def _compute_cut_bounds(
        self,
        data: FDataIrregular,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """
        Compute slicing bounds for the given FDataIrregular object.

        Applies the boundary effect interval to compute new lower and upper
        bounds for each coordinate dimension, based on the original domain.

        Args:
            data: The FDataIrregular object whose domain is used to compute
                the slicing bounds.

        Returns:
            A tuple (a_bounds, b_bounds) of arrays representing the lower and
            upper bounds to apply when slicing each coordinate dimension.
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

        a_bounds = np.array(start_cut)
        b_bounds = np.array(end_cut)

        return a_bounds, b_bounds

    def _filter_fdata_points(
        self,
        data: FDataIrregular,
        a_bounds: NDArrayFloat,
        b_bounds: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Filter observation points and values within the specified bounds.

        For each trajectory in the irregular dataset, retains only the
        time-location pairs that lie within the interval defined by a_bounds
        and b_bounds in all coordinate dimensions.

        Args:
            data: The original FDataIrregular object to be filtered.
            a_bounds: Lower slicing bounds per dimension.
            b_bounds: Upper slicing bounds per dimension.

        Returns:
            A tuple (filtered_points, filtered_values, filtered_start_indices)
            containing the sliced data points, corresponding values, and
            updated trajectory start indices.
        """
        start_indices = data.start_indices
        end_indices = np.append(start_indices[1:], len(data.points))

        new_points = []
        new_values = []
        new_start_indices = [0]

        for start, end in zip(start_indices, end_indices, strict=True):
            pts = data.points[start:end, :]
            values = data.values[start:end, :]
            mask = np.all((pts >= a_bounds) & (pts <= b_bounds), axis=1)

            new_points.append(pts[mask])
            new_values.append(values[mask])
            new_start_indices.append(new_start_indices[-1] + len(pts[mask]))

        filtered_points: NDArrayFloat = np.concatenate(new_points, axis=0)
        filtered_values: NDArrayFloat = np.concatenate(new_values, axis=0)
        filtered_start_indices: NDArrayFloat = np.array(
            new_start_indices[:-1],
            dtype=np.uint32,
        )

        return filtered_points, filtered_values, filtered_start_indices

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
        a_bounds, b_bounds = self._compute_cut_bounds(data)

        points, values, start_indices = self._filter_fdata_points(
            data,
            a_bounds,
            b_bounds,
        )

        cut_domain_range = tuple(
            (float(a), float(b))
            for a, b in zip(
                a_bounds,
                b_bounds,
                strict=True,
            )
        )

        return FDataIrregular(
            points=points,
            values=values,
            start_indices=start_indices,
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

    def _compute_local_estimate(
        self,
        xi: NDArrayFloat,
        yi: NDArrayFloat,
        wi: NDArrayFloat,
        d: int,
        epsilon: float,
    ) -> NDArrayFloat:
        """
        Compute local linear smoother estimate at a single point.

        TODO comment function
        """
        win = wi[:, None]

        k0 = np.sum(wi)
        k1 = np.sum(win * xi, axis=0)
        k2 = np.einsum("ni,nj->ij", win * xi, xi)

        s0 = np.sum(win * yi, axis=0)
        s1 = np.einsum("ni,nj->ij", win * xi, yi)

        # Solve linear system for beta0 (intercept)
        # Build left-hand matrix and right-hand side
        xtwx = np.block(
            [
                [np.array([[k0]]), k1[None, :]],
                [k1[:, None], k2],
            ],
        ) + epsilon * np.eye(d + 1)
        xtwy = np.vstack([s0[None, :], s1])

        beta = np.linalg.solve(xtwx, xtwy)
        return cast("NDArrayFloat", beta[0])  # intercept term

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
            Array with smooth estimates for each query point.
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
            estimates[i] = self._compute_local_estimate(
                xi=diffs[i],
                yi=y_obs,
                wi=weights[i],
                d=d,
                epsilon=epsilon,
            )

        return estimates

    def _collect_raw_covariance_lists(
        self,
        points: NDArrayFloat,
        values: NDArrayFloat,
        start_indices: NDArrayInt,
        end_indices: NDArrayInt,
        time_points: NDArrayFloat,
        mean: NDArrayFloat,
    ) -> RawCovarianceLists:
        """
        Collect raw covariance components as lists from all trajectories.

        Returns:
            t_1: List of first time points in each pair.
            t_2: List of second time points in each pair.
            raw_cov: List of raw covariance matrices (outer products).
            subj_idx: List of subject indices.
        """
        t_1, t_2, raw_cov, subj_idx = [], [], [], []

        for i, start_i in enumerate(start_indices):
            p_i = points[start_i : end_indices[i]]
            v_i = values[start_i : end_indices[i]]

            _, indices = cKDTree(time_points).query(p_i)
            mean_proj = mean[indices]
            r_i = v_i - mean_proj

            for j, p_ij in enumerate(p_i):
                for k, p_ik in enumerate(p_i):
                    t_1.append([float(p_ij)])
                    t_2.append([float(p_ik)])
                    raw_cov.append(r_i[j] * r_i[k])
                    subj_idx.append(i)

        return RawCovarianceLists(
            t_1=t_1,
            t_2=t_2,
            raw_cov=raw_cov,
            subj_idx=subj_idx,
        )

    def _compute_raw_covariance_arrays(
        self,
        points: NDArrayFloat,
        values: NDArrayFloat,
        start_indices: NDArrayInt,
        end_indices: NDArrayInt,
        time_points: NDArrayFloat,
        mean: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayInt]:
        """
        Compute all raw covariance data and reshape into arrays.

        Args:
            points: All data points (n_total, d).
            values: All data values (n_total, q).
            start_indices: Start indices for each subject.
            end_indices: End indices for each subject.
            time_points: Locations where the mean is defined.
            mean: Values of the mean function.

        Returns:
            t_pairs: Array of shape (n_pairs, 2, d) with all time point pairs.
            f_raw_cov: Array of shape (n_pairs, q, q) with raw covariances.
            subj_idx: Array of shape (n_pairs,) with subject indices.
        """
        t_1, t_2, raw_cov, subj_idx = self._collect_raw_covariance_lists(
            points,
            values,
            start_indices,
            end_indices,
            time_points,
            mean,
        )

        # Convert to arrays
        t_pairs = np.array([t_1, t_2]).squeeze().T
        t_pairs = t_pairs.reshape(t_pairs.shape[0], 2, -1)

        f_raw_cov = np.array(raw_cov)
        subj_idx_array = np.array(subj_idx, dtype=np.uint32)

        return t_pairs, f_raw_cov, subj_idx_array

    def _compute_raw_covariances(
        self,
        x_work: FDataIrregular,
        mean: NDArrayFloat,
        time_points: NDArrayFloat,
        *,
        assume_noisy: bool,
    ) -> RawCovarianceResult:
        """
        Compute raw covariances for irregular data.

        Compute raw covariances for irregular data, filtering duplicates based
        on the ``assume_noisy`` parameter.

        Args:
            x_work: FDataIrregular object containing the data.
            mean: Mean function values.
            time_points: Time points for the mean.
            assume_noisy: If True, the covariance is computed assuming noise.

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

        t_pairs, f_raw_cov, subj_idx = self._compute_raw_covariance_arrays(
            points,
            values,
            start_indices,
            end_indices,
            time_points,
            mean,
        )

        if assume_noisy:
            t_neq = np.where(t_pairs[:, 0] != t_pairs[:, 1])[0]
            t_eq = np.where(t_pairs[:, 0] == t_pairs[:, 1])[0]
            t_pairs_neq = t_pairs[t_neq]
            f_raw_cov_neq = f_raw_cov[t_neq]
            t_pairs_eq = t_pairs[t_eq][:, 0]
            f_raw_cov_eq = f_raw_cov[t_eq]
        else:
            t_pairs_neq = t_pairs
            f_raw_cov_neq = f_raw_cov
            t_pairs_eq = np.array([])
            f_raw_cov_eq = np.array([])

        win = np.ones(len(f_raw_cov_neq))

        return RawCovarianceResult(
            t_pairs_neq=t_pairs_neq,
            f_raw_cov_neq=f_raw_cov_neq,
            subj_idx=subj_idx,
            weights=win,
            t_pairs_eq=t_pairs_eq,
            f_raw_cov_eq=f_raw_cov_eq,
        )

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

        # Interpolate at the grid points
        interpolator = CloughTocher2DInterpolator(grid_points, g_hat.ravel())
        g_hat_int = interpolator(cov_coords.squeeze())

        # Calculate residual sum of squares (RSS)
        rss = np.sum(
            (cov_values.squeeze() - g_hat_int)
            * (cov_values.squeeze() - g_hat_int).T,
        )

        # Calculate pairwise distances between points
        domain_diff = np.max(pdist(time_points))
        k0 = self.kernel_cov(np.zeros((1, 1, cov_coords.shape[2])))[0]
        n_obs = len(cov_values)
        if n_obs == 0:
            error_msg = (
                "Unable to perform computations with one measurement per "
                "observation on noisy data."
            )
            raise ValueError(error_msg)
        # Normalize by number of observations and bandwidth
        denom = 1 - (1 / n_obs) * ((domain_diff * k0) / h) ** 2

        return float(rss / denom**2) if denom > 0 else np.inf

    def _compute_cov_weights(
        self,
        h: float,
        r_eval: NDArrayFloat,
        s_eval: NDArrayFloat,
        t_pairs: NDArrayFloat,
        win: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Compute the kernel weight matrix for local linear covariance smoothing.

        This method calculates the product of kernel weights centered at the
        evaluation points `r_eval` and `s_eval`, scaled by the bandwidth `h`,
        and weighted by the observation weights `win`.

        Args:
            h: Bandwidth parameter for the kernel.
            r_eval: Array of shape (n_eval, d) representing evaluation points
                in the r-direction.
            s_eval: Array of shape (n_eval, d) representing evaluation points
                in the s-direction.
            t_pairs: Array of shape (n_obs, 2, d) with observed time point
                pairs.
            win: Array of shape (n_obs,) with observation weights.

        Returns:
            Array of shape (n_eval, n_eval, n_obs) with kernel weight products
                for each evaluation pair.
        """
        diff_r = (t_pairs[:, 0, None] - r_eval[None, :]) / h
        diff_s = (t_pairs[:, 1, None] - s_eval[None, :]) / h
        kernel_r = self.kernel_cov(diff_r).T
        kernel_s = self.kernel_cov(diff_s).T
        return cast(
            "NDArrayFloat", np.einsum("ik,jk->ijk", kernel_r, kernel_s) * win
        )

    def _build_design_matrix(
        self,
        t_pairs: NDArrayFloat,
        r_eval: NDArrayFloat,
        s_eval: NDArrayFloat,
        n_eval: int,
        n_obs: int,
    ) -> NDArrayFloat:
        """
        Construct the design matrix for local linear regression.

        The design matrix contains a constant term and linear terms for both
        r- and s-directions, centered at the evaluation points. It is used in
        the weighted least squares estimation of the covariance surface.

        Args:
            t_pairs: Array of shape (n_obs, 2, d) with observed time point
                pairs.
            r_eval: Array of shape (n_eval, d) with r-direction evaluation
                points.
            s_eval: Array of shape (n_eval, d) with s-direction evaluation
                points.
            n_eval: Number of evaluation points
                (i.e., len(r_eval) == len(s_eval)).
            n_obs: Number of observed point pairs.

        Returns:
            Array of shape (n_eval, n_eval, n_obs, 3) representing the design
            matrix with columns [1, t_r - r_eval, t_s - s_eval].
        """
        x = np.ones((n_eval, n_eval, n_obs, 3))
        for i in range(n_eval):
            for j in range(n_eval):
                x[:, j, :, 1, None] = t_pairs[None, :, 0] - r_eval[:, None]
                x[i, :, :, 2, None] = t_pairs[None, :, 1] - s_eval[:, None]
        return x

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
        active = np.nonzero(win)[0]
        t_pairs = cov_coords[active, :]
        cov_values = cov_values[active]
        win = win[active]

        n_eval, _ = r_eval.shape
        n_obs, _ = cov_values.shape

        weights = self._compute_cov_weights(h, r_eval, s_eval, t_pairs, win)
        x = self._build_design_matrix(t_pairs, r_eval, s_eval, n_eval, n_obs)

        x_t = np.transpose(x, (0, 1, 3, 2))
        xtw = x_t * weights[:, :, None, :]
        xtwx = xtw @ x
        xtwy = xtw @ cov_values

        beta = np.linalg.pinv(xtwx) @ xtwy

        cov = beta[:, :, 0]
        cov_t = np.transpose(cov, (1, 0, 2))
        return np.array((cov + cov_t) / 2.0)  # noqa: WPS432

    def _sort_and_clip_eigenpairs(
        self,
        eigenvalues: NDArrayFloat,
        eigenvectors: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """
        Sort and non-negatively clip eigenvalues, reorder eigenvectors.

        TODO comment method
        """
        eigenvalues = np.maximum(eigenvalues, 0)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    def _normalize_and_orient(
        self,
        eigenvectors: NDArrayFloat,
        t: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Normalize and align the eigenvectors.

        TODO comment method
        """
        for i in range(eigenvectors.shape[1]):
            phi_i = eigenvectors[:, i]
            norm = np.sqrt(trapezoid(phi_i**2, x=t))
            phi_i /= norm
            if phi_i[1] < phi_i[0]:
                phi_i *= -1
            eigenvectors[:, i] = phi_i
        return eigenvectors

    def _interpolate_and_normalize_basis(
        self,
        t_eigen: NDArrayFloat,
        eigenvectors: NDArrayFloat,
        target_grid: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Spline-interpolate and normalize eigenfunctions on new grid.

        TODO comment method
        """
        n_points = len(target_grid)
        n_components = eigenvectors.shape[1]
        phi = np.empty((n_points, n_components))
        for i in range(n_components):
            spline = make_interp_spline(t_eigen, eigenvectors[:, i])
            phi[:, i] = spline(target_grid)
            phi[:, i] /= np.sqrt(trapezoid(phi[:, i] ** 2, x=target_grid))
        return phi

    def _get_pc(
        self,
        cov_matrix: NDArrayFloat,
        n_components: float,
    ) -> tuple[int, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Select the number of principal components.

        Args:
            cov_matrix: The smoothed covariance matrix.
            n_components: Threshold for variance explained or number to retain.

        Returns:
            Number of components, cumulative FVE, eigenvalues, eigenfunctions.
        """
        t_eigen = self.t_covariance_.squeeze()
        h = (t_eigen.max() - t_eigen.min()) / (len(t_eigen) - 1)
        cov = cov_matrix.squeeze()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        if not np.all(np.isfinite(eigenvalues)) or (
            not np.all(np.isfinite(eigenvectors))
        ):
            error_msg = (
                "Covariance matrix has invalid eigenvalues or eigenvectors."
            )
            raise ValueError(error_msg)

        eigenvalues, eigenvectors = self._sort_and_clip_eigenpairs(
            eigenvalues,
            eigenvectors,
        )

        fve = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        if isinstance(n_components, int):
            if n_components > len(eigenvalues):
                error_msg = (
                    "The sample size must be bigger than the number of "
                    "components"
                )
                raise AttributeError(error_msg)
            n_selected_components = n_components
        else:
            # Find the optimal number of components
            n_selected_components = np.where(fve >= n_components)[0][0] + 1

        eigenvalues = eigenvalues[:n_selected_components]
        eigenvectors = eigenvectors[:, :n_selected_components]

        lambda_ = h * eigenvalues
        eigenvectors /= np.sqrt(h)

        eigenvectors = self._normalize_and_orient(
            eigenvectors, self.t_covariance_.squeeze()
        )

        phi = self._interpolate_and_normalize_basis(
            t_eigen,
            eigenvectors,
            self.mean_.grid_points[0],
        )

        return n_selected_components, fve, lambda_, phi.T

    def _get_sigma2(
        self,
        h: float,
        t_eval: NDArrayFloat,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        t_diag: NDArrayFloat,
        cov_diag: NDArrayFloat,
        win: NDArrayFloat,
        domain_range: NDArrayFloat,
    ) -> float:
        """
        Estimate the variance of the covariance matrix.

        Args:
            h: Bandwidth for the kernel. It is the same one used to smooth the
                covariance.
            t_eval: Query points where diagonal is evaluated, expected to be
                (num eval points x 1)-dimensional.
            cov_coords: Coordinates of the covariance.
            cov_values: Values of the covariance.
            t_diag: Coordinates of the raw covariance diagonal.
            cov_diag: Values of the raw covariance diagonal.
            win: Weights for the covariance.
            domain_range: Domain range of the data.

        Returns:
            The estimated variance.
        """
        smooth_diag = self._mean_lls(
            h,
            t_eval,
            t_diag,
            cov_diag,
            self.kernel_cov,
        )

        rotated_cov_diag = self._rotated_cov_lls(
            h,
            t_eval,
            t_eval,
            cov_coords,
            cov_values,
            win,
        )

        min_domain, max_domain = domain_range[0]
        domain_range = max_domain - min_domain
        a = min_domain + domain_range * self.variance_error_interval[0]
        b = max_domain - domain_range * (1 - self.variance_error_interval[1])

        x = t_eval.squeeze()
        y = (smooth_diag - rotated_cov_diag).squeeze()

        # Mask for integration interval [a, b]
        mask = (x > a) & (x < b)

        # Perform Simpson integration and scale
        sigma2 = trapezoid(y[mask], x[mask]) * 2 / domain_range

        if sigma2 < 0:
            warnings.warn(
                "The estimated variance is negative. Setting it to 0.",
                UserWarning,
                stacklevel=2,
            )
            sigma2 = 0
        return float(sigma2)

    def _rotate_coordinates(
        self,
        cov_coords: NDArrayFloat,
        r_eval: NDArrayFloat,
        s_eval: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """
        Rotate covariance and evaluation coordinates using rotation.

        TODO comment method
        """
        r_mat = np.sqrt(2) / 2 * np.array([[1, 1], [-1, 1]])
        r_cov_coords = np.einsum("ijk,jk->ik", cov_coords, r_mat)
        r_cov_coords = r_cov_coords[:, :, np.newaxis]

        t_eval = np.stack((r_eval, s_eval), axis=1).squeeze()
        r_t_eval = t_eval @ r_mat
        r_t_eval = r_t_eval[:, :, np.newaxis]

        return r_cov_coords, r_t_eval

    def _compute_rotated_cov_weights(
        self,
        h: float,
        t_pairs: NDArrayFloat,
        r_t_eval: NDArrayFloat,
        win: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Compute weights for rotated covariance smoothing.

        TODO comment method
        """
        diff_r = (t_pairs[:, 0, None] - r_t_eval[None, :, 0]) / h
        diff_s = (t_pairs[:, 1, None] - r_t_eval[None, :, 1]) / h
        kernel_r = self.kernel_cov(diff_r).T
        kernel_s = self.kernel_cov(diff_s).T
        weights = np.einsum("ik,jk->ijk", kernel_r, kernel_s)
        weighted_diag = (weights * win)[
            np.arange(weights.shape[1]),
            np.arange(weights.shape[1]),
        ]
        return cast("NDArrayFloat", weighted_diag)

    def _build_rotated_design_matrix(
        self,
        t_pairs: NDArrayFloat,
        r_t_eval: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Build design matrix for rotated coordinates.

        TODO comment method
        """
        n_eval = r_t_eval.shape[0]
        n_obs = t_pairs.shape[0]
        x = np.ones((n_eval, n_obs, 3))

        for i in range(n_eval):
            delta_r0 = t_pairs[:, 0, 0] - r_t_eval[i, 0]
            delta_r1 = t_pairs[:, 1, 0] - r_t_eval[i, 1]

            x[i, :, 1] = delta_r0**2
            x[i, :, 2] = delta_r1

        return x

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
        r_cov_coords, r_t_eval = self._rotate_coordinates(
            cov_coords, r_eval, s_eval
        )

        active = np.nonzero(win)[0]
        t_pairs = r_cov_coords[active]
        cov_values = cov_values[active]
        win = win[active]

        weights = self._compute_rotated_cov_weights(h, t_pairs, r_t_eval, win)
        design_matrix = self._build_rotated_design_matrix(t_pairs, r_t_eval)

        x_t = np.transpose(design_matrix, (0, 2, 1))
        xtw = x_t * weights[:, None, :]
        xtwx = xtw @ design_matrix
        xtwy = xtw @ cov_values

        beta = np.linalg.pinv(xtwx) @ xtwy

        return np.array(beta[:, 0])

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,  # noqa: ARG002
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

        if self.boundary_effect_interval == (0.0, 1.0):  # noqa: WPS358
            x_work = X
        else:
            # Slice the data to remove the boundary effect
            x_work = self._slice_fdata_irregular(X)

        # The mean has to be calculated with the points within the boundary
        # region, but over the whole domain
        t_eval = np.sort(np.unique(X.points, axis=0), axis=0)

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
                gaussian_correction_term = 1.1
                self.bandwidth_mean_ *= gaussian_correction_term

        mean = self._mean_lls(
            self.bandwidth_mean_,
            t_eval,
            x_work.points,
            x_work.values,
            self.kernel_mean,
        )

        self.mean_ = FDataGrid(
            data_matrix=mean.reshape(1, -1, 1),
            grid_points=t_eval.ravel(),
            domain_range=X.domain_range,
            dataset_name=X.dataset_name,
            argument_names=X.argument_names,
            coordinate_names=X.coordinate_names,
            sample_names=["Mean function"],
            extrapolation=X.extrapolation,
            interpolation=X.interpolation,
        )

        raw_cov_data = self._compute_raw_covariances(
            x_work,
            self.mean_.data_matrix[0],
            t_eval,
            assume_noisy=self.assume_noisy,
        )

        raw_cov_coords, raw_cov_values = raw_cov_data[:2]
        win, raw_diag_coords, raw_diag_values = raw_cov_data[3:]

        # Create n-dimensional work grid to calculate covariance surface in
        # Create grid of domain points for covariance evaluation
        axes = [
            np.linspace(start, end, self.n_grid_points)
            for start, end in x_work.domain_range
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.t_covariance_ = np.stack([m.ravel() for m in mesh], axis=-1)

        if self.bandwidth_cov_ is None:
            cov_grid = np.linspace(
                self.t_covariance_[0],
                self.t_covariance_[-1],
                self.bw_cov_n_grid_points,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.bandwidth_cov_ = minimize_scalar(
                    self._cov_gcv_score,
                    args=(
                        cov_grid,
                        raw_cov_coords,
                        raw_cov_values,
                        win,
                        t_eval,
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
                gaussian_correction_term = 1.1
                self.bandwidth_cov_ *= gaussian_correction_term

        self.covariance_ = self._cov_lls(
            self.bandwidth_cov_,
            self.t_covariance_,
            self.t_covariance_,
            raw_cov_coords,
            raw_cov_values,
            win,
        )

        pc_data = self._get_pc(
            self.covariance_,
            self.n_components,
        )

        n_components, explained_variance_ratio_ = pc_data[:2]
        eigenvalues, phi = pc_data[2:]

        self.n_components = n_components
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.explained_variance_ = eigenvalues

        self.components_: FDataGrid = FDataGrid(
            data_matrix=phi,
            grid_points=self.mean_.grid_points,
            domain_range=X.domain_range,
            dataset_name=X.dataset_name,
            argument_names=X.argument_names,
            coordinate_names=X.coordinate_names,
            sample_names=[f"Eigenfunction {i+1}" for i in range(phi.shape[0])],
            extrapolation=X.extrapolation,
            interpolation=X.interpolation,
        )

        if self.assume_noisy:
            self.sigma2_ = self._get_sigma2(
                self.bandwidth_cov_,
                self.t_covariance_,
                raw_cov_coords,
                raw_cov_values,
                raw_diag_coords,
                raw_diag_values,
                win,
                np.array(x_work.domain_range),
            )
        else:
            self.sigma2_ = 0

        return self

    def subject_fpc_scores(
        self,
        X: FDataIrregular,
        start: int,
        end: int,
        lambda_: NDArrayFloat,
        t_mean: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Compute the functional principal component scores for a specific obs.

        Args:
            X: The functional data object to be analysed.
            start: The starting index of the subject's data.
            end: The ending index of the subject's data.
            lambda_: The eigenvalues of the covariance matrix.
            t_mean: The time points of the mean function.

        Returns:
            Principal component scores for the specified subject.
        """
        points_i = X.points[start:end].squeeze()
        values_i = X.values[start:end].squeeze()
        if points_i.ndim == 0:
            points_i = np.array([points_i])
        m_i = len(points_i)

        # Get indices in t_mean_ corresponding to points_i
        indices = [np.argmin(np.abs(t_mean - pt)) for pt in points_i]
        mu_i = self.mean_.data_matrix[0, indices].squeeze()
        phi_i_raw = self.components_.data_matrix[:, indices]
        phi_i = phi_i_raw[..., 0].T

        num = lambda_ @ phi_i.T
        denom = phi_i @ lambda_ @ phi_i.T + self.sigma2_ * np.eye(m_i)
        try:
            denom_inv = np.linalg.inv(denom)
        except np.linalg.LinAlgError:
            denom_inv = np.linalg.pinv(denom)
        phi_sigma = num @ denom_inv

        # Residuals
        residual_i = values_i - mu_i
        if residual_i.ndim == 0:
            residual_i = np.array([residual_i])

        return np.array(phi_sigma @ residual_i.T)

    def transform(
        self,
        X: FData,
        y: object = None,  # noqa: ARG002
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
        end_indices = np.append(X.start_indices[1:], len(X.points))
        t_mean = self.mean_.grid_points[0].squeeze()

        fpc_scores = np.zeros((len(X.start_indices), int(self.n_components)))
        lambda_ = np.diag(self.explained_variance_)

        if self.assume_noisy is False:
            eps = 1e-8  # small regularization
            self.sigma2_ = eps

        for i, idx in enumerate(X.start_indices):
            fpc_scores[i, :] = self.subject_fpc_scores(
                X,
                start=idx,
                end=end_indices[i],
                lambda_=lambda_,
                t_mean=t_mean,
            )

        return fpc_scores

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
            pc_scores: NDArray (n_samples, n_components).

        Returns:
            A FData object.
        """
        phi = self.components_.data_matrix[..., 0]
        mean = self.mean_.data_matrix.squeeze()
        reconstructed = pc_scores @ phi + mean

        return FDataGrid(
            data_matrix=reconstructed,
            grid_points=self.mean_.grid_points[0].squeeze(),
            domain_range=self.mean_.domain_range,
            dataset_name=self.mean_.dataset_name,
            argument_names=self.mean_.argument_names,
            coordinate_names=self.mean_.coordinate_names,
            extrapolation=self.mean_.extrapolation,
            interpolation=self.mean_.interpolation,
        )

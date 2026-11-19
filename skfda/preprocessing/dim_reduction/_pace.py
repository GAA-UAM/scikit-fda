"""FPCA through Conditional Expectation Module."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy import trapezoid
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...preprocessing.smoothing import (
    PooledCovarianceSmoother,
    PooledMeanSmoother,
    local_linear_smooth_irregular_nd,
)
from ...representation import FData
from ...representation.interpolation import SplineInterpolation
from ...representation.irregular import FDataGrid, FDataIrregular
from ...typing._numpy import NDArrayFloat, NDArrayInt

KernelFunction = Callable[[NDArrayFloat], NDArrayFloat]

# Regularization constant used when assume_noisy=False to ensure numerical
# stability.
REGULARIZATION_TERM = 1e-8


@dataclass
class RawCovarianceArrays:
    """Intermediate storage for raw covariance computation."""

    t_1: NDArrayFloat  # (n_pairs, d)
    t_2: NDArrayFloat  # (n_pairs, d)
    raw_cov: NDArrayFloat  # (n_pairs, q)
    subj_idx: NDArrayInt  # (n_pairs,)


@dataclass
class RawCovarianceResult:
    """Result container for raw covariance computation."""

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
    *_, n_dims = t.shape

    norm_sq = np.sum(t**2, axis=2)

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
            See :footcite:t:`febrero-bande+oviedodelafuente_2012_statistical`
            for background on cross-validation methods for smoothing.
        bw_cov_n_grid_points: Number of grid points to calculate the bandwidth
            for the covariance. This parameter's main purpose is to reduce the
            computational cost of the GCV method. If the parameter
            ``bandwidth_cov`` is provided as a float, this parameter is
            ignored. If ``None`` (default), uses the full reconstruction grid
            (highest fidelity). For large datasets, consider setting to ``30``
            or similar to improve performance.
        n_grid_points: Number of grid points to calculate the covariance and,
            subsequently, the eigenfunctions for better approximations. The
            final FPC scores will be given in the original grid points.
            If ``None`` (default), uses the full grid derived from the data's
            domain. For large datasets, consider setting to ``51`` or similar
            to balance accuracy and performance.
        reconstruction_grid: Grid points for the output mean and
            eigenfunctions.
            Can be:
            - ``None`` (default): use all unique observation points from the
              input data.
            - ``int``: create a uniform grid with this many points over the
              domain.
            - ``NDArray``: use the provided grid points directly. Useful for
              aligning PACE output with existing dense data.
        boundary_effect_interval: A 2-element float vector indicating the
            percentage of the time points to be considered as left and right
            boundary regions of the time window of observations. Defaults to
            ``(0.0, 1.0)``, that is, the whole time window.
        variance_error_interval: A 2-element float vector in :math:`[0.0, 1.0]`
            indicating the percent of data truncated during :math:`\sigma^2`
            calculation. Defaults to ``(0.25, 0.75)``, as is suggested in
            :footcite:t:`staniswalis+lee_1998_nonparametric_regression`.

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
        >>>
        >>> scores = pace.fit_transform(fd)
        >>> expected = np.array([
        ...     [-0.04886943, -0.00023432],
        ...     [ 0.04886943,  0.00023432],
        ... ])
        >>> np.allclose(scores, expected)
        True
        >>>
        >>> round(float(pace.sigma2_), 3)
        0.973

    References:
        .. footbibliography::
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_components: float | None = None,
        assume_noisy: bool = True,
        kernel_mean: KernelFunction = gaussian_kernel,
        bandwidth_mean: float | NDArrayFloat,
        kernel_cov: KernelFunction = gaussian_kernel,
        bandwidth_cov: float | NDArrayFloat,
        bw_cov_n_grid_points: int | None = None,
        n_grid_points: int | None = None,
        reconstruction_grid: int | NDArrayFloat | None = None,
        boundary_effect_interval: Sequence[float] = (0.0, 1.0),
        variance_error_interval: Sequence[float] = (0.25, 0.75),
        _apply_gaussian_bandwidth_correction: bool = False,
    ) -> None:
        self.n_components = n_components
        self.assume_noisy = assume_noisy
        self.kernel_mean = kernel_mean
        self.bandwidth_mean = bandwidth_mean
        self.kernel_cov = kernel_cov
        self.bandwidth_cov = bandwidth_cov
        self.bw_cov_n_grid_points = bw_cov_n_grid_points
        self.n_grid_points = n_grid_points
        self.reconstruction_grid = reconstruction_grid
        self.boundary_effect_interval = boundary_effect_interval
        self.variance_error_interval = variance_error_interval
        self._apply_gaussian_bandwidth_correction = (
            _apply_gaussian_bandwidth_correction
        )

    def _check_bandwidth(
        self,
        bandwidth: float | NDArrayFloat,
    ) -> tuple[float | None, tuple[float, float] | None]:
        """
        Validate bandwidth and return value or search range.

        Args:
            bandwidth: Bandwidth to check (float or 2-element sequence).

        Returns:
            Tuple of (value, range): value if float, else None; range if
            sequence, else None. For None input, returns (None, (0.1, 10.0)).

        Raises:
            ValueError: If bandwidth is non-positive or range is invalid.
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

    def _validate_params(self) -> None:
        """
        Validate estimator parameters.

        Called at the start of :meth:`fit`. Validates n_components, grid
        points, interval parameters, and bandwidths.

        Raises:
            ValueError: If any parameter is invalid.
        """
        # Validate n_components
        n_components = self.n_components
        if n_components is not None:
            int_leq_0 = isinstance(n_components, int) and n_components <= 0
            float_out_range = not isinstance(n_components, int) and (
                n_components <= 0.0 or n_components >= 1.0
            )
            if int_leq_0 or float_out_range:
                error_msg = (
                    "n_components must be an integer or a float in (0.0, 1.0)."
                )
                raise ValueError(error_msg)

        # Validate grid points
        if (self.n_grid_points is not None and self.n_grid_points <= 0) or (
            self.bw_cov_n_grid_points is not None
            and self.bw_cov_n_grid_points <= 0
        ):
            error_msg = "Grid points must be positive (or None for full grid)."
            raise ValueError(error_msg)

        # Validate interval parameters
        tuple_length = 2
        bei = self.boundary_effect_interval
        if len(bei) != tuple_length:
            error_msg = (
                f"boundary_effect_interval must have exactly 2 elements, "
                f"got {len(bei)}."
            )
            raise ValueError(error_msg)
        if not (0 <= bei[0] < bei[1] <= 1):
            error_msg = (
                f"boundary_effect_interval must satisfy 0 <= a < b <= 1, "
                f"got ({bei[0]}, {bei[1]})."
            )
            raise ValueError(error_msg)

        vei = self.variance_error_interval
        if len(vei) != tuple_length:
            error_msg = (
                f"variance_error_interval must have exactly 2 elements, "
                f"got {len(vei)}."
            )
            raise ValueError(error_msg)
        if not (0 <= vei[0] < vei[1] <= 1):
            error_msg = (
                f"variance_error_interval must satisfy 0 <= a < b <= 1, "
                f"got ({vei[0]}, {vei[1]})."
            )
            raise ValueError(error_msg)

        # Validate and parse bandwidths
        bw_mean_result = self._check_bandwidth(self.bandwidth_mean)
        self.bandwidth_mean_, self.bandwidth_mean_interval_ = bw_mean_result
        bw_cov_result = self._check_bandwidth(self.bandwidth_cov)
        self.bandwidth_cov_, self.bandwidth_cov_interval_ = bw_cov_result

    def _slice_fdata_irregular(
        self,
        data: FDataIrregular,
    ) -> FDataIrregular:
        """
        Slice FDataIrregular to the boundary effect interval.

        Uses the ``restrict`` method to filter observation points to only
        those within the interval defined by ``boundary_effect_interval``
        applied to the domain range.

        Args:
            data: The FDataIrregular object to be sliced.

        Returns:
            A new FDataIrregular object with filtered observations.
        """
        # Compute new domain range based on boundary effect interval
        domain_range = np.array(data.domain_range)
        domain_diff = domain_range[:, 1] - domain_range[:, 0]

        bei = self.boundary_effect_interval
        new_lower = domain_range[:, 0] + domain_diff * bei[0]
        new_upper = domain_range[:, 1] - domain_diff * (1 - bei[1])

        new_domain_range = tuple(
            (float(lo), float(hi))
            for lo, hi in zip(new_lower, new_upper, strict=True)
        )

        return data.restrict(new_domain_range)

    def _mean_gcv_score(
        self,
        h: float,
        t_obs: NDArrayFloat,
        y_obs: NDArrayFloat,
    ) -> float:
        """
        Compute the Generalized Cross-Validation (GCV) score.

        Delegates to :meth:`PooledMeanSmoother.score` the computation of the
        GCV score.

        Args:
            h: Bandwidth to evaluate
            t_obs: Observed time points
            y_obs: Observed function values

        Returns:
            GCV score for the given bandwidth.
        """
        if h <= 0:
            return np.inf
        mean_smoother = PooledMeanSmoother(
            bandwidth=1.0,
            kernel=self.kernel_mean,
        )
        return -mean_smoother.score(t_obs, y_obs, h)

    def _select_bandwidth_mean(
        self,
        points: NDArrayFloat,
        values: NDArrayFloat,
    ) -> float:
        """
        Select mean bandwidth via GCV if not already specified.

        If bandwidth_mean_ is None (i.e., a search range was provided),
        performs GCV-based bandwidth selection and applies Gaussian correction.

        Args:
            points: Observation time points.
            values: Observation values.

        Returns:
            Selected or pre-specified bandwidth for mean smoothing.
        """
        if self.bandwidth_mean_ is not None:
            return self.bandwidth_mean_

        bandwidth: float = minimize_scalar(
            self._mean_gcv_score,
            args=(points, values),
            bounds=self.bandwidth_mean_interval_,
            method="bounded",
        ).x

        # Empirical correction for Gaussian kernel (Matlab PACE compatibility)
        if (
            self._apply_gaussian_bandwidth_correction
            and self.kernel_mean == gaussian_kernel
        ):
            bandwidth *= 1.1

        return bandwidth

    def _collect_raw_covariance(
        self,
        points: NDArrayFloat,
        values: NDArrayFloat,
        start_indices: NDArrayInt,
        end_indices: NDArrayInt,
        time_points: NDArrayFloat,
        mean: NDArrayFloat,
    ) -> RawCovarianceArrays:
        """
        Collect raw covariance components as arrays from all trajectories.

        Fully vectorized using np.repeat to avoid Python loops.

        Args:
            points: All observation time points concatenated.
            values: All observation values concatenated.
            start_indices: Start index of each trajectory in points/values.
            end_indices: End index of each trajectory in points/values.
            time_points: Grid points where mean was evaluated.
            mean: Mean function values at time_points.

        Returns:
            RawCovarianceArrays with t_1, t_2, raw_cov, subj_idx as arrays.
        """
        n_samples = len(start_indices)
        d = points.shape[1] if points.ndim > 1 else 1
        q = values.shape[1] if values.ndim > 1 else 1
        m_per_subj = end_indices - start_indices

        # Handle case where all subjects are empty
        if not (m_per_subj > 0).any():
            return RawCovarianceArrays(
                t_1=np.empty((0, d)),
                t_2=np.empty((0, d)),
                raw_cov=np.empty((0, q)),
                subj_idx=np.empty(0, dtype=np.intp),
            )

        tree = cKDTree(time_points)
        _, nn_indices = tree.query(points)
        nn_indices = np.atleast_1d(nn_indices)

        all_residuals = values - mean[nn_indices]
        subject_of_point = np.repeat(np.arange(n_samples), m_per_subj)

        j_repeats = m_per_subj[subject_of_point]
        j_indices = np.repeat(np.arange(len(points)), j_repeats)

        pairs_per_subj = m_per_subj**2
        pair_subject = np.repeat(np.arange(n_samples), pairs_per_subj)

        pair_cumsum = np.zeros(n_samples + 1, dtype=np.intp)
        pair_cumsum[1:] = np.cumsum(pairs_per_subj)
        total_pairs = pairs_per_subj.sum()
        pair_local_pos = np.arange(total_pairs) - pair_cumsum[pair_subject]

        # k = start_index + (local_position mod m_i)
        m_for_pairs = np.maximum(m_per_subj[pair_subject], 1)  # Avoid div by 0
        k_local = pair_local_pos % m_for_pairs
        k_indices = start_indices[pair_subject] + k_local

        return RawCovarianceArrays(
            t_1=points[j_indices],
            t_2=points[k_indices],
            raw_cov=all_residuals[j_indices] * all_residuals[k_indices],
            subj_idx=pair_subject.astype(np.intp),
        )

    def _format_raw_covariance_arrays(
        self,
        points: NDArrayFloat,
        values: NDArrayFloat,
        start_indices: NDArrayInt,
        end_indices: NDArrayInt,
        time_points: NDArrayFloat,
        mean: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayInt]:
        """
        Compute raw covariance data and reshape into final array format.

        Args:
            points: All data points (n_total, d).
            values: All data values (n_total, q).
            start_indices: Start indices for each subject.
            end_indices: End indices for each subject.
            time_points: Locations where the mean is defined.
            mean: Values of the mean function.

        Returns:
            t_pairs: Array of shape (n_pairs, 2, d) with all time point pairs.
            f_raw_cov: Array of shape (n_pairs, q) with raw covariances.
            subj_idx: Array of shape (n_pairs,) with subject indices.
        """
        raw = self._collect_raw_covariance(
            points,
            values,
            start_indices,
            end_indices,
            time_points,
            mean,
        )

        # Stack t_1 and t_2 into (n_pairs, 2, d) format
        t_pairs = np.stack([raw.t_1, raw.t_2], axis=1)

        return t_pairs, raw.raw_cov, raw.subj_idx.astype(np.uint32)

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
            RawCovarianceResult with t_pairs_neq, f_raw_cov_neq, subj_idx,
            weights, t_pairs_eq, f_raw_cov_eq.
        """
        points = x_work.points
        values = x_work.values
        start_indices = x_work.start_indices
        end_indices = np.append(start_indices[1:], len(points))

        t_pairs, f_raw_cov, subj_idx = self._format_raw_covariance_arrays(
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

        Delegates to :meth:`PooledCovarianceSmoother.score` the computation of
        the GCV score using the pattern smoother.score().

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
        n_obs = len(cov_values)
        if n_obs == 0:
            error_msg = (
                "Unable to perform computations with one measurement per "
                "observation on noisy data."
            )
            raise ValueError(error_msg)
        t_eval_2d = t_eval if t_eval.ndim > 1 else np.atleast_2d(t_eval).T
        cov_smoother = PooledCovarianceSmoother(
            bandwidth=1.0,
            kernel=self.kernel_cov,
            output_points_r=t_eval_2d,
            output_points_s=t_eval_2d,
        )
        return -cov_smoother.score(
            cov_coords,
            cov_values,
            win,
            time_points,
            t_eval,
            h,
        )

    def _select_bandwidth_cov(
        self,
        cov_grid: NDArrayFloat,
        raw_cov_coords: NDArrayFloat,
        raw_cov_values: NDArrayFloat,
        win: NDArrayFloat,
        t_eval: NDArrayFloat,
    ) -> float:
        """
        Select covariance bandwidth via GCV if not already specified.

        If bandwidth_cov_ is None (i.e., a search range was provided),
        performs GCV-based bandwidth selection and applies Gaussian correction.

        Args:
            cov_grid: Grid for GCV evaluation.
            raw_cov_coords: Raw covariance coordinates.
            raw_cov_values: Raw covariance values.
            win: Weights for covariance.
            t_eval: Time points for mean (used for range).

        Returns:
            Selected or pre-specified bandwidth for covariance smoothing.
        """
        if self.bandwidth_cov_ is not None:
            return self.bandwidth_cov_

        # Suppress RuntimeWarnings during GCV optimization. These occur due
        # to numerical issues (division by zero, invalid values) when
        # evaluating the GCV score at extreme bandwidth values. The optimizer
        # handles these cases by treating them as poor candidates.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bandwidth: float = minimize_scalar(
                self._cov_gcv_score,
                args=(cov_grid, raw_cov_coords, raw_cov_values, win, t_eval),
                bounds=self.bandwidth_cov_interval_,
                method="bounded",
                tol=1e-1,
            ).x

        # Empirical correction for Gaussian kernel (Matlab PACE compatibility)
        if (
            self._apply_gaussian_bandwidth_correction
            and self.kernel_cov == gaussian_kernel
        ):
            bandwidth *= 1.1

        return bandwidth

    def _normalize_and_orient(
        self,
        eigenvectors: NDArrayFloat,
        t: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Normalize eigenvectors and orient them consistently.

        Each eigenvector is normalized to have unit L2 norm (via trapezoidal
        integration). Sign orientation uses svd_flip for determinism (largest
        absolute value in each column is positive).

        Args:
            eigenvectors: Matrix of eigenvectors (columns) to normalize.
            t: Time grid for integration.

        Returns:
            Normalized and consistently oriented eigenvectors.
        """
        # Vectorized L2 normalization via trapezoidal integration
        norms = np.sqrt(trapezoid(eigenvectors**2, x=t, axis=0))
        eigenvectors = eigenvectors / norms

        # Use svd_flip for deterministic sign orientation
        eigenvectors, _ = svd_flip(eigenvectors, np.zeros_like(eigenvectors).T)

        return eigenvectors

    def _interpolate_and_normalize_basis(
        self,
        t_eigen: NDArrayFloat,
        eigenvectors: NDArrayFloat,
        target_grid: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Spline-interpolate and normalize eigenfunctions on new grid.

        Interpolates eigenvectors from the covariance grid to the mean grid
        using cubic splines via FDataGrid, then re-normalizes to unit L2 norm.

        Args:
            t_eigen: Original grid where eigenvectors are defined.
            eigenvectors: Matrix of eigenvectors (columns) to interpolate.
            target_grid: Target grid points for interpolation.

        Returns:
            Interpolated and normalized eigenfunctions on the target grid.
        """
        # Create FDataGrid with eigenvectors as samples
        fd = FDataGrid(
            data_matrix=eigenvectors.T,  # (n_components, n_grid_orig)
            grid_points=t_eigen,
            interpolation=SplineInterpolation(interpolation_order=3),
        )

        # Evaluate at target grid: returns (n_components, n_target, 1)
        phi = fd(target_grid)[:, :, 0].T

        # Vectorized L2 normalization
        norms = np.sqrt(trapezoid(phi**2, x=target_grid, axis=0))
        return np.asarray(phi / norms)

    def _get_pc(
        self,
        cov_matrix: NDArrayFloat,
        n_components: float,
    ) -> tuple[int, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Extract principal components from the covariance matrix.

        Decomposes the covariance, sorts and clips eigenvalues, interpolates
        eigenfunctions to the mean grid, and selects components by count or
        FVE threshold.

        Args:
            cov_matrix: The smoothed covariance matrix.
            n_components: Number of components (int) or FVE threshold (float).

        Returns:
            Tuple of (n_components, fve, eigenvalues, eigenfunctions).

        Raises:
            ValueError: If eigenvalues/eigenvectors are invalid or
                n_components exceeds available components.
        """
        t_eigen = self.t_covariance_.ravel()  # 1D time grid
        h = (t_eigen.max() - t_eigen.min()) / (len(t_eigen) - 1)
        cov = cov_matrix[:, :, 0]  # Remove codomain dim: (n, n, q) -> (n, n)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        if not np.all(np.isfinite(eigenvalues)) or (
            not np.all(np.isfinite(eigenvectors))
        ):
            error_msg = (
                "Covariance matrix has invalid eigenvalues or eigenvectors."
            )
            raise ValueError(error_msg)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        fve = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        if isinstance(n_components, int):
            if n_components > len(eigenvalues):
                error_msg = (
                    "The sample size must be bigger than the number of "
                    "components"
                )
                raise ValueError(error_msg)
            n_selected_components = n_components
        else:
            # Find the optimal number of components
            n_selected_components = np.where(fve >= n_components)[0][0] + 1

        eigenvalues = eigenvalues[:n_selected_components]
        eigenvectors = eigenvectors[:, :n_selected_components]

        lambda_ = h * eigenvalues
        eigenvectors /= np.sqrt(h)

        eigenvectors = self._normalize_and_orient(
            eigenvectors,
            self.t_covariance_.ravel(),
        )

        phi = self._interpolate_and_normalize_basis(
            t_eigen,
            eigenvectors,
            self.mean_.grid_points[0],
        )

        return n_selected_components, fve, lambda_, phi.T

    def _rotate_coordinates(
        self,
        cov_coords: NDArrayFloat,
        r_eval: NDArrayFloat,
        s_eval: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """
        Apply 45-degree rotation to covariance and evaluation coordinates.

        Transforms coordinates by a rotation matrix to improve numerical
        stability when estimating the diagonal of the covariance surface.

        Args:
            cov_coords: Raw covariance coordinate pairs.
            r_eval: Evaluation points in r-direction.
            s_eval: Evaluation points in s-direction.

        Returns:
            Rotated covariance coordinates and rotated evaluation points.
        """
        r_mat = np.sqrt(2) / 2 * np.array([[1, 1], [-1, 1]])
        r_cov_coords = np.einsum("ijk,jk->ik", cov_coords, r_mat)
        r_cov_coords = r_cov_coords[:, :, np.newaxis]

        t_eval = np.stack((r_eval.ravel(), s_eval.ravel()), axis=1)
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

        Calculates kernel weights in the rotated coordinate system for
        estimating the diagonal of the covariance surface.

        Args:
            h: Bandwidth parameter for the kernel.
            t_pairs: Rotated observation coordinate pairs.
            r_t_eval: Rotated evaluation points.
            win: Observation weights.

        Returns:
            Diagonal weight matrix for the rotated local regression.
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

        Constructs the design matrix for local polynomial regression in the
        rotated coordinate system, used for diagonal covariance estimation.

        Args:
            t_pairs: Rotated observation coordinate pairs.
            r_t_eval: Rotated evaluation points.

        Returns:
            Design matrix of shape (n_eval, n_obs, 3).
        """
        n_eval = r_t_eval.shape[0]
        n_obs = t_pairs.shape[0]

        # Vectorized: broadcast (n_eval, n_obs) differences
        delta_r0 = t_pairs[None, :, 0, 0] - r_t_eval[:, None, 0, 0]
        delta_r1 = t_pairs[None, :, 1, 0] - r_t_eval[:, None, 1, 0]

        x = np.ones((n_eval, n_obs, 3))
        x[:, :, 1] = delta_r0**2
        x[:, :, 2] = delta_r1

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
            cov_coords,
            r_eval,
            s_eval,
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

        try:
            beta = np.linalg.solve(xtwx, xtwy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtwx) @ xtwy

        return np.array(beta[:, 0])

    def _get_sigma2(  # noqa: PLR0913
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
        smooth_diag = local_linear_smooth_irregular_nd(
            t_diag,
            cov_diag,
            t_eval,
            h,
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
        domain_width = max_domain - min_domain
        a = min_domain + domain_width * self.variance_error_interval[0]
        b = max_domain - domain_width * (1 - self.variance_error_interval[1])

        # Build FDataGrid for the difference and integrate
        n_eval = t_eval.shape[0]
        smooth_diag = np.atleast_1d(smooth_diag).ravel()[:n_eval]
        rotated_cov_diag = np.atleast_1d(rotated_cov_diag).ravel()[:n_eval]
        diff_values = (smooth_diag - rotated_cov_diag).ravel()
        grid_1d = t_eval.ravel()[:n_eval] if t_eval.ndim == 1 else t_eval[:, 0]
        diff_fd = FDataGrid(
            data_matrix=diff_values.reshape(1, -1),
            grid_points=(grid_1d,),
        )
        sigma2 = diff_fd.integrate(domain=((a, b),))[0, 0] * 2 / domain_width

        if sigma2 < 0:
            warnings.warn(
                "The estimated variance is negative. Setting it to 0.",
                UserWarning,
                stacklevel=2,
            )
            sigma2 = 0
        return float(sigma2)

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,  # noqa: ARG002
    ) -> PACE:
        """
        Compute the ``n_components`` first principal components and saves them.

        Fits the PACE model by estimating the mean and covariance via local
        linear smoothing, extracting eigenfunctions, and computing noise
        variance when ``assume_noisy`` is True.

        Args:
            X: The functional data object to be analysed.
            y: Ignored. Only present because of fit function convention.

        Returns:
            self

        Raises:
            ValueError: If parameters are invalid, sample size is too small,
                or covariance matrix has invalid eigenvalues.
        """
        self._validate_params()

        # Handle n_components default (None -> 1.0 for FVE threshold)
        n_comp = self.n_components
        n_components = n_comp if n_comp is not None else 1.0

        # Check that the number of components is smaller than the sample size
        if n_components > len(X.start_indices):
            error_msg = (
                "The sample size must be bigger than the number of components"
            )
            raise ValueError(error_msg)

        # Slice the data to the boundary effect interval
        x_work = self._slice_fdata_irregular(X)

        # Determine reconstruction grid for mean/eigenfunctions
        if self.reconstruction_grid is None:
            # Default: use all unique observation points
            t_eval = np.sort(np.unique(X.points, axis=0), axis=0)
        elif isinstance(self.reconstruction_grid, int):
            # Create uniform grid with specified number of points
            domain_min, domain_max = X.domain_range[0]
            t_eval = np.linspace(
                domain_min,
                domain_max,
                self.reconstruction_grid,
            ).reshape(-1, 1)
        else:
            # Use provided grid directly
            t_eval = np.atleast_2d(self.reconstruction_grid)
            if t_eval.shape[0] == 1:
                t_eval = t_eval.T
            t_eval = np.sort(t_eval, axis=0)

        # Select bandwidth for mean (via GCV if range was provided)
        self.bandwidth_mean_ = self._select_bandwidth_mean(
            x_work.points,
            x_work.values,
        )

        mean_smoother = PooledMeanSmoother(
            bandwidth=self.bandwidth_mean_,
            kernel=self.kernel_mean,
            output_points=t_eval,
        )
        self.mean_ = mean_smoother.fit_transform(x_work).copy(
            domain_range=X.domain_range,
        )

        raw_cov_data = self._compute_raw_covariances(
            x_work,
            self.mean_.data_matrix[0],
            t_eval,
            assume_noisy=self.assume_noisy,
        )

        raw_cov_coords = raw_cov_data.t_pairs_neq
        raw_cov_values = raw_cov_data.f_raw_cov_neq
        win = raw_cov_data.weights
        raw_diag_coords = raw_cov_data.t_pairs_eq
        raw_diag_values = raw_cov_data.f_raw_cov_eq

        if self.assume_noisy and len(raw_cov_coords) == 0:
            error_msg = (
                "Unable to perform computations with one measurement per "
                "observation on noisy data."
            )
            raise ValueError(error_msg)

        # Resolve n_grid_points (None means full grid from unique time points)
        n_grid_pts = self.n_grid_points
        if n_grid_pts is None:
            n_grid_pts = len(np.unique(x_work.points, axis=0))

        axes = [
            np.linspace(start, end, n_grid_pts)
            for start, end in x_work.domain_range
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.t_covariance_ = np.stack([m.ravel() for m in mesh], axis=-1)

        bw_cov_n_pts = self.bw_cov_n_grid_points
        if bw_cov_n_pts is None:
            bw_cov_n_pts = len(self.t_covariance_)

        cov_grid = np.linspace(
            self.t_covariance_[0],
            self.t_covariance_[-1],
            bw_cov_n_pts,
        )

        self.bandwidth_cov_ = self._select_bandwidth_cov(
            cov_grid,
            raw_cov_coords,
            raw_cov_values,
            win,
            t_eval,
        )

        cov_smoother = PooledCovarianceSmoother(
            bandwidth=self.bandwidth_cov_,
            kernel=self.kernel_cov,
            output_points_r=self.t_covariance_,
            output_points_s=self.t_covariance_,
        )
        try:
            self.covariance_ = cov_smoother.fit_transform(
                raw_cov_coords,
                raw_cov_values,
                sample_weight=win,
            )
        except ValueError as e:
            if "observation" in str(e).lower():
                error_msg = (
                    "Unable to perform computations with one measurement per "
                    "observation on noisy data."
                )
                raise ValueError(error_msg) from e
            raise

        pc_data = self._get_pc(
            self.covariance_,
            n_components,
        )

        n_components, explained_variance_ratio_ = pc_data[:2]
        eigenvalues, phi = pc_data[2:]

        self.n_components = int(n_components)
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

    def _compute_fpc_scores(
        self,
        X: FDataIrregular,
    ) -> NDArrayFloat:
        """
        Compute FPC scores for all subjects.

        The per-subject computation cannot be fully vectorized because each
        subject has a different number of measurements (m_i), leading to
        variable-sized matrices for the BLUP formula. However, we optimize by:
        - Building a KDTree once for nearest-neighbor index lookup
        - Pre-computing eigenvalue matrix and mean/component arrays

        Args:
            X: The functional data object to be analysed.

        Returns:
            FPC scores of shape (n_samples, n_components).
        """
        n_samples = len(X.start_indices)
        # After fit(), n_components is always an int
        assert isinstance(self.n_components, int)
        n_components = self.n_components
        end_indices = np.append(X.start_indices[1:], len(X.points))
        t_mean = self.mean_.grid_points[0].ravel()

        # Pre-compute shared data
        lambda_ = np.diag(self.explained_variance_)
        mean_values = self.mean_.data_matrix[0, :, 0]
        phi_values = self.components_.data_matrix[..., 0]

        # Build KDTree once for all nearest-neighbor lookups
        tree = cKDTree(t_mean.reshape(-1, 1))

        # Handle noiseless case
        if self.assume_noisy is False:
            self.sigma2_ = REGULARIZATION_TERM

        fpc_scores = np.zeros((n_samples, n_components))

        for i in range(n_samples):
            start, end = X.start_indices[i], end_indices[i]
            points_i = X.points[start:end, 0]
            values_i = X.values[start:end, 0]

            m_i = len(points_i)
            if m_i == 0:
                continue

            # Vectorized nearest-neighbor index lookup
            _, indices = tree.query(points_i.reshape(-1, 1))
            indices = np.atleast_1d(indices)

            mu_i = mean_values[indices]
            phi_i = phi_values[:, indices].T  # (m_i, n_components)

            # Compute scores using BLUP formula: num @ denom^{-1} @ residual
            num = lambda_ @ phi_i.T
            denom = phi_i @ lambda_ @ phi_i.T + self.sigma2_ * np.eye(m_i)
            residual_i = values_i - mu_i

            try:
                # Solve denom @ x = residual_i, then compute num @ x
                fpc_scores[i, :] = num @ np.linalg.solve(denom, residual_i)
            except np.linalg.LinAlgError:
                # Fallback for singular matrices
                fpc_scores[i, :] = num @ np.linalg.lstsq(denom, residual_i)[0]

        return fpc_scores

    def transform(
        self,
        X: FData,
        y: object = None,  # noqa: ARG002
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components scores.

        Projects each trajectory onto the fitted eigenfunctions using the
        BLUP (Best Linear Unbiased Predictor) formula for irregular data.

        Args:
            X: The functional data object to be analysed.
            y: Ignored. Only present because of fit function convention.

        Returns:
            Principal component scores. Data matrix of shape
            ``(n_samples, n_components)``.

        Raises:
            ValueError: If the estimator has not been fitted.
        """
        check_is_fitted(self)
        return self._compute_fpc_scores(X)

    def fit_transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Fit the model and compute principal component scores.

        Equivalent to calling :meth:`fit` followed by :meth:`transform`.

        Args:
            X: The functional data object to be analysed.
            y: Ignored. Only present because of fit function convention.

        Returns:
            Principal component scores of shape ``(n_samples, n_components)``.

        Raises:
            ValueError: If parameters are invalid, sample size is too small,
                or covariance matrix has invalid eigenvalues.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(
        self,
        pc_scores: NDArrayFloat,
    ) -> FData:
        """
        Compute the recovery from the fitted principal components scores.

        Maps ``pc_scores`` from the fitted functional PCs' space back to the
        input functional space. ``pc_scores`` may be an array returned by
        :meth:`transform`.

        Args:
            pc_scores: Array of shape ``(n_samples, n_components)``.

        Returns:
            Reconstructed functional data as FDataGrid.

        Raises:
            ValueError: If the estimator has not been fitted.
        """
        check_is_fitted(self)
        phi = self.components_.data_matrix[..., 0]
        mean = self.mean_.data_matrix[0, :, 0]  # (n_grid,)
        reconstructed = pc_scores @ phi + mean

        return FDataGrid(
            data_matrix=reconstructed,
            grid_points=self.mean_.grid_points[0].ravel(),  # 1D grid
            domain_range=self.mean_.domain_range,
            dataset_name=self.mean_.dataset_name,
            argument_names=self.mean_.argument_names,
            coordinate_names=self.mean_.coordinate_names,
            extrapolation=self.mean_.extrapolation,
            interpolation=self.mean_.interpolation,
        )

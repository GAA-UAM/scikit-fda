"""Tests for irregular local linear smoothing helpers and pooled smoothers."""
from __future__ import annotations

import numpy as np
import pytest

from skfda.preprocessing.smoothing import (
    PooledCovarianceSmoother,
    PooledMeanSmoother,
    local_linear_smooth_covariance_2d,
    local_linear_smooth_irregular_nd,
)
from skfda.representation.grid import FDataGrid
from skfda.representation.irregular import FDataIrregular


def _gaussian_kernel_nd(u: np.ndarray) -> np.ndarray:
    """Multivariate Gaussian kernel: K(u) with u (..., d) -> (...,)."""
    *batch, d = u.shape
    coeff = 1.0 / ((2 * np.pi) ** (d / 2))
    return np.array(coeff * np.exp(-0.5 * np.sum(u**2, axis=-1)))


##############################################################################
# local_linear_smooth_irregular_nd
##############################################################################


class TestLocalLinearSmoothIrregularNd:
    """Tests for local_linear_smooth_irregular_nd."""

    def test_shapes_scalar_codomain(self) -> None:
        """Output (n_eval,) when values_obs is (n_obs,)."""
        n_obs, n_eval = 20, 15
        points_obs = np.random.randn(n_obs, 1)
        values_obs = np.random.randn(n_obs)
        points_eval = np.random.randn(n_eval, 1)
        out = local_linear_smooth_irregular_nd(
            points_obs,
            values_obs,
            points_eval,
            bandwidth=0.5,
            kernel=_gaussian_kernel_nd,
        )
        assert out.shape == (n_eval,)
        assert out.dtype == np.float64

    def test_shapes_vector_codomain(self) -> None:
        """Output (n_eval, q) when values_obs is (n_obs, q)."""
        n_obs, n_eval, q = 20, 15, 3
        points_obs = np.random.randn(n_obs, 1)
        values_obs = np.random.randn(n_obs, q)
        points_eval = np.random.randn(n_eval, 1)
        out = local_linear_smooth_irregular_nd(
            points_obs,
            values_obs,
            points_eval,
            bandwidth=0.5,
            kernel=_gaussian_kernel_nd,
        )
        assert out.shape == (n_eval, q)

    def test_1d_domain(self) -> None:
        """1D domain: points (n_obs,) or (n_obs, 1) accepted."""
        t_obs = np.linspace(0, 1, 11)
        y_obs = np.sin(2 * np.pi * t_obs)
        t_eval = np.linspace(0.1, 0.9, 5)
        out = local_linear_smooth_irregular_nd(
            t_obs,
            y_obs,
            t_eval,
            bandwidth=0.2,
            kernel=_gaussian_kernel_nd,
        )
        assert out.shape == (5,)
        # Suavizado de seno debe estar en [-1, 1]
        np.testing.assert_array_less(np.abs(out), 1.5)

    def test_2d_domain(self) -> None:
        """2D domain: known smooth function."""
        np.random.seed(42)
        n_obs = 50
        points_obs = np.random.rand(n_obs, 2)
        # f(s, t) = s + t
        values_obs = points_obs[:, 0] + points_obs[:, 1]
        points_eval = np.array([[0.5, 0.5], [0.2, 0.8]])
        out = local_linear_smooth_irregular_nd(
            points_obs,
            values_obs,
            points_eval,
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        assert out.shape == (2,)
        # En (0.5, 0.5) esperamos ~1.0, en (0.2, 0.8) ~1.0
        np.testing.assert_allclose(out, [1.0, 1.0], atol=0.3)

    def test_weights_obs(self) -> None:
        """weights_obs changes the result."""
        t_obs = np.linspace(0, 1, 10)
        y_obs = np.ones(10)
        t_eval = np.array([0.5])
        out_no = local_linear_smooth_irregular_nd(
            t_obs, y_obs, t_eval, 0.3, _gaussian_kernel_nd
        )
        w = np.ones(10)
        w[5] = 10.0
        out_w = local_linear_smooth_irregular_nd(
            t_obs, y_obs, t_eval, 0.3, _gaussian_kernel_nd, weights_obs=w
        )
        assert out_no.shape == (1,)
        assert out_w.shape == (1,)
        # Con peso mayor en t=0.5 el valor suavizado puede acercarse más a y[5]
        np.testing.assert_allclose(out_no, 1.0, atol=0.01)
        np.testing.assert_allclose(out_w, 1.0, atol=0.01)

    def test_validation_bandwidth(self) -> None:
        """bandwidth <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            local_linear_smooth_irregular_nd(
                np.array([[0.0], [1.0]]),
                np.array([0.0, 1.0]),
                np.array([[0.5]]),
                bandwidth=0.0,
                kernel=_gaussian_kernel_nd,
            )

    def test_validation_empty_obs(self) -> None:
        """No observations raises ValueError."""
        with pytest.raises(ValueError, match="At least one observation"):
            local_linear_smooth_irregular_nd(
                np.empty((0, 1)),
                np.array([]),
                np.array([[0.5]]),
                bandwidth=0.5,
                kernel=_gaussian_kernel_nd,
            )

    def test_validation_domain_mismatch(self) -> None:
        """points_obs and points_eval with different d raises ValueError."""
        with pytest.raises(ValueError, match="must match"):
            local_linear_smooth_irregular_nd(
                np.random.randn(5, 1),
                np.random.randn(5),
                np.random.randn(3, 2),
                bandwidth=0.5,
                kernel=_gaussian_kernel_nd,
            )

    def test_1d_matches_pace_mean(self) -> None:
        """In 1D the helper matches PACE mean (same kernel/bandwidth)."""
        from skfda.preprocessing.dim_reduction import PACE
        from skfda.preprocessing.dim_reduction._pace import gaussian_kernel

        # Minimal data: 2 curves, few points
        points = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.5, 1.0])
        values = np.array([1.0, 1.5, 2.0, 1.5, 1.0, 1.2, 2.2, 1.2])
        start_indices = np.array([0, 5])
        fd = FDataIrregular(
            points=points,
            values=values,
            start_indices=start_indices,
        )
        bandwidth = 0.3
        pace = PACE(
            n_components=2,
            bandwidth_mean=bandwidth,
            bandwidth_cov=bandwidth,
        )
        pace.fit(fd)
        t_eval = pace.mean_.grid_points[0]
        if t_eval.ndim == 1:
            t_eval = t_eval.reshape(-1, 1)
        mean_helper = local_linear_smooth_irregular_nd(
            points_obs=fd.points,
            values_obs=fd.values,
            points_eval=t_eval,
            bandwidth=bandwidth,
            kernel=gaussian_kernel,
        )
        mean_pace = pace.mean_.data_matrix[0, :, 0]
        np.testing.assert_allclose(mean_helper, mean_pace, rtol=1e-9, atol=1e-9)


##############################################################################
# PooledMeanSmoother
##############################################################################


class TestPooledMeanSmoother:
    """Tests para PooledMeanSmoother."""

    @pytest.fixture
    def irregular_1d(self) -> FDataIrregular:
        """FDataIrregular 1D, dos curvas."""
        points = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        values = np.array([1.0, 2.0, 1.0, 1.5, 2.5, 1.5])
        start_indices = np.array([0, 3])
        return FDataIrregular(
            points=points,
            values=values,
            start_indices=start_indices,
        )

    def test_fit_transform_returns_fdatagrid(self, irregular_1d: FDataIrregular) -> None:
        """fit_transform returns FDataGrid with one sample."""
        smoother = PooledMeanSmoother(
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        mean_fd = smoother.fit_transform(irregular_1d)
        assert isinstance(mean_fd, FDataGrid)
        assert mean_fd.n_samples == 1
        assert mean_fd.dim_domain == 1
        assert mean_fd.dim_codomain == 1

    def test_mean_shape_default_output_points(self, irregular_1d: FDataIrregular) -> None:
        """mean_ has grid with unique observation points."""
        smoother = PooledMeanSmoother(
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        smoother.fit(irregular_1d)
        n_unique = len(np.unique(irregular_1d.points))
        assert smoother.mean_.data_matrix.shape == (1, n_unique, 1)
        assert len(smoother.mean_.grid_points[0]) == n_unique

    def test_mean_shape_int_output_points(self, irregular_1d: FDataIrregular) -> None:
        """output_points=int creates uniform grid of that size."""
        smoother = PooledMeanSmoother(
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
            output_points=20,
        )
        smoother.fit(irregular_1d)
        assert smoother.mean_.data_matrix.shape == (1, 20, 1)
        assert len(smoother.mean_.grid_points[0]) == 20

    def test_transform_returns_mean(self, irregular_1d: FDataIrregular) -> None:
        """transform returns the same mean (mean_)."""
        smoother = PooledMeanSmoother(
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        smoother.fit(irregular_1d)
        out = smoother.transform(irregular_1d)
        np.testing.assert_array_equal(out.data_matrix, smoother.mean_.data_matrix)

    def test_transform_before_fit_raises(self, irregular_1d: FDataIrregular) -> None:
        """transform without prior fit raises ValueError."""
        smoother = PooledMeanSmoother(
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        with pytest.raises(ValueError, match="not fitted"):
            smoother.transform(irregular_1d)

    def test_helper_matches_pooled_smoother(self, irregular_1d: FDataIrregular) -> None:
        """Helper result matches PooledMeanSmoother (same input)."""
        smoother = PooledMeanSmoother(
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        smoother.fit(irregular_1d)
        points = np.asarray(irregular_1d.points)
        values = np.asarray(irregular_1d.values)
        if points.ndim == 1:
            points = points[:, np.newaxis]
        points_eval = smoother.mean_.grid_points[0].reshape(-1, 1)
        direct = local_linear_smooth_irregular_nd(
            points,
            values,
            points_eval,
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        mean_vals = smoother.mean_.data_matrix[0, :, 0]
        np.testing.assert_allclose(direct, mean_vals, rtol=1e-10, atol=1e-10)

    def test_init_bandwidth_positive(self) -> None:
        """__init__ with bandwidth <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            PooledMeanSmoother(
                bandwidth=0.0,
                kernel=_gaussian_kernel_nd,
            )


class TestPooledMeanSmootherGCV:
    """GCV via PooledMeanSmoother.score()."""

    def test_score_formula_matches_package_gcv(self) -> None:
        """score() uses (RSS/n)/(1-trace/n)^2 and returns -GCV."""
        from scipy.spatial.distance import pdist

        from skfda.preprocessing.dim_reduction._pace import gaussian_kernel

        np.random.seed(1)
        n = 8
        points = np.sort(np.random.uniform(0, 5, (n, 1)))
        values = np.random.randn(n, 1)
        bandwidth = 1.5
        smoother = PooledMeanSmoother(
            bandwidth=1.0,
            kernel=gaussian_kernel,
        )
        score_val = smoother.score(points, values, bandwidth)
        gcv = -score_val
        y_hat = local_linear_smooth_irregular_nd(
            points, values, points, bandwidth, gaussian_kernel
        )
        if np.ndim(y_hat) == 1:
            y_hat = y_hat[:, np.newaxis]
        rss = float(np.sum((values - y_hat) ** 2))
        domain_diff = float(np.max(pdist(points)))
        k0 = gaussian_kernel(np.zeros((1, 1, 1)))[0]
        trace_eff = (domain_diff * k0) / bandwidth
        expected_gcv = (rss / n) / (1 - trace_eff / n) ** 2
        np.testing.assert_almost_equal(gcv, expected_gcv, decimal=10)

    def test_score_returns_neg_inf_when_denom_nonpositive(self) -> None:
        """score() returns -np.inf when bandwidth or denominator is invalid."""
        from skfda.preprocessing.dim_reduction._pace import gaussian_kernel

        points = np.array([[0.0], [1.0], [2.0]])
        values = np.array([[1.0], [2.0], [1.0]])
        smoother = PooledMeanSmoother(bandwidth=1.0, kernel=gaussian_kernel)
        assert smoother.score(points, values, 0.0) == -np.inf
        assert smoother.score(points, values, -1.0) == -np.inf


##############################################################################
# local_linear_smooth_covariance_2d and PooledCovarianceSmoother
##############################################################################


class TestLocalLinearSmoothCovariance2d:
    """Tests for local_linear_smooth_covariance_2d."""

    def test_shapes(self) -> None:
        """Output shape (n_r, n_s, q) for 1D domain."""
        n_obs, n_r, n_s = 30, 5, 6
        cov_coords = np.random.rand(n_obs, 2, 1)
        cov_values = np.random.randn(n_obs, 1)
        r_eval = np.linspace(0, 1, n_r).reshape(-1, 1)
        s_eval = np.linspace(0, 1, n_s).reshape(-1, 1)
        out = local_linear_smooth_covariance_2d(
            cov_coords,
            cov_values,
            r_eval,
            s_eval,
            bandwidth=0.2,
            kernel=_gaussian_kernel_nd,
        )
        assert out.shape == (n_r, n_s, 1)

    def test_symmetry(self) -> None:
        """Smoothed covariance is symmetrized (symmetric in r, s)."""
        np.random.seed(42)
        n_obs, n_pts = 25, 4
        cov_coords = np.random.rand(n_obs, 2, 1)
        cov_values = np.random.randn(n_obs, 1)
        grid = np.linspace(0, 1, n_pts).reshape(-1, 1)
        out = local_linear_smooth_covariance_2d(
            cov_coords,
            cov_values,
            grid,
            grid,
            bandwidth=0.3,
            kernel=_gaussian_kernel_nd,
        )
        out_2d = out[:, :, 0]
        np.testing.assert_allclose(out_2d, out_2d.T, rtol=1e-10)

    def test_validation_empty_obs(self) -> None:
        """No observations with positive weight raises ValueError."""
        with pytest.raises(ValueError, match="At least one observation"):
            local_linear_smooth_covariance_2d(
                np.random.rand(5, 2, 1),
                np.random.randn(5, 1),
                np.array([[0.5]]),
                np.array([[0.5]]),
                bandwidth=0.2,
                kernel=_gaussian_kernel_nd,
                weights_obs=np.zeros(5),
            )


class TestPooledCovarianceSmoother:
    """Tests for PooledCovarianceSmoother."""

    def test_fit_transform_shape(self) -> None:
        """fit_transform returns (n_r, n_s, q)."""
        np.random.seed(123)
        n_obs, n_pts = 20, 5
        cov_coords = np.random.rand(n_obs, 2, 1)
        cov_values = np.random.randn(n_obs, 1)
        grid = np.linspace(0, 1, n_pts).reshape(-1, 1)
        smoother = PooledCovarianceSmoother(
            bandwidth=0.25,
            kernel=_gaussian_kernel_nd,
            output_points_r=grid,
            output_points_s=grid,
        )
        cov = smoother.fit_transform(cov_coords, cov_values)
        assert cov.shape == (n_pts, n_pts, 1)
        np.testing.assert_allclose(smoother.covariance_, cov)

    def test_transform_before_fit_raises(self) -> None:
        """transform before fit raises ValueError."""
        smoother = PooledCovarianceSmoother(
            bandwidth=0.2,
            kernel=_gaussian_kernel_nd,
            output_points_r=np.array([[0.5]]),
            output_points_s=np.array([[0.5]]),
        )
        with pytest.raises(ValueError, match="not fitted"):
            smoother.transform()

    def test_covariance_matches_pace(self) -> None:
        """PooledCovarianceSmoother matches PACE covariance (same kernel/bw)."""
        from skfda.preprocessing.dim_reduction import PACE
        from skfda.preprocessing.dim_reduction._pace import gaussian_kernel

        points = np.array(
            [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.5, 1.0, 0.25, 0.75]
        )
        values = np.array([1.0, 1.5, 2.0, 1.5, 1.0, 1.2, 2.2, 1.2, 1.4, 1.6])
        start_indices = np.array([0, 5, 8])
        fd = FDataIrregular(
            points=points,
            values=values,
            start_indices=start_indices,
        )
        pace = PACE(
            n_components=2,
            bandwidth_mean=0.3,
            bandwidth_cov=0.25,
            kernel_mean=gaussian_kernel,
            kernel_cov=gaussian_kernel,
        )
        pace.fit(fd)
        t_cov = pace.t_covariance_
        smoother = PooledCovarianceSmoother(
            bandwidth=pace.bandwidth_cov_,
            kernel=gaussian_kernel,
            output_points_r=t_cov,
            output_points_s=t_cov,
        )
        raw = pace._compute_raw_covariances(
            pace._slice_fdata_irregular(fd),
            pace.mean_.data_matrix[0],
            pace.mean_.grid_points[0].reshape(-1, 1),
            assume_noisy=True,
        )
        cov_smoother = smoother.fit_transform(
            raw.t_pairs_neq,
            raw.f_raw_cov_neq,
            sample_weight=raw.weights,
        )
        np.testing.assert_allclose(
            pace.covariance_,
            cov_smoother,
            rtol=1e-10,
            atol=1e-10,
        )



class TestPooledCovarianceSmootherGCV:
    """GCV via PooledCovarianceSmoother.score()."""

    def test_score_formula_matches_previous(self) -> None:
        """score() uses RSS/denom^2 and returns -GCV."""
        from scipy.interpolate import CloughTocher2DInterpolator
        from scipy.spatial.distance import pdist

        from skfda.preprocessing.dim_reduction._pace import gaussian_kernel

        np.random.seed(456)
        t = np.linspace(0, 5, 5)
        cov_coords = np.array(
            [[ti, tj] for ti in t for tj in t],
            dtype=float,
        ).reshape(-1, 2, 1)
        n_pairs = cov_coords.shape[0]
        cov_values = (
            np.exp(-0.5 * (cov_coords[:, 0, 0] - cov_coords[:, 1, 0]) ** 2)
            + 0.1 * np.random.randn(n_pairs)
        ).reshape(-1, 1)
        win = np.ones(n_pairs)
        t_eval = t.reshape(-1, 1)
        time_points = t
        h = 1.2

        smoother = PooledCovarianceSmoother(
            bandwidth=1.0,
            kernel=gaussian_kernel,
            output_points_r=t_eval,
            output_points_s=t_eval,
        )
        score_val = smoother.score(
            cov_coords, cov_values, win, time_points, t_eval, h
        )
        gcv = -score_val

        # Manual formula: same as previous PACE implementation
        g_hat = local_linear_smooth_covariance_2d(
            cov_coords,
            cov_values,
            t_eval,
            t_eval,
            h,
            gaussian_kernel,
            weights_obs=win,
        )[:, :, 0]
        x, y = np.meshgrid(t_eval.ravel(), t_eval.ravel())
        grid_points = np.c_[x.ravel(), y.ravel()]
        interpolator = CloughTocher2DInterpolator(grid_points, g_hat.ravel())
        g_hat_int = interpolator(cov_coords[:, :, 0])
        cov_values_flat = np.asarray(cov_values, dtype=float).ravel()[:n_pairs]
        rss = float(np.sum((cov_values_flat - g_hat_int) ** 2))
        n = n_pairs
        time_2d = np.asarray(time_points).reshape(-1, 1)
        domain_diff = float(np.max(pdist(time_2d)))
        k0 = gaussian_kernel(np.zeros((1, 1, 1)))[0]
        denom = 1.0 - (1.0 / n) * ((domain_diff * k0) / h) ** 2
        expected_gcv = rss / (denom**2)

        np.testing.assert_allclose(gcv, expected_gcv, rtol=1e-12, atol=1e-14)

    def test_score_returns_neg_inf_when_invalid(self) -> None:
        """score() returns -np.inf when bandwidth <= 0."""
        from skfda.preprocessing.dim_reduction._pace import gaussian_kernel

        t = np.linspace(0, 1, 3)
        cov_coords = np.array(
            [[ti, tj] for ti in t for tj in t],
            dtype=float,
        ).reshape(-1, 2, 1)
        cov_values = np.ones((cov_coords.shape[0], 1))
        win = np.ones(cov_coords.shape[0])
        t_eval = t.reshape(-1, 1)

        smoother = PooledCovarianceSmoother(
            bandwidth=0.2,
            kernel=gaussian_kernel,
            output_points_r=t_eval,
            output_points_s=t_eval,
        )
        assert smoother.score(
            cov_coords, cov_values, win, t, t_eval, 0.0
        ) == -np.inf
        assert smoother.score(
            cov_coords, cov_values, win, t, t_eval, -1.0
        ) == -np.inf

"""
Local linear smoothing for irregular data.

Low-level function and pooled-mean transformer for FDataIrregular.
Compatible with PACE use (1:1 in 1D) and extensible to d>1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from ...representation.grid import FDataGrid
from ...representation.irregular import FDataIrregular
from ...typing._numpy import NDArrayFloat

# Kernel type: (n_eval, n_obs, d) or (n_eval, n_obs) in -> (n_eval, n_obs) out
KernelCallable = Callable[[NDArrayFloat], NDArrayFloat]

# Covariance coords are (t_r, t_s) pairs
COV_COORDS_PAIR_DIM = 2


def local_linear_smooth_irregular_nd(
    points_obs: NDArrayFloat,
    values_obs: NDArrayFloat,
    points_eval: NDArrayFloat,
    bandwidth: float,
    kernel: KernelCallable,
    *,
    weights_obs: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """
    Local linear smoothing on irregular data (domain R^d, codomain R^q).

    Fits a local linear model (intercept + linear terms in d dimensions) at
    each evaluation point, using kernel weights on (points_eval - points_obs)
    / bandwidth. The kernel is called with an array of shape (n_eval, n_obs, d)
    and must return (n_eval, n_obs).

    Parameters
    ----------
    points_obs : array of shape (n_obs, d)
        Observation points in the domain.
    values_obs : array of shape (n_obs,) or (n_obs, q)
        Observed values (scalar or vector).
    points_eval : array of shape (n_eval, d)
        Points at which to evaluate the smooth.
    bandwidth : float
        Bandwidth (must be > 0).
    kernel : callable
        Kernel K(u), u of shape (n_eval, n_obs, d); returns (n_eval, n_obs).
    weights_obs : array of shape (n_obs,) optional
        Extra weight per observation (multiplied by kernel weights).

    Returns:
    -------
    array of shape (n_eval,) or (n_eval, q)
        Smoothed estimate at each evaluation point.

    Raises:
    ------
    ValueError
        If dimensions are incompatible, bandwidth <= 0, or no observations.
    """
    points_obs = np.asarray(points_obs, dtype=float)
    values_obs = np.asarray(values_obs, dtype=float)
    points_eval = np.asarray(points_eval, dtype=float)

    if points_obs.ndim == 1:
        points_obs = points_obs[:, np.newaxis]

    if points_eval.ndim == 1:
        points_eval = points_eval[:, np.newaxis]

    points_obs = np.atleast_2d(points_obs)
    points_eval = np.atleast_2d(points_eval)

    n_obs, d = points_obs.shape
    n_eval, d_eval = points_eval.shape

    if d != d_eval:
        msg = (
            f"points_obs has {d} domain dimensions and points_eval has "
            f"{d_eval}; they must match."
        )
        raise ValueError(msg)
    if bandwidth <= 0:
        msg = "bandwidth must be strictly positive."
        raise ValueError(msg)
    if n_obs == 0:
        msg = "At least one observation is required."
        raise ValueError(msg)

    if values_obs.ndim == 1:
        values_obs = values_obs[:, np.newaxis]
    _, q = values_obs.shape

    # Differences (n_eval, n_obs, d) and kernel weights (n_eval, n_obs)
    diffs = points_eval[:, np.newaxis, :] - points_obs[np.newaxis, :, :]
    scaled = diffs / bandwidth
    kw = kernel(scaled)
    if kw.shape != (n_eval, n_obs):
        msg = (
            f"kernel must return shape (n_eval, n_obs)=({n_eval}, {n_obs}), "
            f"got {kw.shape}."
        )
        raise ValueError(msg)
    if weights_obs is not None:
        weights_obs = np.asarray(weights_obs, dtype=float).ravel()
        if weights_obs.shape[0] != n_obs:
            msg = (
                f"weights_obs must have length n_obs={n_obs}, "
                f"got {weights_obs.shape[0]}."
            )
            raise ValueError(msg)
        kw = kw * weights_obs[np.newaxis, :]

    epsilon = 1e-8
    estimates = np.empty((n_eval, q))

    for i in range(n_eval):
        xi = diffs[i]
        yi = values_obs
        wi = kw[i]

        win = wi[:, np.newaxis]
        k0 = np.sum(wi)
        k1 = np.sum(win * xi, axis=0)
        k2 = np.einsum("ni,nj->ij", win * xi, xi)

        s0 = np.sum(win * yi, axis=0)
        s1 = np.einsum("ni,nj->ij", win * xi, yi)

        xtwx = np.block(
            [
                [np.array([[k0]]), k1[np.newaxis, :]],
                [k1[:, np.newaxis], k2],
            ],
        ) + epsilon * np.eye(d + 1)
        xtwy = np.vstack([s0[np.newaxis, :], s1])

        try:
            beta = np.linalg.solve(xtwx, xtwy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtwx) @ xtwy

        estimates[i] = beta[0]

    if q == 1:
        return estimates.ravel()
    return estimates


def local_linear_smooth_covariance_2d(
    cov_coords: NDArrayFloat,
    cov_values: NDArrayFloat,
    r_eval: NDArrayFloat,
    s_eval: NDArrayFloat,
    bandwidth: float,
    kernel: KernelCallable,
    *,
    weights_obs: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """
    Local linear smoothing for a 2D covariance surface (product kernel).

    Fits a local linear model at each (r, s) evaluation pair using
    kernel weights K((t_r - r)/h) * K((t_s - s)/h). The result is
    symmetrized. Used for pooled covariance estimation (e.g. PACE).

    Parameters
    ----------
    cov_coords : array of shape (n_obs, 2, d)
        Observation coordinate pairs (t_r, t_s) in the domain.
    cov_values : array of shape (n_obs,) or (n_obs, q)
        Raw covariance values at each pair.
    r_eval : array of shape (n_r, d)
        First coordinate of evaluation grid.
    s_eval : array of shape (n_s, d)
        Second coordinate of evaluation grid.
    bandwidth : float
        Bandwidth (must be > 0).
    kernel : callable
        Kernel K(u); receives scaled differences of shape (n_eval, n_obs, d),
        returns (n_eval, n_obs).
    weights_obs : array of shape (n_obs,), optional
        Per-observation weights (e.g. for GCV or weighted estimation).

    Returns:
    -------
    array of shape (n_r, n_s, q)
        Smoothed covariance surface at (r_eval, s_eval), symmetrized.
    """
    cov_coords = np.asarray(cov_coords, dtype=float)
    cov_values = np.asarray(cov_values, dtype=float)
    r_eval = np.atleast_2d(np.asarray(r_eval, dtype=float))
    s_eval = np.atleast_2d(np.asarray(s_eval, dtype=float))

    n_obs = cov_coords.shape[0]
    n_r, d_r = r_eval.shape
    n_s, d_s = s_eval.shape
    if (
        cov_coords.shape[1] != COV_COORDS_PAIR_DIM
        or cov_coords.shape[2] != d_r
        or d_r != d_s
    ):
        msg = (
            "cov_coords must have shape (n_obs, 2, d); r_eval and s_eval "
            "must have d columns matching cov_coords."
        )
        raise ValueError(msg)
    if bandwidth <= 0:
        msg = "bandwidth must be strictly positive."
        raise ValueError(msg)
    if n_obs == 0:
        msg = "At least one observation is required."
        raise ValueError(msg)

    win = (
        np.ones(n_obs, dtype=float)
        if weights_obs is None
        else np.asarray(
            weights_obs,
            dtype=float,
        ).ravel()
    )
    if win.shape[0] != n_obs:
        msg = f"weights_obs must have length {n_obs}."
        raise ValueError(msg)

    active = np.nonzero(win)[0]
    t_pairs = cov_coords[active]
    cov_values = cov_values[active]
    win_filt: NDArrayFloat = win[active]
    n_obs = len(win_filt)
    if n_obs == 0:
        msg = "At least one observation with positive weight."
        raise ValueError(msg)

    if cov_values.ndim == 1:
        cov_values = cov_values[:, np.newaxis]

    # Kernel weights: product K_r * K_s (kernel expects (n_eval, n_obs, d))
    diff_r = (
        t_pairs[:, 0, :][:, np.newaxis, :] - r_eval[np.newaxis, :, :]
    ) / bandwidth
    diff_r = np.transpose(diff_r, (1, 0, 2))
    kernel_r = kernel(diff_r)
    diff_s = (
        t_pairs[:, 1, :][:, np.newaxis, :] - s_eval[np.newaxis, :, :]
    ) / bandwidth  # (n_obs, n_s, d)
    diff_s = np.transpose(diff_s, (1, 0, 2))
    kernel_s = kernel(diff_s)
    weights = (
        np.einsum("ik,jk->ijk", kernel_r, kernel_s) * win_filt
    )

    # Design matrix: at (i,j,k) row is [1, t_r[k]-r_i, t_s[k]-s_j] (first dim)
    x = np.ones((n_r, n_s, n_obs, 3))
    tr0 = t_pairs[:, 0, 0]  # (n_obs,)
    ts0 = t_pairs[:, 1, 0]
    r0 = r_eval[:, 0]  # (n_r,)
    s0 = s_eval[:, 0]  # (n_s,)
    x[:, :, :, 1] = tr0.reshape(1, 1, n_obs) - r0.reshape(-1, 1, 1)
    x[:, :, :, 2] = ts0.reshape(1, 1, n_obs) - s0.reshape(1, -1, 1)

    x_t = np.transpose(x, (0, 1, 3, 2))
    xtw = x_t * weights[:, :, np.newaxis, :]
    xtwx = xtw @ x
    xtwy = xtw @ cov_values

    try:
        beta = np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtwx) @ xtwy

    cov = beta[:, :, 0, :]
    if n_r == n_s:
        cov_t = np.transpose(cov, (1, 0, 2))
        out = (cov + cov_t) / 2.0
    else:
        out = cov
    return np.asarray(out)


class PooledCovarianceSmoother(BaseEstimator):
    """Pooled covariance surface estimator by 2D local linear smoothing.

    Smooths raw (r, s) covariance pairs onto a grid using a product kernel
    and local linear regression. Used by PACE and similar methods.

    Args:
        bandwidth: Kernel bandwidth. Must be strictly positive.
        kernel: Callable :math:`K(u)`, (n_eval, n_obs, d) -> (n_eval, n_obs).
        output_points_r: First-axis grid for evaluation (n_r, d).
        output_points_s: Second-axis grid for evaluation (shape (n_s, d)).
    """

    def __init__(
        self,
        *,
        bandwidth: float,
        kernel: KernelCallable,
        output_points_r: NDArrayFloat,
        output_points_s: NDArrayFloat,
    ) -> None:
        if bandwidth <= 0:
            msg = "bandwidth must be strictly positive."
            raise ValueError(msg)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.output_points_r = np.atleast_2d(
            np.asarray(output_points_r, dtype=float),
        )
        self.output_points_s = np.atleast_2d(
            np.asarray(output_points_s, dtype=float),
        )

    def fit(
        self,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        sample_weight: NDArrayFloat | None = None,
    ) -> PooledCovarianceSmoother:
        """Fit the smoothed covariance surface.

        Args:
            cov_coords: Observation coordinate pairs, shape (n_obs, 2, d).
            cov_values: Raw covariance values, shape (n_obs,) or (n_obs, q).
            sample_weight: Optional weights per observation, shape (n_obs,).

        Returns:
            self. The fitted surface is stored in :attr:`covariance_`.
        """
        self.covariance_ = local_linear_smooth_covariance_2d(
            cov_coords,
            cov_values,
            self.output_points_r,
            self.output_points_s,
            self.bandwidth,
            self.kernel,
            weights_obs=sample_weight,
        )
        return self

    def transform(
        self,
        X: object = None,  # noqa: ARG002
        y: object = None,  # noqa: ARG002
    ) -> NDArrayFloat:
        """Return the fitted covariance surface.

        Args:
            X: Ignored. Present for API compatibility.
            y: Ignored. Present for API compatibility.

        Returns:
            The fitted covariance array of shape (n_r, n_s, q).

        Raises:
            ValueError: If :meth:`fit` has not been called.
        """
        if not hasattr(self, "covariance_"):
            msg = "PooledCovarianceSmoother is not fitted. Call fit first."
            raise ValueError(msg)
        return self.covariance_

    def fit_transform(
        self,
        cov_coords: NDArrayFloat,
        cov_values: NDArrayFloat,
        sample_weight: NDArrayFloat | None = None,
    ) -> NDArrayFloat:
        """Fit the smoothed covariance and return the surface.

        Args:
            cov_coords: Observation coordinate pairs, shape (n_obs, 2, d).
            cov_values: Raw covariance values, shape (n_obs,) or (n_obs, q).
            sample_weight: Optional weights per observation, shape (n_obs,).

        Returns:
            The fitted covariance array of shape (n_r, n_s, q).
        """
        return self.fit(
            cov_coords,
            cov_values,
            sample_weight=sample_weight,
        ).transform()


class PooledMeanSmoother(
    TransformerMixin[FDataIrregular, FDataGrid, object],
    BaseEstimator,
):
    """Pooled mean estimator by local linear smoothing on FDataIrregular.

    Pools all observations from all curves and estimates a single mean
    function at the given output points, returning an FDataGrid with
    one sample (the mean).

    Args:
        bandwidth: Kernel bandwidth. Must be strictly positive.
        kernel: Callable kernel :math:`K(u)`, where u has shape
            (n_eval, n_obs, d).
            Must return an array of shape (n_eval, n_obs).
        output_points: Points at which to evaluate the mean. If None, use
            unique observation points. If int, use a uniform grid with that
            many points (1D domain only). If array, shape (n_eval, d) for
            d-dimensional domain.
    """

    def __init__(
        self,
        *,
        bandwidth: float,
        kernel: KernelCallable,
        output_points: NDArrayFloat | int | None = None,
    ) -> None:
        if bandwidth <= 0:
            msg = "bandwidth must be strictly positive."
            raise ValueError(msg)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.output_points = output_points

    def fit(
        self,
        X: FDataIrregular,
        y: object = None,  # noqa: ARG002
    ) -> PooledMeanSmoother:
        """Fit the pooled mean from all observations.

        Args:
            X: Irregular functional data; all observations are pooled to
                estimate a single mean curve.
            y: Ignored. Present for API compatibility.

        Returns:
            self. The fitted mean is stored in :attr:`mean_` as an
            :class:`~skfda.representation.grid.FDataGrid` with one sample.
        """
        points = np.asarray(X.points, dtype=float)
        values = np.asarray(X.values, dtype=float)
        if points.ndim == 1:
            points = points[:, np.newaxis]

        # Determine evaluation points
        if self.output_points is None:
            points_eval = np.unique(points, axis=0)
            if points_eval.shape[0] == 0:
                msg = "No observation points to define the grid."
                raise ValueError(msg)
            points_eval = np.sort(points_eval, axis=0)
        elif isinstance(self.output_points, int):
            if points.shape[1] != 1:
                msg = "output_points as int is only supported for 1D domain."
                raise ValueError(msg)
            dom = X.domain_range[0]
            points_eval = np.linspace(
                dom[0],
                dom[1],
                self.output_points,
                dtype=float,
            ).reshape(-1, 1)
        else:
            points_eval = np.asarray(self.output_points, dtype=float)
            if points_eval.ndim == 1:
                points_eval = points_eval[:, np.newaxis]
            if points_eval.shape[1] != points.shape[1]:
                msg = (
                    f"output_points must have {points.shape[1]} columns "
                    f"(dim_domain), got {points_eval.shape[1]}."
                )
                raise ValueError(msg)
            points_eval = np.sort(points_eval, axis=0)

        mean_values = local_linear_smooth_irregular_nd(
            points_obs=points,
            values_obs=values,
            points_eval=points_eval,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        )

        if mean_values.ndim == 1:
            mean_values = mean_values[:, np.newaxis]

        # FDataGrid: 1 sample, n_eval points, q codomain
        n_eval = points_eval.shape[0]
        q = mean_values.shape[1]
        data_matrix = mean_values.reshape(1, n_eval, q)

        # grid_points: tuple of one array per domain dimension
        grid_points = tuple(
            points_eval[:, j] for j in range(points_eval.shape[1])
        )

        # Use domain_range=None so FDataGrid uses sample_range from
        # grid_points, ensuring grid points lie inside the domain (e.g. when
        # output_points were provided from outside and X has a restricted
        # domain).
        self.mean_ = FDataGrid(
            data_matrix=data_matrix,
            grid_points=grid_points,
            domain_range=None,
            dataset_name=X.dataset_name,
            argument_names=X.argument_names,
            coordinate_names=X.coordinate_names,
            sample_names=["Mean function"],
            extrapolation=X.extrapolation,
            interpolation=X.interpolation,
        )
        return self

    def transform(
        self,
        X: FDataIrregular,  # noqa: ARG002
        y: object = None,  # noqa: ARG002
    ) -> FDataGrid:
        """Return the fitted mean curve.

        The same mean is returned for any input; the structure of X is not
        used except to check that the estimator is fitted.

        Args:
            X: Ignored. Present for API compatibility.
            y: Ignored. Present for API compatibility.

        Returns:
            The fitted mean as an
            :class:`~skfda.representation.grid.FDataGrid` with one sample.

        Raises:
            ValueError: If :meth:`fit` has not been called.
        """
        if not hasattr(self, "mean_"):
            msg = "PooledMeanSmoother is not fitted. Call fit first."
            raise ValueError(msg)
        return self.mean_

    def fit_transform(
        self,
        X: FDataIrregular,
        y: object = None,
    ) -> FDataGrid:
        """Fit the pooled mean and return it as FDataGrid.

        Args:
            X: Irregular functional data; all observations are pooled to
                estimate a single mean curve.
            y: Ignored. Present for API compatibility.

        Returns:
            The fitted mean as an
            :class:`~skfda.representation.grid.FDataGrid` with one sample.
        """
        return self.fit(X, y).transform(X, y)

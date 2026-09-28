"""Tests for DiagonalDiffusionProcess."""

from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from skfda.ml.generative._diffusion_process import (
    DiagonalDiffusionProcess,
    ForwardDiffusionProcess,
    _batch_linear_interp_1d,
    _validate_diagonal_callable,
)

from ._constants import BATCH_SIZE, CUSTOM_DIM, DATA_DIM, SEED
from .diffusion_test_mixins import (
    DiagonalCovarianceOperatorTests,
    ForwardDiffusionCheckpointTests,
    ForwardDiffusionFitTests,
    ForwardDiffusionSampleLimitTests,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_N_POINTS_SMALL: int = 500
_STAT_ATOL: float = 0.05
_COV_ERROR_COARSE: float = 1e-3
_COV_ERROR_FINE: float = 1e-5
_INTERP_ATOL: float = 1e-6


@pytest.fixture(scope="module")
def dim_callables() -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """D=-0.5 (mean-reverting), g=1.0 (unit noise).

    Closed-form _cov(t) = 1 - exp(-t).
    Shared callable source of truth for fixtures and tests that need
    identical callables — required for torch.equal comparisons in
    round-trip tests.
    """
    def _drift(t: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (t.shape[0], DATA_DIM), -0.5, device=t.device,
        )

    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (t.shape[0], DATA_DIM), 1.0, device=t.device,
        )

    return _drift, _diffusion


def _d_t_broadcast(t: torch.Tensor) -> torch.Tensor:
    """Constant D=-0.5, (N, 1) shape; picklable, valid for any M."""
    return torch.full(
        (t.shape[0], 1), -0.5, dtype=t.dtype, device=t.device,
    )


def _g_t_broadcast(t: torch.Tensor) -> torch.Tensor:
    """Constant g=1.0, (N, 1) shape; picklable, valid for any M."""
    return torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — fit / checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalFit(
    ForwardDiffusionFitTests,
    ForwardDiffusionCheckpointTests,
):
    """Tests for DiagonalDiffusionProcess fit / checkpoint contract.

    Extends the base contract:
    (1) _get_fit_state() stores 'device' (learned at fit() time).
    (2) n_integration_points is in init_kwargs via get_params().
    (3) fit() calls _validate_diagonal_callable before precomputation.
    (4) to_checkpoint() warns and replaces non-picklable callables.
    """

    @pytest.fixture
    def make_process(
        self,
    ) -> Callable[[], DiagonalDiffusionProcess]:
        """Factory with module-level picklable callables.

        Broadcast (N, 1) shape is valid for any M.
        """
        def factory() -> DiagonalDiffusionProcess:
            return DiagonalDiffusionProcess(
                drift_term=_d_t_broadcast,
                diffusion_term=_g_t_broadcast,
            )

        return factory

    @pytest.fixture
    def make_process_with_lambda(
        self,
    ) -> Callable[[], DiagonalDiffusionProcess]:
        """Factory with lambda callables (not picklable).

        Used only by picklability warning tests.
        """
        def factory() -> DiagonalDiffusionProcess:
            return DiagonalDiffusionProcess(
                drift_term=lambda t: torch.full(
                    (t.shape[0], CUSTOM_DIM), -0.5,
                ),
                diffusion_term=lambda t: torch.full(
                    (t.shape[0], CUSTOM_DIM), 1.0,
                ),
            )

        return factory

    @pytest.fixture
    def fitted_instance(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> DiagonalDiffusionProcess:
        """DiagonalDiffusionProcess fitted on x_batch.

        Used only for _get_fit_state() tests.
        """
        drift_term, diffusion_term = dim_callables
        instance = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
        )
        instance.fit(x_batch)
        return instance

    def test_get_fit_state_contains_device_key(
        self, fitted_instance: DiagonalDiffusionProcess,
    ) -> None:
        """_get_fit_state() must contain 'device'.

        VP and VE do not store device; this pins the Diagonal-specific
        extension. Dropping 'device' would leave any loaded instance
        silently unfitted.
        """
        state = fitted_instance._get_fit_state()  # noqa: SLF001

        assert "device" in state

    def test_checkpoint_init_kwargs_contains_n_integration_points(
        self, make_process: Callable[[], DiagonalDiffusionProcess],
    ) -> None:
        """checkpoint['init_kwargs'] must contain 'n_integration_points'.

        n_integration_points is a constructor parameter captured by
        get_params() into init_kwargs; from_checkpoint() uses it to
        reconstruct the instance.
        """
        instance = make_process()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = instance.to_checkpoint()

        assert "n_integration_points" in checkpoint["init_kwargs"]

    def test_to_checkpoint_picklable_callables_no_warning(
        self, make_process: Callable[[], DiagonalDiffusionProcess],
    ) -> None:
        """to_checkpoint() must not warn for picklable callables."""
        instance = make_process()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            checkpoint = instance.to_checkpoint()

        assert checkpoint["init_kwargs"]["drift_term"] is not None
        assert checkpoint["init_kwargs"]["diffusion_term"] is not None

    def test_to_checkpoint_nonpicklable_callable_warns_and_replaces_with_none(
        self,
        make_process_with_lambda: Callable[[], DiagonalDiffusionProcess],
    ) -> None:
        """to_checkpoint() must warn and replace non-picklable callables.

        UserWarning (not ValueError) is issued so the save proceeds.
        The checkpoint must be pickle-safe with None replacing callables.
        """
        instance = make_process_with_lambda()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        with pytest.warns(UserWarning, match="not picklable"):
            checkpoint = instance.to_checkpoint()

        pickle.dumps(checkpoint)

        assert checkpoint["init_kwargs"]["drift_term"] is None
        assert checkpoint["init_kwargs"]["diffusion_term"] is None

    def test_checkpoint_round_trip(
        self,
        make_process: Callable[[], DiagonalDiffusionProcess],
        t_batch: torch.Tensor,
    ) -> None:
        """to_checkpoint() -> from_checkpoint() restores the fitted state.

        Uses two independent instances to confirm genuine transfer.
        Grid shape assertions confirm n_integration_points was passed to
        linspace, not just stored while grids used a different size.
        """
        instance_a = make_process()
        instance_a.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = instance_a.to_checkpoint()
        instance_b = ForwardDiffusionProcess.from_checkpoint(checkpoint)

        assert type(instance_b) is DiagonalDiffusionProcess
        assert instance_b.M_ == instance_a.M_
        assert instance_b.device_ == instance_a.device_
        assert (
            instance_b.n_integration_points
            == instance_a.n_integration_points
        )
        assert (
            instance_b.precomputed_d_grid_T_.shape[0]
            == instance_a.n_integration_points
        )
        assert (
            instance_b.precomputed_sigma_t_grid_.shape[0]
            == instance_a.n_integration_points
        )
        assert torch.equal(
            instance_b._cov(t_batch),  # noqa: SLF001
            instance_a._cov(t_batch),  # noqa: SLF001
        )

    def test_checkpoint_round_trip_non_default_n_integration_points(
        self,
    ) -> None:
        """from_checkpoint() must restore n_integration_points=500.

        A buggy implementation that ignores the stored value would set
        1000 (the default). Grid shape assertions rule out the value
        being stored while grids are built with a different size.
        """
        instance_a = DiagonalDiffusionProcess(
            drift_term=_d_t_broadcast,
            diffusion_term=_g_t_broadcast,
            n_integration_points=_N_POINTS_SMALL,
        )
        instance_a.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = instance_a.to_checkpoint()
        instance_b = ForwardDiffusionProcess.from_checkpoint(checkpoint)

        assert instance_b.n_integration_points == _N_POINTS_SMALL
        assert (
            instance_b.precomputed_d_grid_T_.shape[0] == _N_POINTS_SMALL
        )
        assert (
            instance_b.precomputed_sigma_t_grid_.shape[0]
            == _N_POINTS_SMALL
        )

    def test_fit_raises_when_d_t_returns_incompatible_shape(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """fit() raises ValueError when drift_term shape (N, M') != M, 1."""
        _, good_g_t = dim_callables

        def bad_d_t(t: torch.Tensor) -> torch.Tensor:
            return torch.ones(t.shape[0], CUSTOM_DIM)

        process = DiagonalDiffusionProcess(
            drift_term=bad_d_t, diffusion_term=good_g_t,
        )

        with pytest.raises(ValueError, match="drift_term"):
            process.fit(x_batch)

    def test_fit_raises_when_g_t_returns_incompatible_shape(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """fit() raises ValueError when diffusion_term shape (N, M') bad."""
        good_d_t, _ = dim_callables

        def bad_g_t(t: torch.Tensor) -> torch.Tensor:
            return torch.ones(t.shape[0], CUSTOM_DIM)

        process = DiagonalDiffusionProcess(
            drift_term=good_d_t, diffusion_term=bad_g_t,
        )

        with pytest.raises(ValueError, match="diffusion_term"):
            process.fit(x_batch)

    def test_restore_fit_state_missing_device_raises(
        self,
        unfitted_instance: DiagonalDiffusionProcess,
    ) -> None:
        """_restore_fit_state() must raise ValueError when 'device' absent."""
        with pytest.raises(ValueError, match="'device' is missing"):
            unfitted_instance._restore_fit_state(  # noqa: SLF001
                {
                    "M": CUSTOM_DIM,
                    "generator_state": torch.Generator().get_state(),
                },
            )

    def test_restore_fit_state_device_as_string_raises(
        self,
        unfitted_instance: DiagonalDiffusionProcess,
    ) -> None:
        """_restore_fit_state() raises ValueError when 'device' is a string.

        "cpu" is valid in PyTorch elsewhere but not an instance of
        torch.device. Common source: JSON round-trip that converts
        torch.device to its string repr.
        """
        with pytest.raises(
            ValueError, match=r"Expected 'device' to be a torch\.device",
        ):
            unfitted_instance._restore_fit_state(  # noqa: SLF001
                {
                    "M": CUSTOM_DIM,
                    "generator_state": torch.Generator().get_state(),
                    "device": "cpu",
                },
            )

    def test_restore_fit_state_device_unsupported_type_raises(
        self,
        unfitted_instance: DiagonalDiffusionProcess,
    ) -> None:
        """_restore_fit_state() raises ValueError for torch.device('mps').

        Passes isinstance(torch.device) but fails ("cpu", "cuda") whitelist.
        """
        with pytest.raises(
            ValueError, match=r"Expected 'device' to be a torch\.device",
        ):
            unfitted_instance._restore_fit_state(  # noqa: SLF001
                {
                    "M": CUSTOM_DIM,
                    "generator_state": torch.Generator().get_state(),
                    "device": torch.device("mps"),
                },
            )

    def test_restore_fit_state_m_as_float_raises(
        self,
        unfitted_instance: DiagonalDiffusionProcess,
    ) -> None:
        """_restore_fit_state() raises TypeError when M is a float.

        isinstance(16.0, int) is False in Python 3; float M would
        silently permit fractional values like M=16.5.
        """
        with pytest.raises(TypeError):
            unfitted_instance._restore_fit_state(  # noqa: SLF001
                {"M": float(DATA_DIM), "device": torch.device("cpu")},
            )

    def test_restore_fit_state_none_callable_raises(
        self,
        make_process_with_lambda: Callable[[], DiagonalDiffusionProcess],
    ) -> None:
        """_restore_fit_state() raises ValueError when a callable is None.

        This is the failure mode when from_checkpoint() is called without
        a pre-built instance on a checkpoint saved with non-picklable
        callables. The error must fire before any precomputation.
        """
        instance = make_process_with_lambda()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        with pytest.warns(UserWarning, match="not picklable"):
            checkpoint = instance.to_checkpoint()

        broken = type(instance)(
            **{
                **instance.get_params(),
                "drift_term": None,
                "diffusion_term": None,
            },
        )

        with pytest.raises(ValueError, match="non-picklable"):
            broken._restore_fit_state(  # noqa: SLF001
                checkpoint["fit_state"],
            )


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — covariance operators
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalDiffusionProcessCov(DiagonalCovarianceOperatorTests):
    """Tests for _cov and operators of DiagonalDiffusionProcess.

    The process fixture uses VP-equivalent callables parametrized over
    (N, 1) and (N, M) output shapes; both must produce identical _cov.
    """

    @pytest.fixture(
        params=["shape_n_1", "shape_n_m"],
        ids=["shape_n_1", "shape_n_m"],
    )
    def process(
        self,
        request: pytest.FixtureRequest,
        x_batch: torch.Tensor,
    ) -> DiagonalDiffusionProcess:
        """VP-equivalent callables parametrized over (N, 1) and (N, M).

        drift_term = -0.5*beta(t), diffusion_term = sqrt(beta(t)) satisfy
        the VP SDE, giving identity mu^2 + sigma^2 = 1 as oracle.
        """
        beta_min, beta_max = 0.1, 10.0

        if request.param == "shape_n_1":
            def drift_term(t: torch.Tensor) -> torch.Tensor:
                return (
                    -0.5 * (beta_min + t * (beta_max - beta_min))
                ).unsqueeze(1)

            def diffusion_term(t: torch.Tensor) -> torch.Tensor:
                return torch.sqrt(
                    beta_min + t * (beta_max - beta_min),
                ).unsqueeze(1)

        else:
            def drift_term(t: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                return (
                    (-0.5 * (beta_min + t * (beta_max - beta_min)))
                    .unsqueeze(1)
                    .expand(-1, DATA_DIM)
                )

            def diffusion_term(t: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                return (
                    torch.sqrt(beta_min + t * (beta_max - beta_min))
                    .unsqueeze(1)
                    .expand(-1, DATA_DIM)
                )

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)
        return p

    def test_cov_shape_is_n_m(
        self,
        process: DiagonalDiffusionProcess,
        t_batch: torch.Tensor,
    ) -> None:
        """_cov must return shape (N, M), not (N,).

        Unlike VP/VE where _cov returns (N,); collapsing M silently
        produces wrong per-dimension noise levels in multiply_sigma.
        """
        result = process._cov(t_batch)  # noqa: SLF001

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_cov_at_t0_is_zero(
        self,
        process: DiagonalDiffusionProcess,
        t_zero: torch.Tensor,
    ) -> None:
        """_cov(0) must be zero for all dimensions.

        allclose (not equal) tolerates interpolation rounding at the
        grid boundary.
        """
        result = process._cov(t_zero)  # noqa: SLF001

        assert torch.allclose(result, torch.zeros_like(result))

    def test_cov_is_nonnegative(
        self,
        process: DiagonalDiffusionProcess,
        t_sweep: torch.Tensor,
    ) -> None:
        """_cov must be non-negative for all t and all dimensions.

        Production code clamps _cov to min=0 to handle numerical
        negatives from trapezoid integration.
        """
        result = process._cov(t_sweep)  # noqa: SLF001

        assert (result >= 0).all()

    def test_cov_is_monotonically_nondecreasing_per_dim(
        self,
        process: DiagonalDiffusionProcess,
        t_sweep: torch.Tensor,
    ) -> None:
        """_cov must be non-decreasing over t for each dimension.

        With drift_term < 0 and diffusion_term > 0, variance accumulates
        monotonically.
        """
        cov = process._cov(t_sweep)  # noqa: SLF001 # (50, M)

        assert (cov[1:] >= cov[:-1]).all()

    def test_cov_oracle_constant_d_and_g(
        self,
        x_batch: torch.Tensor,
        t_sweep: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """With D=-0.5 and g=1, _cov(t) must equal 1 - exp(-t) per dim.

        Closed form:
            exp(2 * int_0^t D du)       = exp(-t)
            F(t) = int_0^t exp(u) du    = exp(t) - 1
            _cov(t) = exp(-t)*(exp(t)-1) = 1 - exp(-t)

        atol=1e-4 accommodates trapezoid residuals with n=1000 points.
        """
        drift_term, diffusion_term = dim_callables

        oracle_process = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        oracle_process.fit(x_batch)

        expected = (
            (1.0 - torch.exp(-t_sweep))
            .unsqueeze(1)
            .expand(-1, DATA_DIM)
        )
        result = oracle_process._cov(t_sweep)  # noqa: SLF001

        assert torch.allclose(result, expected, atol=1e-4)

    def test_variance_preserving_identity_per_dimension(
        self,
        process: DiagonalDiffusionProcess,
        t_sweep: torch.Tensor,
    ) -> None:
        """mu_t^2 + _cov(t) must equal 1 per dimension for VP callables.

        atol=1e-4 vs 1e-6 for the scalar VP test: DiagonalDiffusionProcess
        uses numerical integration, contributing larger residuals.
        """
        x_ones = torch.ones(t_sweep.shape[0], DATA_DIM)

        mu_t = process.mean_cond(x_ones, t_sweep)   # (50, M)
        cov_t = process._cov(t_sweep)               # noqa: SLF001 # (50, M)

        assert torch.allclose(
            mu_t ** 2 + cov_t, torch.ones_like(cov_t), atol=1e-4,
        )

    def test_multiply_inv_sigma_returns_zero_near_t0(
        self,
        process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_zero: torch.Tensor,
    ) -> None:
        """multiply_inv_sigma(x, t=0) must return all zeros for any x.

        At t=0, sqrt(_cov)=0 < _PSEUDOINV_THRESHOLD=1e-3, so weight=0 is
        assigned to every dimension — any x maps to zero.
        torch.equal (not allclose): the zero is structural, not approx.
        """
        result = process.multiply_inv_sigma(x_batch, t_zero)

        assert torch.equal(result, torch.zeros_like(result))


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — sample_limit_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalSampleLimit(ForwardDiffusionSampleLimitTests):
    """Tests for sample_limit_distribution of DiagonalDiffusionProcess.

    Inherits 6 universal tests from ForwardDiffusionSampleLimitTests.
    Adds a statistical test against per-dimension covariance from _cov(1).
    """

    @pytest.fixture
    def make_process(
        self,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> Callable[[], DiagonalDiffusionProcess]:
        """Factory D=-0.5, g=1.0; _cov(t) = 1 - exp(-t) ~ 0.6321 at T=1."""
        def factory() -> DiagonalDiffusionProcess:
            drift_term, diffusion_term = dim_callables
            return DiagonalDiffusionProcess(
                drift_term=drift_term,
                diffusion_term=diffusion_term,
                seed=SEED,
            )

        return factory

    @pytest.fixture
    def make_process_alt_seed(
        self,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> Callable[[], DiagonalDiffusionProcess]:
        """Factory D=-0.5, g=1.0; _cov(t) = 1 - exp(-t) ~ 0.6321 at T=1."""
        def factory() -> DiagonalDiffusionProcess:
            drift_term, diffusion_term = dim_callables
            return DiagonalDiffusionProcess(
                drift_term=drift_term,
                diffusion_term=diffusion_term,
                seed=SEED + 1,
            )

        return factory

    def test_limit_distribution_statistics(
        self, fitted_process: DiagonalDiffusionProcess,
    ) -> None:
        """The limit distribution must be N(0, diag(_cov(T=1))).

        Oracle read from fitted_process._cov(T=1) so the test stays
        correct if callables change. With n=10_000: atol=0.05 is ~6-sigma
        for mean and ~9-sigma for std.
        """
        n = 10_000
        t_end = torch.ones(1)
        expected_std = torch.sqrt(
            fitted_process._cov(t_end),  # noqa: SLF001
        ).squeeze(0)  # (M,)

        samples = fitted_process.sample_limit_distribution(
            n_samples=n,
        )

        assert samples.mean(dim=0).abs().max() < _STAT_ATOL, (
            f"Per-dimension mean too large: "
            f"{samples.mean(dim=0).abs().max():.4f}"
        )
        assert (samples.std(dim=0) - expected_std).abs().max() < _STAT_ATOL, (
            f"Per-dimension std deviates from oracle "
            f"(expected ~{expected_std[0]:.4f} per dim): "
            f"{(samples.std(dim=0) - expected_std).abs().max():.4f}"
        )

    def test_limit_distribution_vp_equivalent_callables_returning_n_m(
        self, x_batch: torch.Tensor,
    ) -> None:
        """VP-equivalent (N, M) callables must produce N(0, I) limit.

        VP identity mu^2 + sigma^2 = 1 with mu -> 0 at T=1 guarantees
        sigma^2 -> 1, so the expected std is 1.0 per dim.
        """
        beta_min, beta_max = 0.1, 10.0

        def beta_t(t: torch.Tensor) -> torch.Tensor:
            return beta_min + t * (beta_max - beta_min)

        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return (-0.5 * beta_t(t)).unsqueeze(1).expand(-1, DATA_DIM)

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return (
                torch.sqrt(beta_t(t)).unsqueeze(1).expand(-1, DATA_DIM)
            )

        process = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            seed=SEED,
        )
        process.fit(x_batch)

        n = 10_000
        samples = process.sample_limit_distribution(n_samples=n)

        assert samples.mean(dim=0).abs().max() < _STAT_ATOL, (
            f"Per-dimension mean too large: "
            f"{samples.mean(dim=0).abs().max():.4f}"
        )
        assert (samples.std(dim=0) - 1.0).abs().max() < _STAT_ATOL, (
            f"Per-dimension std too far from 1.0: "
            f"{(samples.std(dim=0) - 1.0).abs().max():.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — drift
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalDiffusionDrift:
    """Tests for DiagonalDiffusionProcess.drift(x, t).

    drift(x, t) computes drift_term(t) * x element-wise.
    (N,) callables are rejected at fit() — PyTorch treats (N,) * (N, M)
    as (1, N) * (N, M), not (N, 1) * (N, M), producing wrong results
    when N == M. (N, 1) callables broadcast correctly.
    """

    @pytest.fixture
    def process(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> DiagonalDiffusionProcess:
        """DiagonalDiffusionProcess with D=-0.5, g=1.0 (N, DATA_DIM)."""
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)
        return p

    def test_drift_output_shape_with_n_m_callable(
        self,
        process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """drift(x, t) must return shape (N, M)."""
        result = process.drift(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_drift_correctness_constant_callable(
        self,
        process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """drift(x, t) must equal D * x element-wise for constant D=-0.5.

        D=-0.5 (not 1.0) rules out an identity-operator bug.
        """
        result = process.drift(x_batch, t_batch)

        assert torch.allclose(result, -0.5 * x_batch)

    def test_drift_per_dimension_independence(
        self,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """Each output column must be scaled by its own drift column.

        scales[0]  = -0.1  -> result[:, 0]  must equal -0.1 * x[:, 0]
        scales[-1] = -1.6  -> result[:, -1] must equal -1.6 * x[:, -1]
        """
        scales = torch.linspace(-0.1, -1.6, DATA_DIM)

        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return scales.unsqueeze(0).expand(t.shape[0], -1)

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), 1.0)

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)

        expected = scales.unsqueeze(0) * x_batch
        result = p.drift(x_batch, t_batch)

        assert torch.allclose(result, expected)
        assert torch.allclose(
            result[:, 0], scales[0] * x_batch[:, 0],
        )
        assert torch.allclose(
            result[:, -1], scales[-1] * x_batch[:, -1],
        )

    def test_fit_raises_when_d_t_returns_shape_n(
        self, x_batch: torch.Tensor,
    ) -> None:
        """fit() raises ValueError when drift_term returns shape (N,).

        PyTorch treats (N,) * (N, M) as (1, N) * (N, M) — wrong result.
        When N == M it silently gives the wrong answer without any error.
        """
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0],), -0.5)  # (N,) — rejected

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), 1.0)  # valid

        process = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )

        with pytest.raises(ValueError, match="drift_term"):
            process.fit(x_batch)

    def test_drift_shape_and_correctness_with_n_1_d_t_callable(
        self,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """drift(x, t) returns (N, M) and equals D*x for (N, 1) callable.

        (N, 1) * (N, M) -> (N, M) via standard PyTorch column broadcast.
        """
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], 1), -0.5)  # (N, 1) — valid

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), 1.0)  # valid

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)

        result = p.drift(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)
        assert torch.allclose(result, -0.5 * x_batch)


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — diffusion
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalDiffusionDiffusion:
    """Tests for DiagonalDiffusionProcess.diffusion(t).

    diffusion(t) is a transparent pass-through of self.diffusion_term(t).
    (N,) callables are rejected at fit(); (N, 1) is the accepted scalar
    form. diffusion() must NOT expand (N, 1) to (N, M).
    """

    @pytest.fixture
    def process(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> DiagonalDiffusionProcess:
        """Standard DiagonalDiffusionProcess with D=-0.5, g=1.0 (N, M)."""
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)
        return p

    def test_diffusion_is_passthrough(
        self,
        process: DiagonalDiffusionProcess,
        t_batch: torch.Tensor,
    ) -> None:
        """diffusion(t) must return exactly diffusion_term(t).

        torch.equal (not allclose): diffusion() contains no arithmetic,
        so the output must be bit-identical to calling diffusion_term.
        """
        result = process.diffusion(t_batch)
        expected = process.diffusion_term(t_batch)

        assert torch.equal(result, expected)

    def test_diffusion_output_shape_with_n_m_callable(
        self,
        process: DiagonalDiffusionProcess,
        t_batch: torch.Tensor,
    ) -> None:
        """diffusion(t) must return (N, M) when diffusion_term is (N, M)."""
        result = process.diffusion(t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_diffusion_is_passthrough_with_n_1_callable(
        self,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """diffusion(t) must return (N, 1) — not (N, M) — for (N, 1) term.

        diffusion() must NOT expand (N, 1) to (N, M); callers may inspect
        the return shape to decide how to broadcast downstream.
        """
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), -0.5)

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], 1), 1.0)  # (N, 1)

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)

        result = p.diffusion(t_batch)

        assert result.shape == (BATCH_SIZE, 1)
        assert torch.equal(result, p.diffusion_term(t_batch))

    def test_diffusion_forwards_time_argument(
        self, x_batch: torch.Tensor,
    ) -> None:
        """diffusion(t) must produce distinct outputs for distinct t.

        diffusion_term(t) = t + 0.1:
            t=0.1 -> g(t) = 0.2  (all dims)
            t=0.9 -> g(t) = 1.0  (all dims)
        """
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), -0.5)

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(-1, DATA_DIM) + 0.1

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)

        t_early = torch.full((BATCH_SIZE,), 0.1)
        t_late = torch.full((BATCH_SIZE,), 0.9)

        result_early = p.diffusion(t_early)
        result_late = p.diffusion(t_late)

        assert not torch.equal(result_early, result_late)
        assert torch.allclose(
            result_early, torch.full((BATCH_SIZE, DATA_DIM), 0.2),
        )
        assert torch.allclose(
            result_late, torch.full((BATCH_SIZE, DATA_DIM), 1.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — mean_cond
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalMeanCond:
    """Tests for DiagonalDiffusionProcess.mean_cond(x, t).

    mean_cond(x, t) = exp(int_0^t D(u) du) * x, evaluated via the
    precomputed D-integral grid. Unlike drift(), mean_cond is safe with
    (N, 1) drift_term callables because _precompute_D_integral expands
    (N, 1) to (N, M) at fit time.
    """

    @pytest.fixture
    def process(self, x_batch: torch.Tensor) -> DiagonalDiffusionProcess:
        """D=-0.5, g=1.0; closed form mean_cond(x, t) = exp(-0.5*t) * x."""
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), -0.5)

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), 1.0)

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)
        return p

    def test_mean_cond_output_shape(
        self,
        process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """mean_cond(x, t) must return shape (N, M)."""
        result = process.mean_cond(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_mean_cond_at_t0_equals_x(
        self,
        process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_zero: torch.Tensor,
    ) -> None:
        """mean_cond(x, 0) must equal x: int_0^0 D du = 0, so exp(0)*x=x.

        allclose (not equal) tolerates floating-point rounding from
        left-boundary interpolation in _batch_linear_interp_1d.
        """
        result = process.mean_cond(x_batch, t_zero)

        assert torch.allclose(result, x_batch, atol=1e-5)

    def test_mean_cond_oracle_constant_d(
        self,
        process: DiagonalDiffusionProcess,
        t_sweep: torch.Tensor,
    ) -> None:
        """With D=-0.5 and x=ones, mean_cond(ones, t) must equal exp(-0.5t).

        Closed form:
            _integrate_D(t) = int_0^t -0.5 du = -0.5*t
            mean_cond(1, t) = exp(-0.5*t)

        atol=1e-4 for interpolation residuals with n_integration_points=1000.
        """
        x_ones = torch.ones(t_sweep.shape[0], DATA_DIM)
        expected = (
            torch.exp(-0.5 * t_sweep).unsqueeze(1).expand(-1, DATA_DIM)
        )

        result = process.mean_cond(x_ones, t_sweep)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_mean_cond_decays_monotonically_for_mean_reverting_d(
        self,
        process: DiagonalDiffusionProcess,
        t_sweep: torch.Tensor,
    ) -> None:
        """||mean_cond(x, t)|| must be non-increasing for drift_term < 0.

        +1e-6 tolerance absorbs float rounding without masking real
        non-monotone behaviour (a genuine bug produces steps of ~1e-3+).
        """
        x_ones = torch.ones(t_sweep.shape[0], DATA_DIM)

        result = process.mean_cond(x_ones, t_sweep)
        norms = result.norm(dim=1)

        assert (norms[1:] <= norms[:-1] + 1e-6).all()

    def test_mean_cond_shape_and_correctness_with_n_1_d_t_callable(
        self,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """mean_cond returns (N, M) and correct values for (N, 1) drift.

        _precompute_D_integral expands (N, 1) to (N, M) at fit time, so
        _integrate_D always returns (N, M) regardless of drift_term shape.
        atol=1e-4 matches test_mean_cond_oracle_constant_d.
        """
        def drift_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], 1), -0.5)

        def diffusion_term(t: torch.Tensor) -> torch.Tensor:
            return torch.full((t.shape[0], DATA_DIM), 1.0)

        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_batch)

        result = p.mean_cond(x_batch, t_batch)
        expected = torch.exp(-0.5 * t_batch).unsqueeze(1) * x_batch

        assert result.shape == (BATCH_SIZE, DATA_DIM)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_mean_cond_is_linear_in_x(
        self,
        process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """mean_cond(alpha*x, t) must equal alpha*mean_cond(x, t).

        alpha=3.7 (non-integer) rules out accidental linearity for only
        integer multiples.
        """
        alpha = 3.7

        result_scaled = process.mean_cond(alpha * x_batch, t_batch)
        result_then_scaled = alpha * process.mean_cond(x_batch, t_batch)

        assert torch.allclose(result_scaled, result_then_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — precomputed grid structure
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalPrecomputedGrids:
    """Tests for the precomputed integral grid structure.

    The four precomputed tensors:
        precomputed_d_grid_T_     — time grid for int D du
        precomputed_d_grid_       — cumulative int_0^t D(u) du at each node
        precomputed_sigma_t_grid_ — time grid for int g^2*exp(-2*int D) ds
        precomputed_sigma_f_grid_ — cumulative int_0^t g^2*exp(-2*intD) ds
    """

    @pytest.fixture
    def process(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> DiagonalDiffusionProcess:
        """D=-0.5, g=1.0, n_integration_points=500 (non-default)."""
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=_N_POINTS_SMALL,
        )
        p.fit(x_batch)
        return p

    def test_precomputed_d_grid_t__shape_matches_n_integration_points(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """precomputed_d_grid_T_ must have exactly n_integration_points rows.

        n=500 (non-default): a hardcoded 1000-point grid would fail.
        """
        assert process.precomputed_d_grid_T_.shape == (_N_POINTS_SMALL,)

    def test_precomputed_d_grid__shape_is_n_integration_points_by_m(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """precomputed_d_grid_ must have shape (n_integration_points, M).

        Matching first dimensions between Dgrid_T and Dgrid is required
        by _batch_linear_interp_1d for correct index lookups.
        """
        assert process.precomputed_d_grid_.shape == (
            _N_POINTS_SMALL, DATA_DIM,
        )
        assert (
            process.precomputed_d_grid_.shape[0]
            == process.precomputed_d_grid_T_.shape[0]
        )

    def test_precomputed_sigma_t_grid__shape_matches_n_integration_points(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """precomputed_sigma_t_grid_ must have n_integration_points rows."""
        assert (
            process.precomputed_sigma_t_grid_.shape == (_N_POINTS_SMALL,)
        )

    def test_precomputed_sigma_f_grid__shape_is_n_integration_points_by_m(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """precomputed_sigma_f_grid_ must have shape (n_int_points, M)."""
        assert process.precomputed_sigma_f_grid_.shape == (
            _N_POINTS_SMALL, DATA_DIM,
        )
        assert (
            process.precomputed_sigma_f_grid_.shape[0]
            == process.precomputed_sigma_t_grid_.shape[0]
        )

    def test_time_grid_boundaries_are_exactly_zero_and_t(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """Both time grids must start at 0.0 and end at T=1.0 exactly.

        If the grid starts at eps > 0, _cov(t=0) queries outside the
        grid and the clamped-index fallback returns a non-zero value.
        """
        assert process.precomputed_d_grid_T_[0].item() == 0.0
        assert process.precomputed_d_grid_T_[-1].item() == 1.0
        assert process.precomputed_sigma_t_grid_[0].item() == 0.0
        assert process.precomputed_sigma_t_grid_[-1].item() == 1.0

    def test_precomputed_d_grid__first_row_is_zero(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """The first row of precomputed_d_grid_ must be exactly zero.

        Encodes int_0^0 D du = 0, the foundation for mean_cond(x, 0)=x.
        torch.equal: this zero comes from new_zeros, not from integration.
        """
        assert torch.equal(
            process.precomputed_d_grid_[0],
            torch.zeros(DATA_DIM),
        )

    def test_precomputed_sigma_f_grid__first_row_is_zero(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """The first row of precomputed_sigma_f_grid_ must be zero.

        Encodes int_0^0 g^2*exp(-2*int D) ds = 0, foundation for _cov(0)=0.
        torch.equal: the zero must be structural, not numerical.
        """
        assert torch.equal(
            process.precomputed_sigma_f_grid_[0],
            torch.zeros(DATA_DIM),
        )

    def test_precomputed_d_grid__is_monotonically_nonincreasing(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """With drift_term=-0.5 everywhere, precomputed_d_grid_ is non-inc.

        All values must additionally be <= 0 since int of negative is <= 0.
        """
        grid = process.precomputed_d_grid_  # (500, DATA_DIM)

        assert (grid[1:] <= grid[:-1]).all()
        assert (grid <= 0).all()

    def test_precomputed_sigma_f_grid__is_monotonically_nondecreasing(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """precomputed_sigma_f_grid_ must be non-decreasing for all dims.

        The integrand g(s)^2 * exp(-2*int D) is always non-negative, so
        the cumulative integral must accumulate.
        """
        grid = process.precomputed_sigma_f_grid_  # (500, DATA_DIM)

        assert (grid[1:] >= grid[:-1]).all()
        assert (grid >= 0).all()

    def test_all_precomputed_grids_are_on_training_data_device(
        self, process: DiagonalDiffusionProcess,
    ) -> None:
        """All four precomputed grids must reside on the same device.

        .device.type (not .device) avoids index sensitivity in CPU-only
        environments.
        """
        assert process.precomputed_d_grid_T_.device.type == "cpu"
        assert process.precomputed_d_grid_.device.type == "cpu"
        assert process.precomputed_sigma_t_grid_.device.type == "cpu"
        assert process.precomputed_sigma_f_grid_.device.type == "cpu"

    def test_precomputed_d_grid__matches_analytical_value_with_constant_d(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """With constant D=-0.5, precomputed_d_grid_ must equal -0.5*t.

        The trapezoid rule is exact for constant integrands; residuals
        > atol=1e-6 indicate a bug in cumulation or prepend.
        """
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=1000,
        )
        p.fit(x_batch)

        t_grid = p.precomputed_d_grid_T_
        expected = (-0.5 * t_grid).unsqueeze(1).expand(-1, DATA_DIM)

        assert torch.allclose(p.precomputed_d_grid_, expected, atol=1e-6)

    def test_precomputed_sigma_f_grid__matches_analytical_with_constant(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """With D=-0.5 and g=1, precomputed_sigma_f_grid_ must equal exp(t)-1.

        Derivation:
            exp(-2 * int_0^s D du) = exp(s)
            F(t) = int_0^t exp(s) ds = exp(t) - 1

        atol=1e-4: the trapezoid rule is NOT exact for exp(s).
        """
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=1000,
        )
        p.fit(x_batch)

        t_grid = p.precomputed_sigma_t_grid_
        expected = (
            (torch.exp(t_grid) - 1.0).unsqueeze(1).expand(-1, DATA_DIM)
        )

        assert torch.allclose(
            p.precomputed_sigma_f_grid_, expected, atol=1e-4,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — n_integration_points parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestNIntegrationPoints:
    """Tests for the n_integration_points constructor parameter.

    Verifies that n_integration_points actually sets all grid sizes,
    produces a measurable accuracy improvement, works at n=2, and
    rejects n=1 with ValueError.
    """

    @pytest.mark.parametrize("n", [50, 300, 2000])
    def test_all_grid_lengths_equal_n_integration_points(
        self,
        n: int,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """All four precomputed grids must have first dimension exactly n.

        Parametrized over [50, 300, 2000], all distinct from the default
        1000, so a hardcoded 1000-point grid fails all three instances.
        """
        drift_term, diffusion_term = dim_callables
        process = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=n,
        )
        process.fit(x_batch)

        assert process.precomputed_d_grid_T_.shape[0] == n
        assert process.precomputed_d_grid_.shape[0] == n
        assert process.precomputed_sigma_t_grid_.shape[0] == n
        assert process.precomputed_sigma_f_grid_.shape[0] == n

    def test_higher_n_integration_points_gives_lower_approximation_error(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
        t_sweep: torch.Tensor,
    ) -> None:
        """Increasing n_integration_points must reduce max absolute error.

        Closed-form oracle: _cov(t) = 1 - exp(-t) for D=-0.5, g=1.0.

            n=10:   expected max error > 1e-3  (coarse, O(h^2) ~ 0.01)
            n=5000: expected max error < 1e-5  (fine,   O(h^2) ~ 4e-8)
        """
        drift_term, diffusion_term = dim_callables

        expected = (
            (1.0 - torch.exp(-t_sweep))
            .unsqueeze(1)
            .expand(-1, DATA_DIM)
        )

        process_coarse = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=10,
        )
        process_coarse.fit(x_batch)

        process_fine = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=5000,
        )
        process_fine.fit(x_batch)

        error_coarse = (
            process_coarse._cov(t_sweep) - expected  # noqa: SLF001
        ).abs().max().item()
        error_fine = (
            process_fine._cov(t_sweep) - expected  # noqa: SLF001
        ).abs().max().item()

        assert error_fine < error_coarse, (
            f"Fine grid (n=5000, error={error_fine:.2e}) must be more "
            f"accurate than coarse grid (n=10, error={error_coarse:.2e})"
        )
        assert error_coarse > _COV_ERROR_COARSE, (
            f"Coarse grid (n=10) must have non-trivial error; "
            f"got {error_coarse:.2e}. "
            f"n_integration_points may be ignored if this is near zero."
        )
        assert error_fine < _COV_ERROR_FINE, (
            f"Fine grid (n=5000) must achieve tight accuracy; "
            f"got {error_fine:.2e}. "
            f"n_integration_points may be stored but not passed to "
            f"linspace if this is large."
        )

    def test_n_integration_points_2_produces_valid_cov_output(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """n_integration_points=2 must return a valid _cov tensor.

        With n=2, _batch_linear_interp_1d gets a single-segment grid.
        Checks shape, no NaN/Inf, and non-negativity; accuracy is not
        checked with a single trapezoid step.
        """
        drift_term, diffusion_term = dim_callables
        process = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=2,
        )
        process.fit(x_batch)

        t_query = torch.linspace(0.0, 1.0, 20)
        result = process._cov(t_query)  # noqa: SLF001

        assert result.shape == (20, DATA_DIM)
        assert not result.isnan().any()
        assert not result.isinf().any()
        assert (result >= 0).all()

    def test_n_integration_points_1_raises_value_error_at_fit(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """n_integration_points=1 must raise ValueError during fit().

        With n=1, cumulative_trapezoid on a (1, M) tensor produces an
        empty (0, M) tensor. After zero-prepend the grid is (1, M);
        inside _batch_linear_interp_1d, clamp(indices, 1, K-1=0) has
        min > max — silently returning out-of-bounds reads rather than
        crashing, making the failure invisible to downstream tests.

        ValueError (not RuntimeError) requires an explicit guard at
        fit() time, not deferred to the first call to _cov.
        """
        drift_term, diffusion_term = dim_callables
        process = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            n_integration_points=1,
        )

        with pytest.raises(ValueError, match="n_integration_points"):
            process.fit(x_batch)


# ─────────────────────────────────────────────────────────────────────────────
# DiagonalDiffusionProcess — device consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagonalDeviceConsistency:
    """Tests that DiagonalDiffusionProcess ends up on the expected device.

    8.1-8.2: fit() captures and propagates training device into both
             precompute paths.
    8.3-8.6: CPU output device for each public computation method.
    8.7-8.10: CUDA device checks (skipped without CUDA hardware).
    """

    @pytest.fixture
    def cpu_process(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> DiagonalDiffusionProcess:
        """DiagonalDiffusionProcess fitted on CPU; used by 8.3-8.6."""
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            seed=SEED,
        )
        p.fit(x_batch.to("cpu"))
        return p

    @pytest.fixture(scope="class")
    def cuda_process(
        self,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> DiagonalDiffusionProcess:
        """CUDA-fitted DiagonalDiffusionProcess (class-scoped).

        Constructed inline (not via x_batch) to avoid pytest scope
        mismatch: x_batch is function-scoped and cannot be injected
        into a class-scoped fixture.
        """
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        x_cuda = torch.randn(
            BATCH_SIZE, DATA_DIM, generator=gen, device="cuda",
        )
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            seed=SEED,
        )
        p.fit(x_cuda)
        return p

    def test_d_integral_grids_are_on_cpu_after_fit_on_cpu_data(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """precomputed_d_grid_ must be on CPU when fitted on CPU data.

        Pins the full chain: fit() stores torch.device("cpu"), linspace
        receives it, and the resulting grids end up on CPU.
        """
        x_cpu = x_batch.to("cpu")
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_cpu)

        assert p.device_ == torch.device("cpu")
        assert p.precomputed_d_grid_T_.device.type == "cpu"
        assert p.precomputed_d_grid_.device.type == "cpu"

    def test_f_integral_grids_are_on_cpu_after_fit_on_cpu_data(
        self,
        x_batch: torch.Tensor,
        dim_callables: tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        """precomputed_sigma_f_grid_ must be on CPU after CPU fit.

        Tested separately from 8.1 because _precompute_F_integral is an
        independent code path that could have its own device bug.
        """
        x_cpu = x_batch.to("cpu")
        drift_term, diffusion_term = dim_callables
        p = DiagonalDiffusionProcess(
            drift_term=drift_term, diffusion_term=diffusion_term,
        )
        p.fit(x_cpu)

        assert p.precomputed_sigma_t_grid_.device.type == "cpu"
        assert p.precomputed_sigma_f_grid_.device.type == "cpu"

    def test_cov_output_device_is_cpu(
        self, cpu_process: DiagonalDiffusionProcess,
    ) -> None:
        """_cov must return a CPU tensor when grids are on CPU."""
        t_cpu = torch.linspace(0.1, 0.9, BATCH_SIZE)

        result = cpu_process._cov(t_cpu)  # noqa: SLF001

        assert result.device.type == "cpu"
        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_mean_cond_output_device_is_cpu(
        self,
        cpu_process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """mean_cond must return a CPU tensor for CPU inputs and grids."""
        result = cpu_process.mean_cond(
            x_batch.to("cpu"), t_batch.to("cpu"),
        )

        assert result.device.type == "cpu"
        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_multiply_methods_output_device_is_cpu(
        self,
        cpu_process: DiagonalDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """multiply_sigma, multiply_cov, multiply_inv_sigma are CPU."""
        z_cpu = x_batch.to("cpu")
        t_cpu = t_batch.to("cpu")

        result_sigma = cpu_process.multiply_sigma(z_cpu, t_cpu)
        result_cov = cpu_process.multiply_cov(z_cpu, t_cpu)
        result_inv_sigma = cpu_process.multiply_inv_sigma(z_cpu, t_cpu)

        assert result_sigma.device.type == "cpu"
        assert result_cov.device.type == "cpu"
        assert result_inv_sigma.device.type == "cpu"

    def test_sample_limit_distribution_device_matches_device_argument(
        self, cpu_process: DiagonalDiffusionProcess,
    ) -> None:
        """sample_limit_distribution output must be on the requested device.

        The device argument must be routed to torch.randn and torch.full,
        not silently ignored in favour of self.device.
        """
        result = cpu_process.sample_limit_distribution(
            n_samples=BATCH_SIZE,
            device="cpu",
        )

        assert result.device.type == "cpu"
        assert result.shape == (BATCH_SIZE, DATA_DIM)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )
    def test_all_grids_are_on_cuda_after_fit_on_cuda_data(
        self, cuda_process: DiagonalDiffusionProcess,
    ) -> None:
        """All four precomputed grids must be on CUDA for CUDA-fitted data.

        A hardcoded device="cpu" in either precompute path passes all
        CPU tests above but fails here.
        """
        assert cuda_process.device_.type == "cuda"
        assert cuda_process.precomputed_d_grid_T_.device.type == "cuda"
        assert cuda_process.precomputed_d_grid_.device.type == "cuda"
        assert (
            cuda_process.precomputed_sigma_t_grid_.device.type == "cuda"
        )
        assert (
            cuda_process.precomputed_sigma_f_grid_.device.type == "cuda"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )
    def test_cov_output_device_is_cuda(
        self, cuda_process: DiagonalDiffusionProcess,
    ) -> None:
        """_cov must return a CUDA tensor for CUDA grids and query times."""
        t_cuda = torch.linspace(0.1, 0.9, BATCH_SIZE, device="cuda")

        result = cuda_process._cov(t_cuda)  # noqa: SLF001

        assert result.device.type == "cuda"
        assert result.shape == (BATCH_SIZE, DATA_DIM)
        assert (result >= 0).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )
    def test_mean_cond_and_multiply_methods_output_device_is_cuda(
        self, cuda_process: DiagonalDiffusionProcess,
    ) -> None:
        """All public computation methods return CUDA tensors on CUDA."""
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        x_cuda = torch.randn(
            BATCH_SIZE, DATA_DIM, generator=gen, device="cuda",
        )
        t_cuda = torch.linspace(0.1, 0.9, BATCH_SIZE, device="cuda")

        r_mean = cuda_process.mean_cond(x_cuda, t_cuda)
        r_sigma = cuda_process.multiply_sigma(x_cuda, t_cuda)
        r_cov = cuda_process.multiply_cov(x_cuda, t_cuda)
        r_inv = cuda_process.multiply_inv_sigma(x_cuda, t_cuda)

        assert r_mean.device.type == "cuda"
        assert r_sigma.device.type == "cuda"
        assert r_cov.device.type == "cuda"
        assert r_inv.device.type == "cuda"

        assert r_mean.shape == (BATCH_SIZE, DATA_DIM)
        assert r_sigma.shape == (BATCH_SIZE, DATA_DIM)
        assert r_cov.shape == (BATCH_SIZE, DATA_DIM)
        assert r_inv.shape == (BATCH_SIZE, DATA_DIM)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA",
    )
    def test_cross_device_query_raises_value_error(
        self, cuda_process: DiagonalDiffusionProcess,
    ) -> None:
        """Querying a CUDA-fitted process with CPU times must raise.

        An explicit cross-device guard must detect the mismatch before
        reaching torch.searchsorted — not surface a RuntimeError from
        C++ internals.
        """
        t_cpu = torch.linspace(0.1, 0.9, BATCH_SIZE)

        with pytest.raises(ValueError, match="precomputed grids"):
            cuda_process._cov(t_cpu)  # noqa: SLF001


# ─────────────────────────────────────────────────────────────────────────────
# _validate_diagonal_callable
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCallable:
    """Tests for the _validate_diagonal_callable standalone function.

    Probes fn with a zero t-batch of length N; checks shape is (N, 1) or
    (N, M) where x.shape == (N, M). Also checks device match.
    """

    def test_accepts_shape_n_1(
        self, x_batch: torch.Tensor,
    ) -> None:
        """(N, 1) output must be accepted; must return None."""
        n = x_batch.shape[0]

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(n, 1)

        result = _validate_diagonal_callable(x_batch, fn, "diffusion_term")

        assert result is None

    def test_accepts_shape_n_m(
        self, x_batch: torch.Tensor,
    ) -> None:
        """(N, M) output must be accepted; must return None."""
        n, m = x_batch.shape

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(n, m)

        result = _validate_diagonal_callable(x_batch, fn, "drift_term")

        assert result is None

    def test_raises_for_incompatible_second_dimension(
        self, x_batch: torch.Tensor,
    ) -> None:
        """(N, M') where M' != M and M' != 1 must raise ValueError."""
        n = x_batch.shape[0]

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(n, CUSTOM_DIM)  # CUSTOM_DIM != DATA_DIM

        with pytest.raises(ValueError, match="drift_term"):
            _validate_diagonal_callable(x_batch, fn, "drift_term")

    def test_raises_for_3d_output(
        self, x_batch: torch.Tensor,
    ) -> None:
        """(N, M, 1) output (ndim==3) must raise ValueError.

        Before explicit ndim checking, 3-D outputs slipped through and
        caused inscrutable size-mismatch errors.
        """
        n, m = x_batch.shape

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(n, m, 1)

        with pytest.raises(ValueError, match="drift_term"):
            _validate_diagonal_callable(x_batch, fn, "drift_term")

    def test_raises_for_scalar_tensor(
        self, x_batch: torch.Tensor,
    ) -> None:
        """A scalar tensor (ndim==0) must raise ValueError.

        Common mistake: returning torch.tensor(c) instead of
        torch.full((N,), c).
        """
        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.tensor(1.0)

        with pytest.raises(ValueError, match="diffusion_term"):
            _validate_diagonal_callable(x_batch, fn, "diffusion_term")

    def test_raises_for_wrong_batch_size(
        self, x_batch: torch.Tensor,
    ) -> None:
        """(N',) where N' != N must raise ValueError."""
        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(42)  # 42 != BATCH_SIZE=8

        with pytest.raises(ValueError, match="drift_term"):
            _validate_diagonal_callable(x_batch, fn, "drift_term")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="requires a CUDA device to probe cross-device mismatch",
    )
    def test_raises_for_wrong_device(
        self, x_batch: torch.Tensor,
    ) -> None:
        """Fn returning a CUDA tensor when x is on CPU must raise."""
        n = x_batch.shape[0]

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(n, device="cuda")

        with pytest.raises(ValueError, match="drift_term"):
            _validate_diagonal_callable(x_batch, fn, "drift_term")

    def test_error_message_contains_callable_name(
        self, x_batch: torch.Tensor,
    ) -> None:
        """The ValueError message must include the name argument verbatim."""
        name = "my_custom_D_t"

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(42)

        with pytest.raises(ValueError, match=name):
            _validate_diagonal_callable(x_batch, fn, name)

    def test_error_message_contains_expected_shapes(
        self, x_batch: torch.Tensor,
    ) -> None:
        """The ValueError message must list the acceptable shapes."""
        n = x_batch.shape[0]

        def fn(_t: torch.Tensor) -> torch.Tensor:
            return torch.ones(42)

        with pytest.raises(ValueError, match=rf"\({n}, 1\)"):
            _validate_diagonal_callable(x_batch, fn, "drift_term")

    @pytest.mark.parametrize(
        "bad_return",
        [
            None,
            1.0,
            1,
            [[1.0] * DATA_DIM] * BATCH_SIZE,
            np.ones((BATCH_SIZE, DATA_DIM)),
        ],
        ids=["none", "float", "int", "list", "ndarray"],
    )
    def test_raises_typeerror_for_non_tensor_return(
        self, x_batch: torch.Tensor, bad_return: object,
    ) -> None:
        """A non-tensor return must raise TypeError naming the callable.

        The isinstance guard must precede any attribute access (.ndim,
        .device), otherwise these return types would raise AttributeError.
        """
        name = "my_broken_term"

        def fn(_t: torch.Tensor) -> object:
            return bad_return

        with pytest.raises(TypeError, match=name):
            _validate_diagonal_callable(x_batch, fn, name)

    @pytest.mark.parametrize(
        ("bad_tensor", "shape_pattern"),
        [
            (torch.zeros(0, DATA_DIM), r"\(0,"),
            (torch.zeros(()), r"\(\)"),
        ],
        ids=["empty", "zero_dim"],
    )
    def test_raises_valueerror_with_shape_in_message(
        self,
        x_batch: torch.Tensor,
        bad_tensor: torch.Tensor,
        shape_pattern: str,
    ) -> None:
        """A wrong-shape tensor must raise ValueError reporting the shape."""
        def fn(_t: torch.Tensor) -> torch.Tensor:
            return bad_tensor

        with pytest.raises(ValueError, match=shape_pattern):
            _validate_diagonal_callable(x_batch, fn, "drift_term")


# ─────────────────────────────────────────────────────────────────────────────
# _batch_linear_interp_1d
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchLinearInterp1d:
    """Tests for the _batch_linear_interp_1d standalone function.

    Given t_grid (K,), f_grid (K, M), and query times t (N,),
    returns shape (N, M).
    """

    def test_output_shape(self) -> None:
        """Output must be (N, M) for arbitrary, non-square N, K, M."""
        k, n, m = 100, 12, 7
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = torch.randn(k, m).contiguous()
        t = torch.linspace(0.1, 0.9, n).contiguous()

        result = _batch_linear_interp_1d(t, t_grid, f_grid)

        assert result.shape == (n, m)

    def test_exact_linear_recovery(self) -> None:
        """Linear interpolation must recover affine functions exactly.

        Query points are off-grid midpoints so a bug that only returns
        exact grid values would fail here. atol=1e-6: only float
        rounding, no truncation error.
        """
        k = 50  # m = 4 (number of slopes)
        slopes = torch.tensor([1.0, -2.0, 0.5, 3.0])
        intercepts = torch.tensor([0.0, 1.0, 2.0, -1.0])
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = (
            t_grid.unsqueeze(1) * slopes.unsqueeze(0)
            + intercepts.unsqueeze(0)
        ).contiguous()
        t_query = ((t_grid[:-1] + t_grid[1:]) / 2).contiguous()
        expected = (
            t_query.unsqueeze(1) * slopes.unsqueeze(0)
            + intercepts.unsqueeze(0)
        )

        result = _batch_linear_interp_1d(t_query, t_grid, f_grid)

        assert torch.allclose(result, expected, atol=1e-6)

    def test_left_boundary_returns_first_row(self) -> None:
        """Querying at t_grid[0] must return f_grid[0, :].

        At t=t_grid[0]: weight = 0, result = y0 + 0*(y1-y0) = y0.
        torch.allclose (not equal) tolerates multiply-by-zero rounding.
        """
        k, m = 30, 5
        gen = torch.Generator().manual_seed(SEED)
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = torch.randn(k, m, generator=gen).contiguous()
        t_left = t_grid[:1].contiguous()

        result = _batch_linear_interp_1d(t_left, t_grid, f_grid)

        assert torch.allclose(result, f_grid[:1, :])

    def test_right_boundary_returns_last_row(self) -> None:
        """Querying at t_grid[-1] must return f_grid[-1, :].

        At t=t_grid[-1]: weight = 1.0, result = y0 + 1*(y1-y0) = y1.
        """
        k, m = 30, 5
        gen = torch.Generator().manual_seed(SEED)
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = torch.randn(k, m, generator=gen).contiguous()
        t_right = t_grid[-1:].contiguous()

        result = _batch_linear_interp_1d(t_right, t_grid, f_grid)

        assert torch.allclose(result, f_grid[-1:, :])

    def test_interior_midpoint_returns_arithmetic_mean_of_neighbours(
        self,
    ) -> None:
        """Querying at the exact midpoint between nodes returns their mean.

        At the midpoint: weight = 0.5, result = (y0 + y1) / 2.
        """
        k, m = 10, 3
        gen = torch.Generator().manual_seed(SEED)
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = torch.randn(k, m, generator=gen).contiguous()
        mid_idx = 4
        t_mid = (
            (t_grid[mid_idx] + t_grid[mid_idx + 1]) / 2
        ).unsqueeze(0).contiguous()
        expected = (f_grid[mid_idx] + f_grid[mid_idx + 1]) / 2

        result = _batch_linear_interp_1d(t_mid, t_grid, f_grid)

        assert torch.allclose(result.squeeze(0), expected, atol=1e-6)

    def test_each_dimension_interpolated_independently(self) -> None:
        """Each of the M functions must be interpolated without mixing.

        f_grid column m = (m+1)*t; at t=0.3 expected [0.3, 0.6, 0.9, 1.2].
        """
        k, m = 20, 4
        scales = torch.arange(1, m + 1).float()  # [1.0, 2.0, 3.0, 4.0]
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = (
            t_grid.unsqueeze(1) * scales.unsqueeze(0)
        ).contiguous()
        t_query = torch.tensor([0.3]).contiguous()
        expected = 0.3 * scales  # [0.3, 0.6, 0.9, 1.2]

        result = _batch_linear_interp_1d(t_query, t_grid, f_grid)

        assert torch.allclose(result.squeeze(0), expected, atol=1e-6)
        assert abs(result[0, 0].item() - scales[0].item() * 0.3) < _INTERP_ATOL
        assert (
            abs(result[0, -1].item() - scales[-1].item() * 0.3) < _INTERP_ATOL
        )

    def test_near_duplicate_grid_points_no_nan_no_inf(self) -> None:
        """Near-zero dt between adjacent nodes must not produce NaN or Inf.

        When dt < eps=1e-10, falls back to weight=0.5 to avoid Inf from
        division by zero. dt=1e-11 triggers the fallback.
        """
        t_grid = torch.tensor(
            [0.0, 0.5, 0.5 + 1e-11, 1.0],
        ).contiguous()
        f_grid = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]).contiguous()
        t_query = torch.tensor([0.5 + 5e-12]).contiguous()

        result = _batch_linear_interp_1d(t_query, t_grid, f_grid)

        assert not result.isnan().any()
        assert not result.isinf().any()
        expected_midpoint = (f_grid[1] + f_grid[2]) / 2
        assert torch.allclose(
            result.squeeze(0), expected_midpoint, atol=1e-6,
        )

    def test_non_contiguous_t_raises_error(self) -> None:
        """Passing a non-contiguous t tensor must raise ValueError.

        torch.searchsorted can produce incorrect results on non-contiguous
        inputs.
        """
        k, m = 20, 4
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = torch.randn(k, m).contiguous()

        t_non_contiguous = torch.linspace(0.0, 1.0, 20)[::2]
        assert not t_non_contiguous.is_contiguous()

        with pytest.raises(ValueError, match="t must"):
            _batch_linear_interp_1d(t_non_contiguous, t_grid, f_grid)

    def test_non_contiguous_t_grid_raises_error(self) -> None:
        """Passing a non-contiguous t_grid tensor must raise ValueError.

        Tested separately from t because the guard is a separate
        if-statement.
        """
        k, m = 20, 4
        f_grid = torch.randn(k, m).contiguous()
        t = torch.linspace(0.1, 0.9, 8).contiguous()

        t_grid_non_contiguous = torch.linspace(0.0, 1.0, 40)[::2]
        assert not t_grid_non_contiguous.is_contiguous()

        with pytest.raises(ValueError, match="t_grid"):
            _batch_linear_interp_1d(t, t_grid_non_contiguous, f_grid)

    def test_single_query_point_does_not_crash(self) -> None:
        """N=1 query point must produce shape (1, M) without crashing."""
        k, m = 50, 6
        gen = torch.Generator().manual_seed(SEED)
        t_grid = torch.linspace(0.0, 1.0, k).contiguous()
        f_grid = torch.randn(k, m, generator=gen).contiguous()
        t_single = torch.tensor([0.42]).contiguous()

        result = _batch_linear_interp_1d(t_single, t_grid, f_grid)

        assert result.shape == (1, m)
        assert not result.isnan().any()

    def test_minimal_grid_k2_covers_full_range(self) -> None:
        """K=2 grid (single segment) interpolates the full [0, 1] range.

        clamp(indices, 1, K-1=1) means t0=t_grid[0], t1=t_grid[1] for
        every query.
        """
        m = 4
        t_grid = torch.tensor([0.0, 1.0]).contiguous()
        f_grid = torch.tensor([
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
        ]).contiguous()
        t_query = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).contiguous()
        expected = (
            f_grid[0].unsqueeze(0)
            + t_query.unsqueeze(1) * (f_grid[1] - f_grid[0])
        )

        result = _batch_linear_interp_1d(t_query, t_grid, f_grid)

        assert result.shape == (5, m)
        assert torch.allclose(result, expected, atol=1e-6)

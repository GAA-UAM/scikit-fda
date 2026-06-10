"""Correctness tests for diffusion process implementations."""

from __future__ import annotations

import pytest
import torch

from skfda.ml.generative._diffusion_process import (
    CustomDiffusionProcess,
    DiagonalDiffusionProcess,
    ForwardDiffusionProcess,
    VarianceExplodingDiffusionProcess,
    VariancePreservingDiffusionProcess,
)

from ._constants import BATCH_SIZE, CUSTOM_DIM, DATA_DIM, SEED
from .diffusion_test_mixins import (
    ForwardDiffusionCheckpointTests,
    ForwardDiffusionFitTests,
    ForwardDiffusionSampleLimitTests,
    ScalarCovarianceOperatorTests,
)


@pytest.fixture(scope="module")
def v_batch() -> torch.Tensor:
    """Noise tensor shape (N, M); independent from x_batch (different seed)."""
    gen = torch.Generator()
    gen.manual_seed(SEED + 1)
    return torch.randn(BATCH_SIZE, DATA_DIM, generator=gen)


# ─────────────────────────────────────────────────────────────
# Drift and diffusion callables for CustomDiffusionProcess
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def zero_drift():
    """f(x, t) = 0."""
    def _drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    return _drift


@pytest.fixture(scope="module")
def identity_drift():
    """f(x, t) = x."""
    def _drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x.clone()
    return _drift


@pytest.fixture(scope="module")
def scalar_diffusion():
    """g(t) = t — scalar per sample; triggers the ndim==1 branch."""
    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return t.clone()
    return _diffusion


@pytest.fixture(scope="module")
def diagonal_diffusion():
    """g(t) = ones(N, M) — diagonal identity; triggers the ndim==2 branch."""
    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.ones(t.shape[0], DATA_DIM)
    return _diffusion


@pytest.fixture(scope="module")
def matrix_diffusion():
    """g(t) = I expanded to (N, M, M) — triggers the ndim==3 branch."""
    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(DATA_DIM)
        return eye.unsqueeze(0).expand(t.shape[0], -1, -1)
    return _diffusion


@pytest.fixture(scope="module")
def invalid_diffusion():
    """g(t) returns a 4D tensor — triggers the ValueError branch."""
    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.ones(t.shape[0], DATA_DIM, DATA_DIM, 2)
    return _diffusion


# ─────────────────────────────────────────────────────────────
# CustomDiffusionProcess instances
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def custom_process_scalar(zero_drift, scalar_diffusion):
    """CustomDiffusionProcess with scalar (ndim=1) diffusion."""
    return CustomDiffusionProcess(drift=zero_drift, diffusion=scalar_diffusion)


@pytest.fixture(scope="module")
def custom_process_diagonal(zero_drift, diagonal_diffusion):
    """CustomDiffusionProcess with diagonal (ndim=2) diffusion."""
    return CustomDiffusionProcess(drift=zero_drift, diffusion=diagonal_diffusion)


@pytest.fixture(scope="module")
def custom_process_matrix(zero_drift, matrix_diffusion):
    """CustomDiffusionProcess with full matrix (ndim=3) diffusion."""
    return CustomDiffusionProcess(drift=zero_drift, diffusion=matrix_diffusion)


# ─────────────────────────────────────────────────────────────
# VariancePreservingDiffusionProcess instances (fitted)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vp_linear(x_batch) -> "VariancePreservingDiffusionProcess":
    """Linear VP fitted on x_batch."""
    vp = VariancePreservingDiffusionProcess(
        beta_schedule="linear",
        beta_min=0.1,
        beta_max=10.0,
    )
    vp.fit(x_batch)
    return vp


@pytest.fixture(scope="module")
def vp_cosine(x_batch) -> "VariancePreservingDiffusionProcess":
    """Cosine VP fitted on x_batch (beta_min/max act as clamp)."""
    vp = VariancePreservingDiffusionProcess(
        beta_schedule="cosine",
        beta_min=0.0,
        beta_max=10.0,
    )
    vp.fit(x_batch)
    return vp


@pytest.fixture
def vp_linear_unfitted() -> "VariancePreservingDiffusionProcess":
    """Linear VP, unfitted. Function-scoped so tests can call fit() freely."""
    return VariancePreservingDiffusionProcess(
        beta_schedule="linear",
        beta_min=0.1,
        beta_max=10.0,
    )


@pytest.fixture
def vp_cosine_unfitted() -> "VariancePreservingDiffusionProcess":
    """Cosine VP, unfitted. Function-scoped."""
    return VariancePreservingDiffusionProcess(
        beta_schedule="cosine",
        beta_min=0.0,
        beta_max=10.0,
    )


@pytest.fixture(params=["linear", "cosine"], ids=["linear", "cosine"])
def vp_any_schedule(request, x_batch) -> "VariancePreservingDiffusionProcess":
    """Fitted VP parametrized over both schedules; generates [linear]/[cosine]."""
    beta_min = 0.1 if request.param == "linear" else 0.0
    vp = VariancePreservingDiffusionProcess(
        beta_schedule=request.param,
        beta_min=beta_min,
        beta_max=10.0,
    )
    vp.fit(x_batch)
    return vp


class TestDiffusionTimesV:
    """Tests for the diffusion_times_v method of DiffusionProcess."""

    def test_output_shape_scalar(self, custom_process_scalar, v_batch, t_batch):
        """Output shape must match v."""
        result = custom_process_scalar.diffusion_times_v(v_batch, t_batch)

        assert result.shape == v_batch.shape

    def test_output_values_scalar(self, custom_process_scalar, v_batch, t_batch):
        """Row n must be scaled by diffusion_term[n] = t[n] across all M dimensions."""
        result = custom_process_scalar.diffusion_times_v(v_batch, t_batch)

        expected = t_batch.unsqueeze(1) * v_batch
        assert torch.allclose(result, expected)

    def test_each_row_scaled_by_its_own_scalar(
        self, custom_process_scalar, v_batch, t_batch,
    ):
        """Row n must be scaled by diffusion_term[n], not by any other entry."""
        result = custom_process_scalar.diffusion_times_v(v_batch, t_batch)

        # Avoid division by zero: only check positions where v != 0
        nonzero_mask = v_batch.abs() > 1e-6
        ratios = result[nonzero_mask] / v_batch[nonzero_mask]

        expected_ratios = t_batch.unsqueeze(1).expand_as(v_batch)[nonzero_mask]
        assert torch.allclose(ratios, expected_ratios, atol=1e-6)

    def test_zero_scalar_gives_zero_output(
        self, custom_process_scalar, v_batch, t_zero,
    ):
        """When g(t) = t and t = 0, output must be zero."""
        result = custom_process_scalar.diffusion_times_v(v_batch, t_zero)

        assert torch.allclose(result, torch.zeros_like(v_batch))

    def test_output_shape_diagonal(
        self, custom_process_diagonal, v_batch, t_batch,
    ):
        """Output shape must match v."""
        result = custom_process_diagonal.diffusion_times_v(v_batch, t_batch)

        assert result.shape == v_batch.shape

    def test_output_values_ones_diagonal(
        self, custom_process_diagonal, v_batch, t_batch,
    ):
        """Ones diagonal leaves v unchanged."""
        result = custom_process_diagonal.diffusion_times_v(v_batch, t_batch)

        assert torch.allclose(result, v_batch)

    def test_non_trivial_diagonal_scales_each_dimension_independently(
        self, zero_drift, v_batch, t_batch,
    ):
        """Each dimension m must be scaled by diffusion_term[n, m] independently."""
        M = v_batch.shape[1]
        scale = torch.arange(1, M + 1, dtype=torch.float32)

        def column_scaled_diffusion(t: torch.Tensor) -> torch.Tensor:
            return scale.unsqueeze(0).expand(t.shape[0], -1)

        proc = CustomDiffusionProcess(drift=zero_drift, diffusion=column_scaled_diffusion)
        result = proc.diffusion_times_v(v_batch, t_batch)

        expected = scale.unsqueeze(0) * v_batch
        assert torch.allclose(result, expected)

    def test_zero_diagonal_gives_zero_output(self, zero_drift, v_batch, t_batch):
        """Zero diagonal maps any v to zero."""
        M = v_batch.shape[1]

        def zero_diagonal_diffusion(t: torch.Tensor) -> torch.Tensor:
            return torch.zeros(t.shape[0], M)

        proc = CustomDiffusionProcess(drift=zero_drift, diffusion=zero_diagonal_diffusion)
        result = proc.diffusion_times_v(v_batch, t_batch)

        assert torch.allclose(result, torch.zeros_like(v_batch))

    def test_output_shape_matrix(self, custom_process_matrix, v_batch, t_batch):
        """Output shape must match v."""
        result = custom_process_matrix.diffusion_times_v(v_batch, t_batch)

        assert result.shape == v_batch.shape

    def test_identity_matrix_leaves_v_unchanged(
        self, custom_process_matrix, v_batch, t_batch,
    ):
        """Identity matrix diffusion leaves v unchanged."""
        result = custom_process_matrix.diffusion_times_v(v_batch, t_batch)

        assert torch.allclose(result, v_batch)

    def test_result_matches_torch_bmm(self, zero_drift, v_batch, t_batch):
        """The einsum must match torch.bmm (verified with upper-triangular matrix)."""
        N, M = v_batch.shape
        fixed_matrix = torch.triu(torch.ones(M, M))

        def upper_tri_diffusion(t: torch.Tensor) -> torch.Tensor:
            return fixed_matrix.unsqueeze(0).expand(t.shape[0], -1, -1)

        proc = CustomDiffusionProcess(drift=zero_drift, diffusion=upper_tri_diffusion)
        result = proc.diffusion_times_v(v_batch, t_batch)

        # Reference: batched matrix-vector product via torch.bmm
        g_expanded = fixed_matrix.unsqueeze(0).expand(N, -1, -1)
        expected = torch.bmm(g_expanded, v_batch.unsqueeze(2)).squeeze(2)

        assert torch.allclose(result, expected)

    def test_per_sample_matrices_are_applied_independently(
        self, zero_drift, v_batch, t_batch,
    ):
        """Each sample n must use its own matrix; diffusion_term[n] = (n+1)*I → result[n] = (n+1)*v[n]."""
        N, M = v_batch.shape

        def sample_scaled_identity(t: torch.Tensor) -> torch.Tensor:
            scale = torch.arange(1, N + 1, dtype=torch.float32)
            eye = torch.eye(M)
            return scale.view(N, 1, 1) * eye.unsqueeze(0)

        proc = CustomDiffusionProcess(
            drift=zero_drift,
            diffusion=sample_scaled_identity,
        )
        result = proc.diffusion_times_v(v_batch, t_batch)

        scale = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(1)
        expected = scale * v_batch
        assert torch.allclose(result, expected)

    def test_zero_dim_scalar_tensor_raises_value_error(
        self, zero_drift, v_batch, t_batch,
    ):
        """A 0-dimensional diffusion tensor must raise ValueError.

        A plausible mistake producing ndim=0:
            return torch.tensor(0.5)            # wrong: scalar, ndim=0
        instead of:
            return torch.full((t.shape[0],), 0.5)  # correct: (N,), ndim=1
        """
        def scalar_tensor_diffusion(t: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.5)  # ndim=0

        proc = CustomDiffusionProcess(drift=zero_drift, diffusion=scalar_tensor_diffusion)

        with pytest.raises(ValueError, match="Invalid diffusion term shape"):
            proc.diffusion_times_v(v_batch, t_batch)

    @pytest.mark.parametrize("extra_dims", [1, 2, 3], ids=["ndim4", "ndim5", "ndim6"])
    def test_any_ndim_above_3_raises(self, zero_drift, v_batch, t_batch, extra_dims):
        """All tensors with ndim > 3 must raise ValueError."""
        M = v_batch.shape[1]

        def high_dim_diffusion(t: torch.Tensor) -> torch.Tensor:
            extra_shape = (2,) * extra_dims
            return torch.ones(t.shape[0], M, M, *extra_shape)

        proc = CustomDiffusionProcess(
            drift=zero_drift,
            diffusion=high_dim_diffusion,
        )

        with pytest.raises(ValueError, match="Invalid diffusion term shape"):
            proc.diffusion_times_v(v_batch, t_batch)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VariancePreservingDiffusionProcess — fit / checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestVPFit(ForwardDiffusionFitTests, ForwardDiffusionCheckpointTests):
    """Fit and checkpoint tests for VariancePreservingDiffusionProcess.

    VP's fit() learns only M. If VP ever overrides fit() to learn additional
    parameters, extend _get_fit_state() and update this class.
    """

    @pytest.fixture
    def make_process(self):
        return lambda: VariancePreservingDiffusionProcess(
            beta_schedule="linear",
            beta_min=0.1,
            beta_max=10.0,
        )

    def test_checkpoint_round_trip(self, make_process):
        """Full state must survive to_checkpoint() → from_checkpoint().

        Uses two independent instances to confirm genuine transfer.
        """
        instance_a = make_process()
        instance_a.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = instance_a.to_checkpoint()
        instance_b = ForwardDiffusionProcess.from_checkpoint(checkpoint)

        assert type(instance_b) is VariancePreservingDiffusionProcess
        assert instance_b.M_ == instance_a.M_
        assert instance_b.beta_schedule == instance_a.beta_schedule
        assert instance_b.beta_min == instance_a.beta_min
        assert instance_b.beta_max == instance_a.beta_max


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VarianceExplodingDiffusionProcess — fit / checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestVEFit(ForwardDiffusionFitTests, ForwardDiffusionCheckpointTests):
    """Fit and checkpoint tests for VarianceExplodingDiffusionProcess.

    VE's fit() learns only M; g_schedule/g_0/g_T are stored via get_params().
    If VE ever overrides fit() to learn additional parameters, update here.
    """

    @pytest.fixture
    def make_process(self):
        return lambda: VarianceExplodingDiffusionProcess(
            g_schedule="exponential",
            g_0=0.1,
            g_T=15.0,
        )

    def test_checkpoint_round_trip(self, make_process):
        """Full state must survive to_checkpoint() → from_checkpoint().

        Uses two independent instances to confirm genuine transfer.
        """
        instance_a = make_process()
        instance_a.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = instance_a.to_checkpoint()
        instance_b = ForwardDiffusionProcess.from_checkpoint(checkpoint)

        assert type(instance_b) is VarianceExplodingDiffusionProcess
        assert instance_b.M_ == instance_a.M_
        assert instance_b.g_schedule == instance_a.g_schedule
        assert instance_b.g_0 == instance_a.g_0
        assert instance_b.g_T == instance_a.g_T


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VariancePreservingDiffusionProcess — covariance operators
# ─────────────────────────────────────────────────────────────────────────────

class TestVariancePreservingDiffusionProcessCov(ScalarCovarianceOperatorTests):
    """Tests for _cov and covariance operators of VP.

    Inherits 4 tests from ScalarCovarianceOperatorTests. Adds VP-specific
    tests for _cov boundary values and the μ_t² + sigma_t² = 1 identity.
    Parametrized over linear and cosine schedules.
    """

    @pytest.fixture(params=["linear", "cosine"], ids=["linear", "cosine"])
    def process(self, request, x_batch) -> VariancePreservingDiffusionProcess:
        """Fitted VP for both schedules; provides 'process' for mixin tests."""
        beta_min = 0.1 if request.param == "linear" else 0.0
        vp = VariancePreservingDiffusionProcess(
            beta_schedule=request.param,
            beta_min=beta_min,
            beta_max=10.0,
        )
        vp.fit(x_batch)
        return vp

    def test_cov_at_t0_is_zero(self, process, t_zero):
        """_cov(0) must equal zero exactly for both schedules.

        Linear:  1 - exp(-0) = 0
        Cosine:  1 - f(0)/f(0) = 0
        """
        result = process._cov(t_zero)

        assert torch.allclose(result, torch.zeros_like(result))

    def test_cov_at_tT_is_close_to_one(self, process, t_one):
        """_cov(T) must be close to 1 (atol=0.01 covers both schedules).

        Cosine: 1 - cos(π/2)²/f(0) = 1.000
        Linear: 1 - exp(-5.05)     = 0.993  (beta_min=0.1, beta_max=10.0)
        """
        result = process._cov(t_one)

        assert torch.allclose(result, torch.ones_like(result), atol=0.01)

    def test_variance_preserving_identity(self, process, t_sweep):
        """μ_t² + _cov(t) must equal 1 for every t ∈ (0, 1).

        The defining VP contract; does not hold for VE or other processes.
        μ_t is extracted as mean_cond(ones, t)[:, 0] — x=ones isolates the
        scalar decay coefficient. atol=1e-6 gives 100x margin over float32
        residual (~6e-8, verified analytically for both schedules).
        """
        x_ones = torch.ones(t_sweep.shape[0], DATA_DIM)

        mu_t  = process.mean_cond(x_ones, t_sweep)[:, 0]  # scalar decay coefficient
        cov_t = process._cov(t_sweep)

        assert torch.allclose(mu_t ** 2 + cov_t, torch.ones_like(cov_t), atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VarianceExplodingDiffusionProcess — covariance operators
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceExplodingDiffusionProcessCov(ScalarCovarianceOperatorTests):
    """Tests for _cov and covariance operators of VE.

    Inherits 4 tests from ScalarCovarianceOperatorTests. Adds VE-specific
    tests for _cov(0)=0, monotonicity, variance-explosion at T, and
    closed-form oracle checks per schedule.
    """

    @pytest.fixture(params=["linear", "exponential"], ids=["linear", "exponential"])
    def process(self, request, x_batch) -> VarianceExplodingDiffusionProcess:
        """Fitted VE for both schedules; provides 'process' for mixin tests."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule=request.param,
            g_0=0.1,
            g_T=15.0,
        )
        ve.fit(x_batch)
        return ve

    @pytest.fixture
    def process_linear(self, x_batch) -> VarianceExplodingDiffusionProcess:
        """VE with linear schedule; used only by the linear oracle test."""
        ve = VarianceExplodingDiffusionProcess(g_schedule="linear", g_0=0.1, g_T=15.0)
        ve.fit(x_batch)
        return ve

    @pytest.fixture
    def process_exp(self, x_batch) -> VarianceExplodingDiffusionProcess:
        """VE with exponential schedule; used only by the exponential oracle test."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule="exponential", g_0=0.1, g_T=15.0,
        )
        ve.fit(x_batch)
        return ve

    def test_cov_at_t0_is_zero(self, process, t_zero):
        """_cov(0) must equal zero exactly for both schedules.

        sigma₀² = ∫₀⁰ g(s)² ds = 0. Both schedules achieve this algebraically:
        Linear: every term carries a factor of t → 0.
        Exponential: (gT/g₀)^0 - 1 = 0.
        """
        result = process._cov(t_zero)

        assert torch.allclose(result, torch.zeros_like(result))

    def test_cov_is_monotonically_nondecreasing(self, process, t_sweep):
        """_cov(t) must be non-decreasing: sigma_t² = ∫₀ᵗ g(s)² ds, g > 0."""
        cov_values = process._cov(t_sweep)
        diffs      = cov_values[1:] - cov_values[:-1]

        assert torch.all(diffs >= 0), (
            f"_cov is not monotonically non-decreasing: "
            f"min consecutive diff = {diffs.min().item():.6e}"
        )

    def test_cov_at_tT_is_large(self, process, t_one):
        """_cov(T) must be >> 1 for both schedules (variance-exploding property).

        With g₀=0.1, gT=15.0:
            Linear:      _cov(1) ≈ 75.5
            Exponential: _cov(1) ≈ 22.5

        Threshold 10.0 is conservative but clearly distinguishes VE from VP.
        """
        result = process._cov(t_one)

        assert torch.all(result > 10.0), (
            f"_cov(T) too small for a variance-exploding process: "
            f"min found = {result.min().item():.4f}"
        )

    def test_linear_cov_matches_formula_at_known_points(self, process_linear):
        """_cov(t) for the linear schedule must match hand-computed oracle values.

        sigma_t² = g₀²·t + g₀·(gT-g₀)·t²/T + (gT-g₀)²·t³/(3T²)

        With g₀=0.1, gT=15.0, T=1:
            t=0.00 →  0.000000
            t=0.25 →  1.251927
            t=0.50 →  9.627917
            t=0.75 → 32.065781
            t=1.00 → 75.503333
        """
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = torch.tensor([0.000000, 1.251927, 9.627917, 32.065781, 75.503333])

        result = process_linear._cov(t)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_exponential_cov_matches_formula_at_known_points(self, process_exp):
        """_cov(t) for the exponential schedule must match hand-computed oracle values.

        sigma_t² = g₀²T / (2·log(gT/g₀)) · [(gT/g₀)^(2t/T) - 1]

        With g₀=0.1, gT=15.0, T=1 (factor ≈ 9.979e-4):
            t=0.00 →  0.000000
            t=0.25 →  0.011224
            t=0.50 →  0.148684
            t=0.75 →  1.832221
            t=1.00 → 22.451267

        atol=1e-3 accounts for limited precision of the hand-computed log constant.
        """
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = torch.tensor(
            [0.000000, 0.011224, 0.148684, 1.832221, 22.451267],
        )

        result = process_exp._cov(t)

        assert torch.allclose(result, expected, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VariancePreservingDiffusionProcess — sample_limit_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestVPSampleLimit(ForwardDiffusionSampleLimitTests):
    """sample_limit_distribution tests for VP.

    Inherits 6 universal tests from the mixin. Adds one VP-specific test
    confirming the limit distribution is N(0, I).
    """

    @pytest.fixture
    def make_process(self):
        return lambda: VariancePreservingDiffusionProcess(
            beta_schedule="linear",
            beta_min=0.1,
            beta_max=10.0,
            seed = SEED,
        )

    @pytest.fixture
    def make_process_alt_seed(self):
        """Same VP schedule but different seed; used to confirm genuine sampling."""
        return lambda: VariancePreservingDiffusionProcess(
            beta_schedule="linear",
            beta_min=0.1,
            beta_max=10.0,
            seed = SEED + 1,
        )

    def test_limit_distribution_is_approximately_standard_normal(
        self, fitted_process
    ):
        """VP limit distribution must be approximately N(0, I).

        With n=10_000: SE of mean estimator ≈ 0.01 per dimension.
        atol=0.05 is a ~5-sigma guard; P(false failure across M=16 dims) < 1e-5.
        """
        n = 10_000
        samples = fitted_process.sample_limit_distribution(
            n_samples=n,
        )

        assert samples.mean(dim=0).abs().max() < 0.05, (
            f"Per-dimension mean too large: {samples.mean(dim=0).abs().max():.4f}"
        )
        assert (samples.std(dim=0) - 1.0).abs().max() < 0.05, (
            f"Per-dimension std too far from 1: "
            f"{(samples.std(dim=0) - 1.0).abs().max():.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VarianceExplodingDiffusionProcess — sample_limit_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestVESampleLimit(ForwardDiffusionSampleLimitTests):
    """sample_limit_distribution tests for VE.

    Inherits 6 universal tests. Adds a statistical test that the limit
    distribution is N(0, sigma_T² · I) where sigma_T² = _cov(T).
    """

    @pytest.fixture
    def make_process(self):
        return lambda: VarianceExplodingDiffusionProcess(
            g_schedule="exponential",
            g_0=0.1,
            g_T=15.0,
            seed = SEED,
        )

    @pytest.fixture
    def make_process_alt_seed(self):
        """Same VE schedule but different seed; used to confirm genuine sampling."""
        return lambda: VarianceExplodingDiffusionProcess(
            g_schedule="exponential",
            g_0=0.1,
            g_T=15.0,
            seed = SEED + 1,
        )

    def test_limit_distribution_statistics(self, fitted_process):
        """Samples must be approximately N(0, sigma_T² · I).

        With n=10_000 and the exponential fixture (sigma_T ≈ 4.74):
            mean_tol  = 5 · sigma_T / √n ≈ 0.237  (5-SE guard per dimension)
            std check = |std_d / sigma_T - 1| < 0.05  (~5% relative tolerance)

        P(spurious failure across M=16 dimensions) < 1e-4.
        """
        n = 10_000
        samples = fitted_process.sample_limit_distribution(
            n_samples=n,
        )

        t_end   = torch.tensor([float(fitted_process.T)])
        sigma_T = torch.sqrt(fitted_process._cov(t_end)).item()

        mean_tol = 5.0 * sigma_T / (n ** 0.5)
        assert samples.mean(dim=0).abs().max().item() < mean_tol, (
            f"Per-dimension mean too large (tol={mean_tol:.4f}): "
            f"max |mean| = {samples.mean(dim=0).abs().max().item():.4f}"
        )
        assert ((samples.std(dim=0) / sigma_T) - 1.0).abs().max().item() < 0.05, (
            f"Per-dimension std not close to sigma_T={sigma_T:.4f}: "
            f"max relative error = "
            f"{((samples.std(dim=0) / sigma_T) - 1.0).abs().max().item():.4f}"
        )


class TestVariancePreservingDiffusionProcessInit:
    """Tests for the VariancePreservingDiffusionProcess constructor."""

    def test_invalid_schedule_raises_error(self):
        """Unsupported beta_schedule must raise ValueError in __init__."""
        with pytest.raises(ValueError):
            VariancePreservingDiffusionProcess(beta_schedule="exponential")

    def test_linear_schedule_initializes_correctly(self):
        """Constructor args must be stored verbatim.

        Non-default values rule out accidental matches with defaults.
        """
        vp = VariancePreservingDiffusionProcess(
            beta_schedule="linear",
            beta_min=0.1,
            beta_max=5.0,
        )

        assert vp.beta_schedule == "linear"
        assert vp.beta_min == 0.1
        assert vp.beta_max == 5.0

    def test_cosine_schedule_initializes_correctly(self):
        """Cosine schedule must be accepted and stored; M must not exist before fit()."""
        vp = VariancePreservingDiffusionProcess(beta_schedule="cosine")

        assert vp.beta_schedule == "cosine"
        assert not hasattr(vp, "M")


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VarianceExplodingDiffusionProcess — Init tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceExplodingDiffusionProcessInit:
    """Tests for the VarianceExplodingDiffusionProcess constructor."""

    def test_invalid_schedule_raises_error(self):
        """Unsupported g_schedule must raise ValueError in __init__."""
        with pytest.raises(ValueError):
            VarianceExplodingDiffusionProcess(g_schedule="cosine")

    def test_linear_schedule_initializes_correctly(self):
        """Constructor args must be stored verbatim.

        Non-default values rule out accidental matches with defaults.
        """
        ve = VarianceExplodingDiffusionProcess(
            g_schedule="linear",
            g_0=0.5,
            g_T=10.0,
        )

        assert ve.g_schedule == "linear"
        assert ve.g_0 == 0.5
        assert ve.g_T == 10.0

    def test_exponential_schedule_initializes_correctly(self):
        """Exponential schedule must be accepted; M must not exist before fit()."""
        ve = VarianceExplodingDiffusionProcess(g_schedule="exponential")

        assert ve.g_schedule == "exponential"
        assert not hasattr(ve, "M")


class TestVariancePreservingDiffusionProcessMeanCond:
    """Tests for mean_cond of VP.

    mean_cond(x, t) = μ_t · x where μ_t ∈ [0,1] decreases from 1 to ≈0.
    All tests are parametrized over linear and cosine via vp_any_schedule.
    """

    def test_mean_cond_at_t0_equals_x(self, vp_any_schedule, x_batch, t_zero):
        """mean_cond(x, 0) must equal x exactly for both schedules.

        Linear:  μ_0 = exp(-0.5 · ∫β ds) = exp(0) = 1
        Cosine:  μ_0 = √(f(0)/f(0)) = 1
        """
        result = vp_any_schedule.mean_cond(x_batch, t_zero)

        assert torch.allclose(result, x_batch)

    def test_mean_cond_at_tT_is_close_to_zero(self, vp_any_schedule, t_one):
        """mean_cond(x, T) must be close to zero for both schedules.

        Cosine: μ_T ≈ 4.4e-8  (machine zero)
        Linear: μ_T ≈ 0.080   (beta_min=0.1, beta_max=10.0, integral≈5.05)

        x=ones so result[n, m] = μ_T directly. atol=0.1 covers both.
        """
        x_ones = torch.ones(BATCH_SIZE, DATA_DIM)

        result = vp_any_schedule.mean_cond(x_ones, t_one)

        assert torch.allclose(result, torch.zeros_like(result), atol=0.1)

    def test_mean_cond_output_shape(self, vp_any_schedule, x_batch, t_batch):
        """Output shape must match x."""
        result = vp_any_schedule.mean_cond(x_batch, t_batch)

        assert result.shape == x_batch.shape

    def test_mean_cond_linearity_in_x(
        self, vp_any_schedule, x_batch, x_batch_scaled, t_batch,
    ):
        """mean_cond(2·x, t) must equal 2·mean_cond(x, t)."""
        result_x  = vp_any_schedule.mean_cond(x_batch, t_batch)
        result_2x = vp_any_schedule.mean_cond(x_batch_scaled, t_batch)

        assert torch.allclose(result_2x, 2.0 * result_x)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VarianceExplodingDiffusionProcess — MeanCond tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceExplodingDiffusionProcessMeanCond:
    """Tests for mean_cond of VE.

    VE has zero drift: mean_cond(x, t) = x for all t. Noise is added but
    the signal is never scaled or removed — the defining contrast with VP.
    Parametrized over linear and exponential schedules.
    """

    @pytest.fixture(params=["linear", "exponential"], ids=["linear", "exponential"])
    def process(self, request, x_batch) -> VarianceExplodingDiffusionProcess:
        """Fitted VE for both schedules."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule=request.param,
            g_0=0.1,
            g_T=15.0,
        )
        ve.fit(x_batch)
        return ve

    def test_mean_cond_at_t0_equals_x(self, process, x_batch, t_zero):
        """mean_cond(x, 0) must equal x."""
        result = process.mean_cond(x_batch, t_zero)

        assert torch.allclose(result, x_batch)

    def test_mean_cond_at_tT_equals_x(self, process, x_batch, t_one):
        """mean_cond(x, T) must equal x — zero-drift VE never scales the signal."""
        result = process.mean_cond(x_batch, t_one)

        assert torch.allclose(result, x_batch)

    def test_mean_cond_equals_x_for_interior_t(self, process, x_batch, t_batch):
        """mean_cond(x, t) must equal x for any interior t."""
        result = process.mean_cond(x_batch, t_batch)

        assert torch.allclose(result, x_batch)

    def test_mean_cond_output_shape(self, process, x_batch, t_batch):
        """Output shape must match x."""
        result = process.mean_cond(x_batch, t_batch)

        assert result.shape == x_batch.shape

    def test_mean_cond_is_independent_of_schedule_parameters(
        self, x_batch, t_batch,
    ):
        """mean_cond must equal x regardless of g_schedule, g_0, or g_T.

        torch.equal (not allclose) is intentional: the implementation returns
        x directly with no arithmetic, so there is no rounding error.
        """
        ve_linear = VarianceExplodingDiffusionProcess(
            g_schedule="linear", g_0=0.01, g_T=100.0,
        )
        ve_linear.fit(x_batch)

        ve_exp = VarianceExplodingDiffusionProcess(
            g_schedule="exponential", g_0=0.1, g_T=15.0,
        )
        ve_exp.fit(x_batch)

        result_linear = ve_linear.mean_cond(x_batch, t_batch)
        result_exp    = ve_exp.mean_cond(x_batch, t_batch)

        assert torch.equal(result_linear, x_batch)
        assert torch.equal(result_exp,    x_batch)


class TestVariancePreservingDiffusionProcessBetaT:
    """Tests for the _beta_t method of VP.

    _beta_t(t) maps time steps to noise levels β(t) ∈ [beta_min, beta_max].
    All higher-level methods delegate time-dependent behaviour to this function.
    """

    def test_linear_beta_t_at_t0_equals_beta_min(self, vp_linear, t_zero):
        """β(0) must equal beta_min exactly.

        β(t) = beta_min + (beta_max - beta_min)·t/T; second term vanishes at t=0.
        """
        result = vp_linear._beta_t(t_zero)

        expected = torch.full_like(t_zero, vp_linear.beta_min)
        assert torch.allclose(result, expected)

    def test_linear_beta_t_at_tT_equals_beta_max(self, vp_linear, t_one):
        """β(T) must equal beta_max exactly."""
        result = vp_linear._beta_t(t_one)

        expected = torch.full_like(t_one, vp_linear.beta_max)
        assert torch.allclose(result, expected)

    def test_linear_beta_t_matches_formula_at_known_points(self, vp_linear):
        """β(t) must match hand-computed values.

        β(t) = beta_min + (beta_max - beta_min)·t/T
        With beta_min=0.1, beta_max=10.0, T=1:

            t=0.00 →  0.100
            t=0.25 →  2.575
            t=0.50 →  5.050
            t=0.75 →  7.525
            t=1.00 → 10.000
        """
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        result = vp_linear._beta_t(t)

        expected = torch.tensor([0.100, 2.575, 5.050, 7.525, 10.000])
        assert torch.allclose(result, expected)

    def test_cosine_beta_t_is_clamped_between_min_and_max(
        self, vp_cosine, t_sweep,
    ):
        """β(t) (cosine schedule) must stay in [beta_min, beta_max].

        The cosine tangent grows unboundedly near t=T; torch.clamp keeps it in
        range. t_sweep exercises the interior including the high-curvature region.
        """
        result = vp_cosine._beta_t(t_sweep)

        assert torch.all(result >= vp_cosine.beta_min), (
            f"Some β(t) values fall below beta_min={vp_cosine.beta_min}: "
            f"min found {result.min().item():.6f}"
        )
        assert torch.all(result <= vp_cosine.beta_max), (
            f"Some β(t) values exceed beta_max={vp_cosine.beta_max}: "
            f"max found {result.max().item():.6f}"
        )

    def test_beta_t_output_shape_matches_input(self, vp_any_schedule, t_batch):
        """Output shape must match t."""
        result = vp_any_schedule._beta_t(t_batch)

        assert result.shape == t_batch.shape


# ─────────────────────────────────────────────────────────────────────────────
# Concrete: VarianceExplodingDiffusionProcess — Diffusion tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVarianceExplodingDiffusionProcessDiffusion:
    """Tests for the diffusion method of VE: g(t) maps (N,) → (N,).

    g(0) = g_0, g(T) = g_T, g(t) > 0 for all t.
    g(t) is the square root of the _cov integrand: sigma_t² = ∫₀ᵗ g(s)² ds.
    """

    @pytest.fixture(params=["linear", "exponential"], ids=["linear", "exponential"])
    def process(self, request, x_batch) -> VarianceExplodingDiffusionProcess:
        """Fitted VE for both schedules."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule=request.param,
            g_0=0.1,
            g_T=15.0,
        )
        ve.fit(x_batch)
        return ve

    @pytest.fixture
    def process_linear(self, x_batch) -> VarianceExplodingDiffusionProcess:
        """VE with linear schedule; used only by the linear oracle test."""
        ve = VarianceExplodingDiffusionProcess(g_schedule="linear", g_0=0.1, g_T=15.0)
        ve.fit(x_batch)
        return ve

    @pytest.fixture
    def process_exp(self, x_batch) -> VarianceExplodingDiffusionProcess:
        """VE with exponential schedule; used only by the exponential oracle test."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule="exponential", g_0=0.1, g_T=15.0,
        )
        ve.fit(x_batch)
        return ve

    def test_output_shape_matches_input(self, process, t_batch):
        """Output shape must match t."""
        result = process.diffusion(t_batch)

        assert result.shape == t_batch.shape

    def test_diffusion_at_t0_equals_g0(self, process, t_zero):
        """g(0) must equal g_0 exactly for both schedules.

        Linear:      g_0 + (g_T - g_0)·0/T = g_0
        Exponential: g_0·(g_T/g_0)^0 = g_0
        """
        result = process.diffusion(t_zero)

        expected = torch.full_like(t_zero, process.g_0)
        assert torch.allclose(result, expected)

    def test_diffusion_at_tT_equals_gT(self, process, t_one):
        """g(T) must equal g_T exactly for both schedules.

        Linear:      g_0 + (g_T - g_0)·T/T = g_T
        Exponential: g_0·(g_T/g_0)^1 = g_T
        """
        result = process.diffusion(t_one)

        expected = torch.full_like(t_one, process.g_T)
        assert torch.allclose(result, expected)

    def test_diffusion_is_strictly_positive(self, process, t_sweep):
        """g(t) must be strictly positive for all t ∈ (0.01, 0.99).

        Required for _cov to accumulate variance and multiply_inv_sigma to
        remain numerically stable.
        """
        result = process.diffusion(t_sweep)

        assert torch.all(result > 0), (
            f"g(t) is not strictly positive: min found {result.min().item():.6e}"
        )

    def test_diffusion_is_monotonically_increasing(self, process, t_sweep):
        """g(t) must be non-decreasing when g_T > g_0."""
        g_values = process.diffusion(t_sweep)
        diffs    = g_values[1:] - g_values[:-1]

        assert torch.all(diffs >= 0), (
            f"diffusion() is not non-decreasing: "
            f"min consecutive diff = {diffs.min().item():.6e}"
        )

    def test_linear_diffusion_matches_formula_at_known_points(self, process_linear):
        """diffusion(t) must match hand-computed values for the linear schedule.

        g(t) = g_0 + (g_T - g_0)·t/T = 0.1 + 14.9·t:

            t=0.00 →  0.1000
            t=0.25 →  3.8250
            t=0.50 →  7.5500
            t=0.75 → 11.2750
            t=1.00 → 15.0000

        atol=1e-5: 14.9 has a float32 rounding residual of ~3e-7 per multiply.
        """
        t        = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = torch.tensor([0.1000, 3.8250, 7.5500, 11.2750, 15.0000])

        result = process_linear.diffusion(t)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_exponential_diffusion_matches_formula_at_known_points(
        self, process_exp,
    ):
        """diffusion(t) must match hand-computed values for the exponential schedule.

        g(t) = g_0·(g_T/g_0)^(t/T) = 0.1·150^t:

            t=0.00 →  0.10000
            t=0.25 →  0.34996   (0.1·⁴√150)
            t=0.50 →  1.22474   (0.1·√150)
            t=0.75 →  4.28623   (0.1·150^0.75)
            t=1.00 → 15.00000

        atol=1e-4: torch.pow introduces larger rounding than a polynomial.
        """
        t        = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = torch.tensor([0.10000, 0.34996, 1.22474, 4.28623, 15.00000])

        result = process_exp.diffusion(t)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_exponential_diffusion_with_equal_g0_gT_is_constant(
        self, x_batch, t_batch,
    ):
        """When g_0 == g_T, the exponential schedule must return g(t) = g_0.

        g_0·(g_T/g_0)^(t/T) = g_0·1^(t/T) = g_0 for all t.

        Note: _cov() for this edge case is currently broken (returns zeros
        instead of g_0²·t). This test only covers diffusion(), which is correct.
        """
        g_const = 5.0
        ve = VarianceExplodingDiffusionProcess(
            g_schedule="exponential",
            g_0=g_const,
            g_T=g_const,
        )
        ve.fit(x_batch)

        result = ve.diffusion(t_batch)

        expected = torch.full_like(t_batch, g_const)
        assert torch.allclose(result, expected)

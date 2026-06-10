"""Tests for EulerMaruyama, RK4, and SDEReverseDiffusionProcess."""
from __future__ import annotations

import math

import pytest
import torch

from skfda.ml.generative.diffusion_process import (
    CirculantSymmetricMatrixDiffusionProcess,
    CustomDiffusionProcess,
    VariancePreservingDiffusionProcess,
    _eigenspace_to_fft,
    _fft_to_eigenspace,
)
from skfda.ml.generative.reverse_diffusion import (
    EulerMaruyamaIntegrator,
    ProbabilityFlowODEReverseProcess,
    RK4Integrator,
    SDEReverseDiffusionProcess,
)
from skfda.ml.generative.score_model import ScoreModel

from ._constants import DATA_DIM

_T_START = 0.0
_T_END = 1.0
_T_NOISY: float = 0.8
_T_CLEAN: float = 0.1   # must be > 1e-3 to skip Tweedie
_CIRCULANT_HALF_DIM: int = DATA_DIM // 2 + 1
_TWEEDIE_EPSILON_SEED: int = 99


# Process fixtures — module scope is safe: these objects hold no mutable state.

@pytest.fixture(scope="module")
def zero_process() -> CustomDiffusionProcess:
    """drift(x,t) = 0, diffusion(t) = zeros(N)."""
    def _drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(t.shape[0])  # scalar branch: (N,)

    return CustomDiffusionProcess(drift=_drift, diffusion=_diffusion)


@pytest.fixture(scope="module")
def unit_drift_process() -> CustomDiffusionProcess:
    """drift(x,t) = ones_like(x), diffusion(t) = zeros(N).

    Exact solution: x(T) = x_0 + T·ones, independent of n_steps.
    """
    def _drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(t.shape[0])

    return CustomDiffusionProcess(drift=_drift, diffusion=_diffusion)


@pytest.fixture(scope="module")
def unit_diffusion_process() -> CustomDiffusionProcess:
    """drift(x,t) = 0, diffusion(t) = ones(N) — standard Brownian motion."""
    def _drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def _diffusion(t: torch.Tensor) -> torch.Tensor:
        return torch.ones(t.shape[0])  # scalar branch: (N,)

    return CustomDiffusionProcess(drift=_drift, diffusion=_diffusion)


# Factory rather than fixture: EulerMaruyamaIntegrator holds a consumable generator;
# tests comparing two runs with the same seed must create two separate instances.

def _make_integrator(
    n_steps: int = 50,
    seed: int | None = None,
    device: str = "cpu",
) -> EulerMaruyamaIntegrator:
    """Return a fresh EulerMaruyamaIntegrator."""
    return EulerMaruyamaIntegrator(
        n_steps=n_steps, seed=seed, device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Score model classes
# ─────────────────────────────────────────────────────────────────────────────


class _ZeroScoreModel(ScoreModel):
    """Returns zeros_like(x) for every (x, t)."""

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """f(x, t) = 0."""
        return torch.zeros_like(x)


class _ConstantOnesScoreModel(ScoreModel):
    """Returns ones_like(x) for every (x, t).

    Non-zero and time-independent: makes the G·Gᵀ·score term numerically
    significant at every step, enabling sign and factor correctness tests.
    """

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """f(x, t) = 1."""
        return torch.ones_like(x)


class _PerfectConditionalScoreModel(ScoreModel):
    """Exact conditional score for the VP process given a known x_0.

    For x_t = μ_t·x_0 + σ_t·ε, the exact conditional score is:
        ∇ log p_t(x_t | x_0) = −(x_t − μ_t·x_0) / σ_t²

    Tweedie's identity then recovers x_0 exactly:
        x̂_0 = (x_t + σ_t²·score) / μ_t = x_0
    """

    def __init__(
        self,
        x_0: torch.Tensor,
        vp: VariancePreservingDiffusionProcess,
    ) -> None:
        """Store x_0 and the fitted VP process."""
        super().__init__()
        self.register_buffer("x_0", x_0)
        self.vp = vp  # not an nn.Module; stored as a plain attribute

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns −(x − μ_t·x_0) / σ_t², shape (N, M)."""
        mu_t_x0 = self.vp.mean_cond(self.x_0, t)
        sigma2_t = self.vp._cov(t)
        return -(x - mu_t_x0) / sigma2_t.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vp_process(x_batch: torch.Tensor) -> VariancePreservingDiffusionProcess:
    """Fitted VP with linear schedule; β(t), μ_t, σ_t² have closed-form expressions."""
    vp = VariancePreservingDiffusionProcess(
        beta_schedule="linear", beta_min=0.1, beta_max=20.0,
    )
    return vp.fit(x_batch)


@pytest.fixture(scope="module")
def circulant_process(
    x_batch: torch.Tensor,
) -> CirculantSymmetricMatrixDiffusionProcess:
    """Fitted CirculantSymmetricMatrixDiffusionProcess with constant Fourier eigenvalues.

    Drift eigenvalues: −0.5, diffusion eigenvalues: 1.0.
    Shape: (N, _CIRCULANT_HALF_DIM=9) for t of shape (N,).
    """
    def _drift_eigs(t: torch.Tensor) -> torch.Tensor:
        return torch.full((t.shape[0], _CIRCULANT_HALF_DIM), -0.5)

    def _diff_eigs(t: torch.Tensor) -> torch.Tensor:
        return torch.full((t.shape[0], _CIRCULANT_HALF_DIM), 1.0)

    process = CirculantSymmetricMatrixDiffusionProcess(
        drift_term=_drift_eigs,
        diffusion_term=_diff_eigs,
        fourier_drift=True,
        fourier_diffusion=True,
    )
    return process.fit(x_batch)


def _make_em_integrator(
    n_steps: int = 50,
    seed: int | None = None,
) -> EulerMaruyamaIntegrator:
    """Return a fresh EulerMaruyamaIntegrator."""
    return EulerMaruyamaIntegrator(n_steps=n_steps, seed=seed)


def _make_sde_reverse(
    n_steps: int = 50,
    seed: int | None = None,
) -> SDEReverseDiffusionProcess:
    """Return a fresh SDEReverseDiffusionProcess backed by a fresh integrator."""
    return SDEReverseDiffusionProcess(_make_em_integrator(n_steps, seed))


# ─────────────────────────────────────────────────────────────────────────────


class TestEulerMaruyamaIntegrator:
    """Tests for EulerMaruyamaIntegrator."""

    def test_output_shape_matches_input_shape(self, zero_process, x_batch):
        """Output shape must equal x_0's shape: (N, M) → (N, M)."""
        integrator = _make_integrator(n_steps=10)
        result = integrator(zero_process, x_batch, _T_START, _T_END)
        assert result.shape == x_batch.shape

    def test_zero_drift_zero_diffusion_returns_input_unchanged(
        self, zero_process, x_batch,
    ):
        """With f=0 and g=0, every Euler step contributes exactly zero.

        allclose (not torch.equal): IEEE-754 additions of exact zero are exact
        in practice but allclose tolerates ±0.0 sign differences that equal would not.
        """
        integrator = _make_integrator(n_steps=50)
        result = integrator(zero_process, x_batch, _T_START, _T_END)
        assert torch.allclose(result, x_batch)

    def test_constant_drift_zero_diffusion_produces_exact_euler_result(
        self, unit_drift_process, x_batch,
    ):
        """With constant drift c=1 and zero diffusion, x(T) must equal x_0 + T·ones.

        Total displacement telescopes to c·(T−t₀), independent of step count.

        atol=1e-4: float32 cannot represent 1/100 exactly; 100 accumulated
        additions introduce ~n_steps·ε_float32 ≈ 1.2e-5.
        """
        T = _T_END - _T_START
        integrator = _make_integrator(n_steps=100)
        result = integrator(unit_drift_process, x_batch, _T_START, _T_END)
        expected = x_batch + T * torch.ones_like(x_batch)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_same_seed_gives_identical_results(
        self, unit_diffusion_process, x_batch,
    ):
        """Two integrators with the same seed must give bit-identical output.

        torch.equal (not allclose): identical seeds must produce exactly the same
        floating-point Brownian increments. Two separate instances are required:
        calling one instance twice advances its generator state.
        """
        integrator_a = _make_integrator(n_steps=50, seed=13)
        integrator_b = _make_integrator(n_steps=50, seed=13)
        result_a = integrator_a(unit_diffusion_process, x_batch, _T_START, _T_END)
        result_b = integrator_b(unit_diffusion_process, x_batch, _T_START, _T_END)
        assert torch.equal(result_a, result_b)

    def test_different_seeds_give_different_results(
        self, unit_diffusion_process, x_batch,
    ):
        """Two integrators with different seeds must produce different output.

        The probability of two independent Brownian paths being bit-identical
        is negligible for any continuous distribution.
        """
        integrator_a = _make_integrator(n_steps=50, seed=13)
        integrator_b = _make_integrator(n_steps=50, seed=14)
        result_a = integrator_a(unit_diffusion_process, x_batch, _T_START, _T_END)
        result_b = integrator_b(unit_diffusion_process, x_batch, _T_START, _T_END)
        assert not torch.equal(result_a, result_b)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available; device mismatch test requires two distinct devices.",
    )
    def test_device_mismatch_raises_value_error(self, zero_process, x_batch):
        """Passing x_0 on a different device than the integrator must raise ValueError.

        Validation fires at __call__ entry before any computation begins.
        """
        cpu_integrator = _make_integrator(n_steps=10, device="cpu")
        x_cuda = x_batch.cuda()
        with pytest.raises(ValueError):
            cpu_integrator(zero_process, x_cuda, _T_START, _T_END)

    def test_backward_integration_direction_inverts_constant_drift(
        self, unit_drift_process, x_batch,
    ):
        """Forward then backward integration with constant drift must recover x_0.

        backward (T → 0): linspace(T, 0, n_steps+1) gives negative dt.

            forward  (0 → T): x_T        = x_0 + 1.0·ones
            backward (T → 0): x_recovered = x_T − 1.0·ones = x_0

        A single instance is reused: diffusion=0 so stochastic draws are
        multiplied by zero and the trajectory is independent of generator state.

        atol=1e-4: 200 accumulated float32 additions (100 forward + 100 backward).
        """
        integrator = _make_integrator(n_steps=100)
        x_T = integrator(unit_drift_process, x_batch, _T_START, _T_END)
        x_recovered = integrator(unit_drift_process, x_T, _T_END, _T_START)
        assert torch.allclose(x_recovered, x_batch, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────


def _zero_ode(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """f(t, x) = 0."""
    return torch.zeros_like(x)


def _unit_drift_ode(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """f(t, x) = ones_like(x). Exact solution: x(T) = x_0 + T·ones."""
    return torch.ones_like(x)


def _exp_decay_ode(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """f(t, x) = -x. Exact solution: x(T) = x_0·e^{-T}."""
    return -x


# ─────────────────────────────────────────────────────────────────────────────


class TestRK4Integrator:
    """Tests for RK4Integrator.

    RK4 takes f(t, x) directly rather than a DiffusionProcess; tests use the
    plain ODE helpers above. No random state → no reproducibility or device tests;
    determinism replaces reproducibility (two calls on the same instance must agree).
    """

    def test_output_shape_matches_input_shape(self, x_batch):
        """Output shape must equal x_0's shape: (N, M) → (N, M)."""
        integrator = RK4Integrator(n_steps=10)
        result = integrator(_zero_ode, x_batch, _T_START, _T_END)
        assert result.shape == x_batch.shape

    @pytest.mark.parametrize("n_steps", [10, 100], ids=["n10", "n100"])
    def test_constant_drift_produces_exact_result_for_any_step_count(
        self, x_batch, n_steps,
    ):
        """With f(t,x)=ones, x(T) must equal x_0 + T·ones for any n_steps.

        For constant f all four RK4 slopes equal ones, so each step reduces to
        x_{n+1} = x_n + dt·ones and the total telescopes to T·ones.

        Parametrising over n_steps=10 and 100: an off-by-one in the loop bound
        would produce T·(1 ± 1/n_steps), which fails at n_steps=10.

        atol=1e-4: float32 rounding ≤ n_steps·ε_float32·|x| ≈ 100·1.2e-7·2 ≈ 2.4e-5.
        """
        T = _T_END - _T_START
        integrator = RK4Integrator(n_steps=n_steps)
        result = integrator(_unit_drift_ode, x_batch, _T_START, _T_END)
        expected = x_batch + T * torch.ones_like(x_batch)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_exponential_decay_ode_matches_analytical_solution(self, x_batch):
        """RK4 applied to dx/dt = -x must match x_0·e^{-T} to float64 precision.

        float64 is required: in float32 accumulated rounding (~6e-5) dwarfs the
        RK4 truncation error (~1e-11), so only float64 tests 4th-order accuracy.

        atol=1e-9: RK4 global error for n_steps=500 is ≈ (1/500)^4·T·|x_0| ≈ 5e-11.
        """
        x_f64 = x_batch.double()
        integrator = RK4Integrator(n_steps=500)
        result = integrator(_exp_decay_ode, x_f64, _T_START, _T_END)
        expected = x_f64 * math.exp(-(_T_END - _T_START))
        assert torch.allclose(result, expected, atol=1e-9)

    def test_result_is_deterministic_across_calls(self, x_batch):
        """Two calls with identical inputs must give bit-identical output.

        torch.equal (not allclose): any accidentally introduced non-deterministic
        call (e.g. torch.randn) would differ. A single instance is reused since
        RK4 has no generator to exhaust.
        """
        integrator = RK4Integrator(n_steps=50)
        result_a = integrator(_exp_decay_ode, x_batch, _T_START, _T_END)
        result_b = integrator(_exp_decay_ode, x_batch, _T_START, _T_END)
        assert torch.equal(result_a, result_b)

    def test_backward_integration_inverts_forward_for_constant_drift(
        self, x_batch,
    ):
        """Forward then backward integration with constant drift must recover x_0.

        backward (T → 0): negative dt gives x_{n+1} = x_n − dt·ones, so the
        round-trip cancels: forward gives x_0 + T·ones; backward gives x_0.

        atol=1e-4: 200 accumulated float32 additions (100 forward + 100 backward).
        """
        integrator = RK4Integrator(n_steps=100)
        x_T = integrator(_unit_drift_ode, x_batch, _T_START, _T_END)
        x_recovered = integrator(_unit_drift_ode, x_T, _T_END, _T_START)
        assert torch.allclose(x_recovered, x_batch, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────


class TestSDEReverseDiffusionProcess:
    """Tests for SDEReverseDiffusionProcess."""

    def test_output_shape_matches_input_shape(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Output shape must equal x_t's shape: (N, M) → (N, M)."""
        zero_score = _ZeroScoreModel()
        sde_reverse = _make_sde_reverse(n_steps=10)
        result = sde_reverse.reverse(
            vp_process, zero_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )
        assert result.shape == x_batch.shape

    def test_backward_drift_subtracts_score_term_not_adds_it(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """SDE backward drift must be f(x,t) − G·Gᵀ·score, not f(x,t) + G·Gᵀ·score.

        A sign error pushes samples in the wrong direction at every step without
        raising an exception. The oracle integrates the wrong (+ sign) drift using
        the same EM seed, so both paths differ only in the sign of G·Gᵀ·score.
        """
        constant_score = _ConstantOnesScoreModel()

        sde_reverse = _make_sde_reverse(n_steps=50, seed=13)
        sde_result = sde_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        def _wrong_drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            score = constant_score(x, t, None)
            return vp_process.drift(x, t) + vp_process.diffusion_gram_times_v(score, t)

        wrong_process = CustomDiffusionProcess(
            drift=_wrong_drift, diffusion=vp_process.diffusion,
        )
        wrong_em = _make_em_integrator(n_steps=50, seed=13)
        wrong_result = wrong_em(wrong_process, x_batch, _T_NOISY, _T_CLEAN)

        assert not torch.allclose(sde_result, wrong_result)

    def test_backward_drift_uses_full_diffusion_gram_not_half(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """SDE backward drift must use G·Gᵀ·score, not 0.5·G·Gᵀ·score.

        SDE uses the full Gram; probability flow ODE uses half. A copy-paste error
        introducing 0.5 produces a numerically valid but mathematically wrong trajectory.
        """
        constant_score = _ConstantOnesScoreModel()

        sde_reverse = _make_sde_reverse(n_steps=50, seed=13)
        sde_result = sde_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        def _half_factor_drift(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            score = constant_score(x, t, None)
            return vp_process.drift(x, t) - 0.5 * vp_process.diffusion_gram_times_v(score, t)

        half_process = CustomDiffusionProcess(
            drift=_half_factor_drift, diffusion=vp_process.diffusion,
        )
        half_em = _make_em_integrator(n_steps=50, seed=13)
        half_result = half_em(half_process, x_batch, _T_NOISY, _T_CLEAN)

        assert not torch.allclose(sde_result, half_result)

    def test_tweedie_step_is_applied_when_t0_equals_zero(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Calling reverse() with t_0=0 must apply Tweedie's formula; t_0=1e-3 must not.

        Both calls share the same integration window (both stop at t_0_safe=1e-3)
        and the same EM seed, so the integration phase is bit-identical; only the
        Tweedie branch differs. At t=1e-3 the correction shifts each element by
        ~σ²_{1e-3}/μ_{1e-3} ≈ 1.1e-4, well above allclose's default atol=1e-8.
        """
        constant_score = _ConstantOnesScoreModel()

        sde_with_tweedie = _make_sde_reverse(n_steps=50, seed=13)
        result_with_tweedie = sde_with_tweedie.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=0.0,
        )

        sde_without_tweedie = _make_sde_reverse(n_steps=50, seed=13)
        result_without_tweedie = sde_without_tweedie.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=1e-3,
        )

        assert not torch.allclose(result_with_tweedie, result_without_tweedie)

    def test_tweedie_formula_recovers_x0_with_perfect_conditional_score(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Tweedie's formula must exactly recover x_0 given the perfect conditional score.

        Tweedie's identity for the VP process:
            x̂_0 = (x_t + Σ(t)·score) / μ_t

        With score = −(x_t − μ_t·x_0)/σ_t² this reduces to x_0 exactly.

        t_1 = t_0_safe = 1e-3 makes dt=0 throughout so only the Tweedie step runs.

        atol=1e-4: float32 rounding in the σ_t²·score/μ_t chain at t=1e-3.
        """
        N, M = x_batch.shape
        t_1e3 = torch.full((N,), 1e-3)

        gen = torch.Generator()
        gen.manual_seed(_TWEEDIE_EPSILON_SEED)
        epsilon = torch.randn(N, M, generator=gen)
        x_t = (
            vp_process.mean_cond(x_batch, t_1e3)
            + vp_process.multiply_sigma(epsilon, t_1e3)
        )

        perfect_score = _PerfectConditionalScoreModel(x_0=x_batch, vp=vp_process)

        # t_1 = t_0_safe = 1e-3 makes dt=0 throughout; n_steps is irrelevant
        sde_reverse = _make_sde_reverse(n_steps=1, seed=0)
        result = sde_reverse.reverse(vp_process, perfect_score, x_t, t_1=1e-3, t_0=0.0)

        assert torch.allclose(result, x_batch, atol=1e-4)

    def test_zero_score_model_makes_reverse_equal_to_pure_backward_drift(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """With score=0, the reverse trajectory must equal pure backward drift integration.

        backward_drift reduces to f(x, t) − G·Gᵀ·zeros = f(x, t).

        allclose (not torch.equal): the extra G·Gᵀ·zeros arithmetic path should
        give bit-identical results (β_t·0 = 0 exactly in IEEE-754), but allclose
        is appropriate when two numerically equivalent derivations are compared.
        """
        zero_score = _ZeroScoreModel()

        sde_reverse = _make_sde_reverse(n_steps=50, seed=13)
        sde_result = sde_reverse.reverse(
            vp_process, zero_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        manual_process = CustomDiffusionProcess(
            drift=vp_process.drift, diffusion=vp_process.diffusion,
        )
        manual_em = _make_em_integrator(n_steps=50, seed=13)
        manual_result = manual_em(manual_process, x_batch, _T_NOISY, _T_CLEAN)

        assert torch.allclose(sde_result, manual_result)

    def test_circulant_process_result_matches_manual_fft_eigenspace_path(
        self,
        circulant_process: CirculantSymmetricMatrixDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Circulant dispatch must produce the same result as the manual eigenspace path.

        _reverse_circulant implements:
          1. x_diag = _fft_to_eigenspace(x_t)
          2. score_diag(x_diag, t) = _fft_to_eigenspace(score(_eigenspace_to_fft(x_diag), t))
          3. result_diag = self.reverse(diagonal_process, score_diag, x_diag, t_1, t_0)
          4. return _eigenspace_to_fft(result_diag)

        This test replicates steps 1–4 manually. Failures indicate broken dispatch
        registration or wrong FFT projection order in the score wrapping.

        atol=1e-5: float32 rounding from repeated FFT evaluations along both paths.
        """
        constant_score = _ConstantOnesScoreModel()

        dispatched_sde = _make_sde_reverse(n_steps=20, seed=13)
        dispatched_result = dispatched_sde.reverse(
            circulant_process, constant_score, x_batch,
            t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        x_t_diag = _fft_to_eigenspace(x_batch)

        def _score_diag(
            x_diag: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor | None = None,
        ) -> torch.Tensor:
            score = constant_score(_eigenspace_to_fft(x_diag), t, y)
            return _fft_to_eigenspace(score)

        manual_sde = _make_sde_reverse(n_steps=20, seed=13)
        result_diag = manual_sde.reverse(
            circulant_process.diagonal_process, _score_diag, x_t_diag,
            t_1=_T_NOISY, t_0=_T_CLEAN,
        )
        manual_result = _eigenspace_to_fft(result_diag)

        assert torch.allclose(dispatched_result, manual_result, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────


def _make_ode_reverse(n_steps: int = 50) -> ProbabilityFlowODEReverseProcess:
    """Return a fresh ProbabilityFlowODEReverseProcess backed by a fresh RK4Integrator.

    No seed parameter: RK4 is deterministic — a fresh instance with the same
    n_steps and identical inputs always produces the same output.
    """
    return ProbabilityFlowODEReverseProcess(RK4Integrator(n_steps=n_steps))


class TestProbabilityFlowODEReverseProcess:
    """Tests for ProbabilityFlowODEReverseProcess."""

    def test_output_shape_matches_input_shape(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Output shape must equal x_t's shape: (N, M) → (N, M)."""
        zero_score = _ZeroScoreModel()
        ode_reverse = _make_ode_reverse(n_steps=10)
        result = ode_reverse.reverse(
            vp_process, zero_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )
        assert result.shape == x_batch.shape

    def test_result_is_deterministic_across_calls(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Two calls with identical inputs on the same instance must give bit-identical output.

        torch.equal (not allclose): any accidentally introduced non-deterministic
        call (e.g. torch.randn) would differ across calls. A single instance is reused
        since RK4 has no generator state to exhaust.
        """
        constant_score = _ConstantOnesScoreModel()
        ode_reverse = _make_ode_reverse(n_steps=50)
        result_a = ode_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )
        result_b = ode_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )
        assert torch.equal(result_a, result_b)

    def test_backward_drift_subtracts_score_term_not_adds_it(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """ODE backward drift must be f(x,t) − 0.5·G·Gᵀ·score, not f(x,t) + 0.5·G·Gᵀ·score.

        A sign error pushes samples in the wrong direction at every step without
        raising an exception. The oracle drives a raw RK4Integrator with the wrong
        (+ sign) drift; both paths diverge only where the sign of G·Gᵀ·score differs.
        Note that oracle drift uses (t, x) argument order, matching RK4Integrator's
        convention, while vp_process.drift takes (x, t).
        """
        constant_score = _ConstantOnesScoreModel()

        ode_reverse = _make_ode_reverse(n_steps=50)
        ode_result = ode_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        def _wrong_drift(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            score = constant_score(x, t, None)
            return vp_process.drift(x, t) + 0.5 * vp_process.diffusion_gram_times_v(score, t)

        wrong_rk4 = RK4Integrator(n_steps=50)
        wrong_result = wrong_rk4(_wrong_drift, x_batch, _T_NOISY, _T_CLEAN)

        assert not torch.allclose(ode_result, wrong_result)

    def test_backward_drift_uses_half_diffusion_gram_not_full(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """ODE backward drift must use 0.5·G·Gᵀ·score, not 1.0·G·Gᵀ·score.

        SDE uses the full Gram; probability flow ODE uses half. A copy-paste of the
        SDE implementation carrying over the coefficient of 1.0 would pass the sign
        test above but fail here, since the factor difference is numerically significant
        with _ConstantOnesScoreModel active at every step.
        """
        constant_score = _ConstantOnesScoreModel()

        ode_reverse = _make_ode_reverse(n_steps=50)
        ode_result = ode_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        def _full_factor_drift(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            score = constant_score(x, t, None)
            return vp_process.drift(x, t) - 1.0 * vp_process.diffusion_gram_times_v(score, t)

        full_rk4 = RK4Integrator(n_steps=50)
        full_result = full_rk4(_full_factor_drift, x_batch, _T_NOISY, _T_CLEAN)

        assert not torch.allclose(ode_result, full_result)

    def test_zero_score_model_makes_reverse_equal_to_pure_backward_drift(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """With score=0, the reverse trajectory must equal pure backward drift integration.

        backward_drift reduces to f(x, t) − 0.5·G·Gᵀ·zeros = f(x, t).

        allclose (not torch.equal): the extra 0.5·G·Gᵀ·zeros arithmetic should
        be bit-identical (β_t·0 = 0 exactly in IEEE-754), but allclose is appropriate
        when two numerically equivalent derivations are compared.
        """
        zero_score = _ZeroScoreModel()

        ode_reverse = _make_ode_reverse(n_steps=50)
        ode_result = ode_reverse.reverse(
            vp_process, zero_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        def _pure_drift(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return vp_process.drift(x, t)

        manual_rk4 = RK4Integrator(n_steps=50)
        manual_result = manual_rk4(_pure_drift, x_batch, _T_NOISY, _T_CLEAN)

        assert torch.allclose(ode_result, manual_result)

    def test_ode_result_differs_from_sde_result(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """ODE and SDE reverse processes must give different outputs with a non-zero score.

        ODE uses 0.5·G·Gᵀ·score deterministically; SDE uses 1.0·G·Gᵀ·score plus
        Brownian noise. Identical outputs would indicate one silently delegates to
        the other, or that a factor-of-two error in one cancels the noise term by
        coincidence.
        """
        constant_score = _ConstantOnesScoreModel()

        ode_reverse = _make_ode_reverse(n_steps=50)
        ode_result = ode_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        sde_reverse = _make_sde_reverse(n_steps=50, seed=42)
        sde_result = sde_reverse.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        assert not torch.allclose(ode_result, sde_result)

    def test_circulant_process_result_matches_manual_fft_eigenspace_path(
        self,
        circulant_process: CirculantSymmetricMatrixDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Circulant dispatch must produce the same result as the manual eigenspace path.

        _reverse_circulant implements:
          1. x_diag = _fft_to_eigenspace(x_t)
          2. score_diag(x_diag, t) = _fft_to_eigenspace(score(_eigenspace_to_fft(x_diag), t))
          3. result_diag = self.reverse(diagonal_process, score_diag, x_diag, t_1, t_0)
          4. return _eigenspace_to_fft(result_diag)

        This test replicates steps 1–4 manually. Failures indicate broken dispatch
        registration or wrong FFT projection order in the score wrapping.

        atol=1e-5: float32 rounding from repeated FFT evaluations along both paths.
        """
        constant_score = _ConstantOnesScoreModel()

        dispatched_ode = _make_ode_reverse(n_steps=20)
        dispatched_result = dispatched_ode.reverse(
            circulant_process, constant_score, x_batch,
            t_1=_T_NOISY, t_0=_T_CLEAN,
        )

        x_t_diag = _fft_to_eigenspace(x_batch)

        def _score_diag(
            x_diag: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor | None = None,
        ) -> torch.Tensor:
            score = constant_score(_eigenspace_to_fft(x_diag), t, y)
            return _fft_to_eigenspace(score)

        manual_ode = _make_ode_reverse(n_steps=20)
        result_diag = manual_ode.reverse(
            circulant_process.diagonal_process, _score_diag, x_t_diag,
            t_1=_T_NOISY, t_0=_T_CLEAN,
        )
        manual_result = _eigenspace_to_fft(result_diag)

        assert torch.allclose(dispatched_result, manual_result, atol=1e-5)

    def test_tweedie_step_is_applied_when_t0_equals_zero(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Calling reverse() with t_0=0 must apply Tweedie's formula; t_0=1e-3 must not.

        Both calls share the same integration window (both stop at t_0_safe=1e-3)
        and use RK4 (deterministic), so the integration phases are bit-identical;
        only the Tweedie branch differs. At t=1e-3 the correction shifts each element
        by ~σ²_{1e-3}/μ_{1e-3} ≈ 1.1e-4, well above allclose's default atol=1e-8.
        """
        constant_score = _ConstantOnesScoreModel()

        ode_with_tweedie = _make_ode_reverse(n_steps=50)
        result_with_tweedie = ode_with_tweedie.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=0.0,
        )

        ode_without_tweedie = _make_ode_reverse(n_steps=50)
        result_without_tweedie = ode_without_tweedie.reverse(
            vp_process, constant_score, x_batch, t_1=_T_NOISY, t_0=1e-3,
        )

        assert not torch.allclose(result_with_tweedie, result_without_tweedie)

    def test_tweedie_formula_recovers_x0_with_perfect_conditional_score(
        self,
        vp_process: VariancePreservingDiffusionProcess,
        x_batch: torch.Tensor,
    ) -> None:
        """Tweedie's formula must exactly recover x_0 given the perfect conditional score.

        Tweedie's identity for the VP process:
            x̂_0 = (x_t + Σ(t)·score) / μ_t

        With score = −(x_t − μ_t·x_0)/σ_t² this reduces to x_0 exactly.

        Setting t_1 = t_0_safe = 1e-3 collapses the integration window to zero
        (dt=0 at every RK4 step), so only the Tweedie step runs. n_steps=1 since
        step count is irrelevant when dt=0.

        atol=1e-4: float32 rounding in the σ_t²·score/μ_t chain at t=1e-3.
        """
        N, M = x_batch.shape
        t_1e3 = torch.full((N,), 1e-3)

        gen = torch.Generator()
        gen.manual_seed(_TWEEDIE_EPSILON_SEED)
        epsilon = torch.randn(N, M, generator=gen)
        x_t = (
            vp_process.mean_cond(x_batch, t_1e3)
            + vp_process.multiply_sigma(epsilon, t_1e3)
        )

        perfect_score = _PerfectConditionalScoreModel(x_0=x_batch, vp=vp_process)

        ode_reverse = _make_ode_reverse(n_steps=1)
        result = ode_reverse.reverse(vp_process, perfect_score, x_t, t_1=1e-3, t_0=0.0)

        assert torch.allclose(result, x_batch, atol=1e-4)
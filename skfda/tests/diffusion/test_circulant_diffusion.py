from __future__ import annotations

import warnings

import pytest
import torch

from skfda.ml.generative._diffusion_process import (
    CirculantSymmetricMatrixDiffusionProcess,
    DiagonalDiffusionProcess,
    ForwardDiffusionProcess,
    _eigenspace_to_fft,
    _fft_to_eigenspace,
    _get_cosine_basis,
    _duplicate_symmetric_eigenvalues,
    _duplicate_symmetric_row_and_get_eigenvalues,
)

from ._constants import BATCH_SIZE, CUSTOM_DIM, DATA_DIM, SEED
from .diffusion_test_mixins import (
    ForwardDiffusionCheckpointTests,
    ForwardDiffusionFitTests,
    ForwardDiffusionOperatorTests,
    ForwardDiffusionSampleLimitTests,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level fixtures and factory
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def circulant_callables():
    """D=-0.5, g=1.0; closed-form _cov(t) = (1 - exp(-t)) · I_M."""
    half_dim = DATA_DIM // 2 + 1  # 9 for DATA_DIM=16
    drift_term     = lambda t: torch.full((t.shape[0], half_dim), -0.5, device=t.device)
    diffusion_term = lambda t: torch.full((t.shape[0], half_dim),  1.0, device=t.device)
    return drift_term, diffusion_term


@pytest.fixture(scope="module")
def vp_eigenvalue_callables():
    """VP-equivalent callables; identity μ_k(t)² + sigma_k(t)² = 1 per eigenmode.

    At T=1 with beta_min=0.1, beta_max=10.0:
        ∫₀¹ d_t dt = -2.525, μ_k(1) ≈ 0.080, sigma_k(1) ≈ 0.997
    Limit distribution ≈ N(0, 0.994·I_M); systematic deviation from N(0,I) is 0.003,
    within atol=0.05.
    """
    beta_min, beta_max = 0.1, 10.0
    half_dim = DATA_DIM // 2 + 1  # 9 for DATA_DIM=16

    drift_term = lambda t: (
        (-0.5 * (beta_min + t * (beta_max - beta_min)))
        .unsqueeze(1)
        .expand(-1, half_dim)
    )
    diffusion_term = lambda t: (
        torch.sqrt(beta_min + t * (beta_max - beta_min))
        .unsqueeze(1)
        .expand(-1, half_dim)
    )
    return drift_term, diffusion_term


@pytest.fixture(scope="module")
def constant_row_callables():
    """Row-convention callables representing the same operator as circulant_callables.

    c_0 = -0.5 (drift), c_0 = 1.0 (diffusion), all other entries zero.
    FFT([c_0, 0, …, 0])[k] = c_0 for all k — same eigenvalue spectrum as circulant_callables.
    """
    half_dim = DATA_DIM // 2 + 1  # 9 for DATA_DIM=16

    drift_term = lambda t: torch.cat(
        [
            torch.full((t.shape[0], 1), -0.5),
            torch.zeros(t.shape[0], half_dim - 1),
        ],
        dim=1,
    )
    diffusion_term = lambda t: torch.cat(
        [
            torch.full((t.shape[0], 1), 1.0),
            torch.zeros(t.shape[0], half_dim - 1),
        ],
        dim=1,
    )
    return drift_term, diffusion_term


@pytest.fixture(scope="module")
def p_fourier(circulant_callables):
    """CirculantSymmetricMatrixDiffusionProcess; fourier_drift=True, fourier_diffusion=True."""
    torch.manual_seed(SEED)
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    return _make_fitted_constant_process(x, circulant_callables)


@pytest.fixture(scope="module")
def p_row(constant_row_callables):
    """CirculantSymmetricMatrixDiffusionProcess; fourier_drift=False, fourier_diffusion=False.

    Same operator as p_fourier: FFT([c_0, 0, …, 0]).real = c_0·ones(M).
    fourier_diffusion=False (not True) exercises _duplicate_symmetric_row_and_get_eigenvalues
    for the diffusion callable specifically, making multiply_sigma and multiply_cov
    tests meaningful as independent path checks.
    """
    torch.manual_seed(SEED)
    x = torch.randn(BATCH_SIZE, DATA_DIM)
    drift_term, diffusion_term = constant_row_callables
    p = CirculantSymmetricMatrixDiffusionProcess(
        drift_term=drift_term,
        diffusion_term=diffusion_term,
        fourier_drift=False,
        fourier_diffusion=False,
    )
    p.fit(x)
    return p


def _make_fitted_constant_process(
    x_batch: torch.Tensor,
    circulant_callables: tuple,
) -> CirculantSymmetricMatrixDiffusionProcess:
    """Construct and fit a CirculantSymmetricMatrixDiffusionProcess with D=-0.5, g=1.0."""
    drift_term, diffusion_term = circulant_callables
    p = CirculantSymmetricMatrixDiffusionProcess(
        drift_term=drift_term,
        diffusion_term=diffusion_term,
        fourier_drift=True,
        fourier_diffusion=True,
    )
    p.fit(x_batch)
    return p


def _circulant_drift_broadcast(t: torch.Tensor) -> torch.Tensor:
    """Constant drift eigenvalue -0.5, broadcast shape (N, 1); picklable for any M."""
    return torch.full((t.shape[0], 1), -0.5, dtype=t.dtype, device=t.device)


def _circulant_diffusion_broadcast(t: torch.Tensor) -> torch.Tensor:
    """Constant diffusion eigenvalue 1.0, broadcast shape (N, 1); picklable for any M."""
    return torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantFit
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantFit(ForwardDiffusionFitTests, ForwardDiffusionCheckpointTests):
    """Tests for the fit / checkpoint contract of CirculantSymmetricMatrixDiffusionProcess.

    Extends the base contract:
    (1) fit() builds an internal DiagonalDiffusionProcess in eigenspace (self.diagonal_process).
    (2) _get_fit_state() stores 'device' alongside the inherited 'M'.
    (3) to_checkpoint() warns and replaces non-picklable callables with None.

    _restore_fit_state() re-runs fit() with a dummy zero tensor, so round-trip tests
    use allclose rather than torch.equal — precomputed grids are rebuilt via an
    independent FFT path.
    """

    @pytest.fixture
    def make_process(self):
        """Factory with picklable broadcast (N, 1) callables; fourier_drift=True, fourier_diffusion=True."""
        def factory() -> CirculantSymmetricMatrixDiffusionProcess:
            return CirculantSymmetricMatrixDiffusionProcess(
                drift_term=_circulant_drift_broadcast,
                diffusion_term=_circulant_diffusion_broadcast,
                fourier_drift=True,
                fourier_diffusion=True,
            )
        return factory

    @pytest.fixture
    def make_process_with_lambda(self):
        """Factory with lambda callables — not picklable; used only by picklability warning tests."""
        half_dim = CUSTOM_DIM // 2 + 1  # 9

        def factory() -> CirculantSymmetricMatrixDiffusionProcess:
            return CirculantSymmetricMatrixDiffusionProcess(
                drift_term=lambda t: torch.full((t.shape[0], half_dim), -0.5),
                diffusion_term=lambda t: torch.full((t.shape[0], half_dim), 1.0),
                fourier_drift=True,
                fourier_diffusion=True,
            )
        return factory

    @pytest.fixture
    def fitted_instance(self, x_batch, circulant_callables):
        """CirculantSymmetricMatrixDiffusionProcess fitted on x_batch."""
        drift_term, diffusion_term = circulant_callables
        instance = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            fourier_drift=True,
            fourier_diffusion=True,
        )
        instance.fit(x_batch)
        return instance

    def test_get_fit_state_contains_device_key(self, fitted_instance):
        """_get_fit_state() must contain 'device'.

        VP and VE do not store device; this pins the Circulant-specific extension.
        Dropping 'device' would make _restore_fit_state() raise ValueError on any load.
        """
        state = fitted_instance._get_fit_state()

        assert "device" in state

    def test_fit_creates_fitted_diagonal_process(self, fitted_instance):
        """fit() must create a fitted DiagonalDiffusionProcess at self.diagonal_process.

        All operator arithmetic is delegated to diagonal_process in eigenspace.
        M == DATA_DIM confirms it was fitted on correct-dimension data.
        """
        assert isinstance(fitted_instance.diagonal_process_, DiagonalDiffusionProcess)
        assert fitted_instance.diagonal_process_.M_ == DATA_DIM

    def test_fit_q_mat_has_correct_shape(self, fitted_instance):
        """After fit(), q_mat_ must have shape (M, M).

        An incorrect shape — e.g., (M//2+1, M) from a missing duplication step —
        would corrupt every batched matrix product through silent broadcasting.
        """
        assert fitted_instance.q_mat_.shape == (DATA_DIM, DATA_DIM)

    def test_fit_q_mat_is_orthogonal(self, fitted_instance):
        """After fit(), q_mat_ must satisfy Q @ Q.T ≈ I and Q.T @ Q ≈ I.

        Both products checked independently so a one-sided failure is caught.
        atol=1e-5 accommodates float32 rounding; a normalization bug produces
        deviations of order 1 - 1/M ≈ 0.94.
        """
        Q = fitted_instance.q_mat_
        eye = torch.eye(DATA_DIM, device=Q.device)

        assert torch.allclose(Q @ Q.T, eye, atol=1e-5), (
            "Q @ Q.T must equal I_M; orthogonality violated in the left product"
        )
        assert torch.allclose(Q.T @ Q, eye, atol=1e-5), (
            "Q.T @ Q must equal I_M; orthogonality violated in the right product"
        )

    def test_to_checkpoint_picklable_callables_no_warning(self, make_process):
        """to_checkpoint() must not warn when drift_term and diffusion_term are picklable."""
        instance = make_process()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            checkpoint = instance.to_checkpoint()

        assert checkpoint["init_kwargs"]["drift_term"] is not None
        assert checkpoint["init_kwargs"]["diffusion_term"] is not None

    def test_to_checkpoint_nonpicklable_callable_warns_and_replaces_with_none(
        self, make_process_with_lambda,
    ):
        """to_checkpoint() must warn and replace non-picklable callables with None.

        UserWarning (not ValueError) is issued so the save proceeds.
        The checkpoint must be pickle-safe with None replacing the callables.
        """
        instance = make_process_with_lambda()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        with pytest.warns(UserWarning, match="not picklable"):
            checkpoint = instance.to_checkpoint()

        import pickle
        pickle.dumps(checkpoint)

        assert checkpoint["init_kwargs"]["drift_term"] is None
        assert checkpoint["init_kwargs"]["diffusion_term"] is None

    def test_checkpoint_round_trip(self, make_process, x_batch, t_batch):
        """to_checkpoint() → from_checkpoint() must restore M, device, and equivalent operator output.

        allclose (not torch.equal): _restore_fit_state() re-runs self.fit() with a dummy
        zero tensor; grids are rebuilt via independent FFT calls, so outputs are
        mathematically equal but not bit-identical across invocations.
        """
        instance_a = make_process()
        instance_a.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = instance_a.to_checkpoint()
        instance_b = ForwardDiffusionProcess.from_checkpoint(checkpoint)

        h = torch.randn(BATCH_SIZE, CUSTOM_DIM)
        assert type(instance_b) is CirculantSymmetricMatrixDiffusionProcess
        assert instance_b.M_ == instance_a.M_
        assert instance_b.device_ == instance_a.device_
        assert instance_b.n_integration_points == instance_a.n_integration_points
        assert torch.allclose(
            instance_b.multiply_cov(h, t_batch),
            instance_a.multiply_cov(h, t_batch),
            atol=1e-5,
        )

    def test_restore_fit_state_missing_device_raises(self, unfitted_instance):
        """_restore_fit_state() must raise ValueError when 'device' is absent."""
        with pytest.raises(ValueError):
            unfitted_instance._restore_fit_state({"M": CUSTOM_DIM})

    def test_restore_fit_state_device_as_string_raises(self, unfitted_instance):
        """_restore_fit_state() must raise ValueError when 'device' is a plain string.

        Common source: JSON round-trip converting torch.device to its string repr.
        """
        with pytest.raises(ValueError):
            unfitted_instance._restore_fit_state(
                {"M": CUSTOM_DIM, "device": "cpu"},
            )

    def test_restore_fit_state_device_unsupported_type_raises(self, unfitted_instance):
        """_restore_fit_state() must raise ValueError for torch.device('mps').

        Passes isinstance(torch.device) but fails the ("cpu", "cuda") whitelist.
        """
        with pytest.raises(ValueError):
            unfitted_instance._restore_fit_state(
                {"M": CUSTOM_DIM, "device": torch.device("mps")},
            )

    def test_restore_fit_state_none_callable_raises(self, make_process_with_lambda):
        """_restore_fit_state() must raise ValueError when a callable is None.

        Failure mode: from_checkpoint() called without a pre-built instance on a
        checkpoint saved with non-picklable callables.
        """
        instance = make_process_with_lambda()
        instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        with pytest.warns(UserWarning):
            checkpoint = instance.to_checkpoint()

        broken = type(instance)(
            **{**instance.get_params(), "drift_term": None, "diffusion_term": None}
        )

        with pytest.raises(ValueError, match="non-picklable"):
            broken._restore_fit_state(checkpoint["fit_state"])


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantOperators
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantOperators(ForwardDiffusionOperatorTests):
    """Algebraic operator contracts; overrides both mixin tests with atol=1e-5 (vs default 1e-8).

    The circulant process adds one extra _fft_to_eigenspace/_eigenspace_to_fft roundtrip
    absent in VP and diagonal implementations, contributing ~1e-6 per element.
    In the inv-sigma round-trip, this error is further amplified by 1/√_cov ≈ 3.2x.
    """

    @pytest.fixture
    def process(self, x_batch, circulant_callables):
        """Fitted constant-eigenvalue process; at t_batch interior times _cov ∈ [0.095, 0.551]."""
        return _make_fitted_constant_process(x_batch, circulant_callables)

    def test_multiply_cov_equals_multiply_sigma_applied_twice(
        self, process, x_batch, t_batch,
    ):
        """multiply_cov(h, t) must equal multiply_sigma(multiply_sigma(h, t), t).

        atol=1e-5 replaces the default 1e-8: the RHS contains one extra
        _fft_to_eigenspace(_eigenspace_to_fft(·)) roundtrip absent in the LHS.
        """
        result_cov = process.multiply_cov(x_batch, t_batch)
        result_sigma_twice = process.multiply_sigma(
            process.multiply_sigma(x_batch, t_batch),
            t_batch,
        )

        assert torch.allclose(result_cov, result_sigma_twice, atol=1e-5)

    def test_sigma_inv_sigma_round_trip(self, process, x_batch, t_batch):
        """multiply_inv_sigma(multiply_sigma(z, t), t) must recover z.

        atol=1e-5 replaces the default 1e-8: extra FFT roundtrip errors amplified
        by 1/√_cov inside multiply_inv_sigma. t_batch interior times keep √_cov
        well above _PSEUDOINV_THRESHOLD=1e-3.
        """
        z_sigma = process.multiply_sigma(x_batch, t_batch)
        result  = process.multiply_inv_sigma(z_sigma, t_batch)

        assert torch.allclose(result, x_batch, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantSampleLimit
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantSampleLimit(ForwardDiffusionSampleLimitTests):
    """Tests for sample_limit_distribution; inherits 6 universal tests.

    Two additional statistical tests use orthogonal oracles: the VP identity
    μ_k(T)² + sigma_k(T)² = 1, and the per-dimension variance from _cov(T=1).
    """

    @pytest.fixture
    def make_process(self, vp_eigenvalue_callables):
        """Factory using VP-equivalent callables; limit distribution ≈ N(0, I_M)."""
        def factory() -> CirculantSymmetricMatrixDiffusionProcess:
            drift_term, diffusion_term = vp_eigenvalue_callables
            return CirculantSymmetricMatrixDiffusionProcess(
                drift_term=drift_term,
                diffusion_term=diffusion_term,
                fourier_drift=True,
                fourier_diffusion=True,
            )
        return factory

    @pytest.fixture
    def make_process_alt_seed(self, circulant_callables):
        """Factory using circulant_callables; limit distribution ≈ N(0, c_Y(1)·I_M)."""
        def factory() -> CirculantSymmetricMatrixDiffusionProcess:
            drift_term, diffusion_term = circulant_callables
            return CirculantSymmetricMatrixDiffusionProcess(
                drift_term=drift_term,
                diffusion_term=diffusion_term,
                fourier_drift=True,
                fourier_diffusion=True,
            )
        return factory


    def test_limit_distribution_is_standard_normal_for_vp_equivalent_callables(
        self, fitted_process,
    ):
        """The limit distribution must be approximately N(0, I_M) for VP eigenvalues.

        Derivation: d_t = -0.5 β(t), diffusion_term = √β(t) satisfy μ_k(t)² + sigma_k(t)² = 1.
        At T=1: μ_k(1) ≈ 0.080, sigma_k(1) ≈ 0.997; systematic deviation from 1.0 is 0.003.

        atol=0.05 (n=10 000): sampling noise SE ≈ 0.007 → combined |std - 1.0| ≈ 0.010.
        """
        n = 10_000
        samples = fitted_process.sample_limit_distribution(
            n_samples=n,
        )

        assert samples.mean(dim=0).abs().max() < 0.05, (
            f"Per-dimension mean too large: "
            f"{samples.mean(dim=0).abs().max():.4f}"
        )
        assert (samples.std(dim=0) - 1.0).abs().max() < 0.05, (
            f"Per-dimension std deviates from oracle ≈ 1.0: "
            f"{(samples.std(dim=0) - 1.0).abs().max():.4f}"
        )

    def test_limit_distribution_statistics(self, x_batch, circulant_callables):
        """The limit distribution must match the per-dimension variance from _cov(T=1).

        With D=-0.5, g=1.0: c_Y(1) = 1 - exp(-1) ≈ 0.632; expected_std ≈ 0.795.
        Oracle read from process._cov(T=1) so the test stays correct if callables change.

        atol=0.05 (n=10 000): mean SE ≈ 0.008 (6sigma guard), std SE ≈ 0.006 (9sigma guard).
        """
        n = 10_000

        drift_term, diffusion_term = circulant_callables
        process = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            fourier_drift=True,
            fourier_diffusion=True,
        )
        process.fit(x_batch)

        t_end        = torch.ones(1)
        cov_matrix   = process._cov(t_end)
        expected_var = cov_matrix[0].diagonal()
        expected_std = torch.sqrt(expected_var)

        samples = process.sample_limit_distribution(n_samples=n)

        assert samples.mean(dim=0).abs().max() < 0.05, (
            f"Per-dimension mean too large: "
            f"{samples.mean(dim=0).abs().max():.4f}"
        )
        assert (samples.std(dim=0) - expected_std).abs().max() < 0.05, (
            f"Per-dimension std deviates from oracle "
            f"(expected ≈ {expected_std[0].item():.4f} per dim): "
            f"{(samples.std(dim=0) - expected_std).abs().max():.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantCov
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantCov:
    """Tests for _cov(t); returns shape (N, M, M) — a full covariance matrix per sample.

    Float32 tolerances for Q diag(c_Y) Q^T with ‖Q Q^T − I‖_F ≤ 1e-5:
        symmetry: atol=1e-6; diagonal consistency: atol=1e-6 (on-diag), 1e-5 (off-diag);
        eigenspace invariant: atol=1e-5.
    """

    @pytest.fixture
    def process(self, x_batch, circulant_callables):
        """Fitted constant-eigenvalue process; closed-form _cov(t) = (1 - exp(-t)) · I_M."""
        return _make_fitted_constant_process(x_batch, circulant_callables)

    def test_cov_shape_is_N_M_M(self, process, t_batch):
        """_cov must return shape (N, M, M)."""
        result = process._cov(t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM, DATA_DIM)

    def test_cov_at_t0_is_zero_matrix(self, process, t_zero):
        """_cov(0) must be the zero matrix for all N.

        allclose with atol=1e-5 (not torch.equal) tolerates interpolation
        rounding from _batch_linear_interp_1d at the left boundary.
        """
        result = process._cov(t_zero)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_cov_is_symmetric(self, process, t_batch):
        """_cov(t) must equal its own transpose: cov[n] = cov[n].T for all n.

        atol=1e-6: |cov[i,j] - cov[j,i]| ≤ 2·sqrt(M)·ε₃₂·c_max ≈ 5e-7.
        """
        cov = process._cov(t_batch)

        assert torch.allclose(cov, cov.transpose(-2, -1), atol=1e-6)

    def test_cov_is_positive_semidefinite(self, process, t_batch):
        """_cov(t) must be positive semi-definite for all n.

        -1e-6 (not 0) accommodates float32 rounding in the matrix products.
        """
        eigvals = torch.linalg.eigvalsh(process._cov(t_batch))

        assert eigvals.min() >= -1e-6, (
            f"_cov has negative eigenvalue: {eigvals.min().item():.4e}"
        )

    def test_cov_diagonal_equals_diagonal_process_cov_for_uniform_eigenvalues(
        self, process, t_batch,
    ):
        """Main diagonal of _cov(t) must equal diagonal_process._cov(t) for uniform eigenvalues.

        For uniform c_Y: (Q diag(c_Y) Q^T)[i,i] = c_Y · (Q Q^T)[i,i] = c_Y.
        atol=1e-6 (on-diag): diagonal error ≤ c_Y · sqrt(M) · ε₃₂ ≈ 3e-7.
        atol=1e-5 (off-diag): off-diagonal ≤ c_Y · ‖Q Q^T - I‖_F / sqrt(M) ≈ 6e-6.
        """
        cov      = process._cov(t_batch)                            # (N, M, M)
        cov_diag = cov.diagonal(dim1=-2, dim2=-1)                   # (N, M)

        assert torch.allclose(
            cov_diag,
            process.diagonal_process_._cov(t_batch),
            atol=1e-6,
        ), (
            f"Diagonal of _cov disagrees with diagonal_process._cov: "
            f"max error = {(cov_diag - process.diagonal_process_._cov(t_batch)).abs().max().item():.2e}"
        )

        off_diag = cov - torch.diag_embed(cov_diag)                 # (N, M, M)
        assert torch.allclose(
            off_diag, torch.zeros_like(off_diag), atol=1e-5,
        ), (
            f"Off-diagonal elements of _cov are not near zero: "
            f"max |off-diag| = {off_diag.abs().max().item():.2e}"
        )

    def test_cov_in_eigenspace_is_diagonal(self, process, t_batch):
        """Q^T _cov(t) Q must equal diag(c_Y(t)); fundamental algebraic invariant.

        _cov(t) = Q diag(c_Y(t)) Q^T  ⟹  Q^T _cov(t) Q = diag(c_Y(t)).
        atol=1e-5 for both on-diag and off-diag, from Q^T Q ≈ I up to ~1e-5.
        """
        cov = process._cov(t_batch)                          # (N, M, M)
        Q   = process.q_mat_                                 # (M, M)

        cov_y = Q.T.unsqueeze(0) @ cov @ Q.unsqueeze(0)      # (N, M, M)

        cov_y_diag    = cov_y.diagonal(dim1=-2, dim2=-1)     # (N, M)
        cov_y_offdiag = cov_y - torch.diag_embed(cov_y_diag) # (N, M, M)

        assert torch.allclose(
            cov_y_offdiag, torch.zeros_like(cov_y_offdiag), atol=1e-5,
        ), (
            f"Eigenspace covariance is not diagonal: "
            f"max |off-diag| = {cov_y_offdiag.abs().max().item():.2e}"
        )
        assert torch.allclose(
            cov_y_diag,
            process.diagonal_process_._cov(t_batch),
            atol=1e-5,
        ), (
            f"Eigenspace diagonal disagrees with diagonal_process_._cov: "
            f"max error = {(cov_y_diag - process.diagonal_process_._cov(t_batch)).abs().max().item():.2e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantDrift
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantDrift:
    """Tests for drift(x, t) = Q diag(λ(t)) Qᵀ x via the FFT shortcut.

    Five tests in escalating specificity: shape, linearity, oracle (uniform eigenvalues),
    FFT-vs-matmul equivalence, and fourier/row convention equivalence.
    """

    @pytest.fixture
    def process(self, x_batch, circulant_callables):
        """Fitted constant-eigenvalue process; oracle: drift(x, t) = -0.5 · x."""
        return _make_fitted_constant_process(x_batch, circulant_callables)

    def test_drift_output_shape(self, process, x_batch, t_batch):
        """drift(x, t) must return shape (N, M)."""
        result = process.drift(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_drift_is_linear_in_x(self, process, x_batch, t_batch):
        """drift(α·x, t) must equal α·drift(x, t) for any scalar α.

        α=3.7 (non-integer) rules out a pass that is linear only for integer multiples.
        """
        alpha = 3.7

        result_scaled      = process.drift(alpha * x_batch, t_batch)
        result_then_scaled = alpha * process.drift(x_batch, t_batch)

        assert torch.allclose(result_scaled, result_then_scaled, atol=1e-5, rtol=1e-8)

    def test_drift_with_uniform_eigenvalues_equals_scalar_times_x(
        self, process, x_batch, t_batch,
    ):
        """drift(x, t) must equal -0.5 · x when all eigenvalues equal -0.5.

        Q diag(c) Qᵀ x = c · Q Qᵀ x ≈ c · x by Q-orthogonality.
        atol=1e-5: |c| · ‖Q Qᵀ - I‖ · ‖x‖∞/√M ≈ 3.75e-6.
        """
        result = process.drift(x_batch, t_batch)

        assert torch.allclose(result, -0.5 * x_batch, atol=1e-5)

    def test_drift_fft_path_matches_explicit_matrix_vector_product(
        self, process, x_batch, t_batch,
    ):
        """drift(x, t) must equal the explicit Q diag(λ(t)) Qᵀ x computation.

        Pins the FFT shortcut against the O(N·M²) ground truth.
        atol=1e-5: combined float32 error from both paths ≤ 3.06e-6.
        """
        Q        = process.q_mat_                              # (M, M)
        lambda_t = process.lambdas_(t_batch)                  # (N, M)
        N        = BATCH_SIZE

        lambda_diag = torch.diag_embed(lambda_t)             # (N, M, M)
        x_col       = x_batch.unsqueeze(-1)                  # (N, M, 1)
        Q_batch     = Q.unsqueeze(0).expand(N, -1, -1)       # (N, M, M)
        Qt_batch    = Q.T.unsqueeze(0).expand(N, -1, -1)     # (N, M, M)

        expected = torch.bmm(
            Q_batch,
            torch.bmm(lambda_diag, torch.bmm(Qt_batch, x_col)),
        ).squeeze(-1)                                         # (N, M)

        result = process.drift(x_batch, t_batch)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_drift_row_and_fourier_inputs_give_same_result(
        self,
        x_batch,
        t_batch,
        circulant_callables,
        constant_row_callables,
    ):
        """Drift must agree for fourier_drift=True and fourier_drift=False for the same operator.

        atol=1e-4: the row path applies one extra FFT to convert the generating sequence
        to eigenvalues, adding ~1e-6 on top of the shared drift FFT operations.
        """
        drift_fourier, diffusion_fourier = circulant_callables
        drift_row, _                     = constant_row_callables

        p_fourier = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_fourier,
            diffusion_term=diffusion_fourier,
            fourier_drift=True,
            fourier_diffusion=True,
        )
        p_fourier.fit(x_batch)

        p_row = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_row,
            diffusion_term=diffusion_fourier,
            fourier_drift=False,
            fourier_diffusion=True,
        )
        p_row.fit(x_batch)

        result_fourier = p_fourier.drift(x_batch, t_batch)
        result_row     = p_row.drift(x_batch, t_batch)

        assert torch.allclose(result_fourier, result_row, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantDiffusionMethods
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantDiffusionMethods:
    """Tests for diffusion(t), diffusion_times_v(v, t), and diffusion_gram_times_v(v, t).

    diffusion(t)                = Q diag(g(t)) Qᵀ             → (N, M, M)
    diffusion_times_v(v, t)     = Q diag(g(t)) Qᵀ v           → (N, M) [FFT shortcut]
    diffusion_gram_times_v(v,t) = Q diag(g(t)²) Qᵀ v          → (N, M) [eigenvalue squaring]

    Tolerances: symmetry atol=1e-6; diffusion_times_v vs G@v atol=1e-5;
    diffusion_gram_times_v vs applied-twice atol=1e-5 (extra Q^T·Q roundtrip in RHS).
    """

    @pytest.fixture
    def process(self, x_batch, circulant_callables):
        """Fitted constant-eigenvalue process; g=1.0 for all eigenmodes, G(t) ≈ I_M."""
        return _make_fitted_constant_process(x_batch, circulant_callables)

    def test_diffusion_output_shape(self, process, t_batch):
        """diffusion(t) must return shape (N, M, M)."""
        result = process.diffusion(t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM, DATA_DIM)

    def test_diffusion_is_symmetric(self, process, t_batch):
        """diffusion(t) must equal its own transpose: G[n] = G[n].T for all n.

        atol=1e-6: |G[i,j] - G[j,i]| ≤ 2·sqrt(M)·ε₃₂·g_max ≈ 9.6e-7.
        """
        G = process.diffusion(t_batch)

        assert torch.allclose(G, G.transpose(-2, -1), atol=1e-6)

    def test_diffusion_before_fit_raises_error(self, circulant_callables, t_batch):
        """diffusion(t) must raise ValueError before fit() is called."""
        drift_term, diffusion_term = circulant_callables
        p = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            fourier_drift=True,
            fourier_diffusion=True,
        )

        with pytest.raises(ValueError):
            p.diffusion(t_batch)

    def test_diffusion_times_v_output_shape(self, process, x_batch, t_batch):
        """diffusion_times_v(v, t) must return shape (N, M)."""
        result = process.diffusion_times_v(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_diffusion_times_v_matches_full_matrix_times_v(
        self, process, x_batch, t_batch,
    ):
        """diffusion_times_v(v, t) must equal the full G(t) @ v computation.

        Pins the FFT shortcut against the O(N·M²) matmul ground truth.
        atol=1e-5: combined rounding from both paths ≤ 4.8e-6.
        """
        G        = process.diffusion(t_batch)                              # (N, M, M)
        expected = torch.bmm(G, x_batch.unsqueeze(-1)).squeeze(-1)        # (N, M)

        result = process.diffusion_times_v(x_batch, t_batch)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_diffusion_gram_times_v_output_shape(self, process, x_batch, t_batch):
        """diffusion_gram_times_v(v, t) must return shape (N, M)."""
        result = process.diffusion_gram_times_v(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_diffusion_gram_times_v_equals_diffusion_times_v_applied_twice(
        self, process, x_batch, t_batch,
    ):
        """diffusion_gram_times_v(v, t) must equal diffusion_times_v applied twice.

        G is symmetric, so G G^T = G². The implementation squares eigenvalues (g²)
        directly rather than composing two matrix applications.
        atol=1e-5: RHS has one extra Q^T·Q roundtrip absent in LHS; total ≤ 7.2e-6.
        """
        expected = process.diffusion_times_v(
            process.diffusion_times_v(x_batch, t_batch),
            t_batch,
        )
        result = process.diffusion_gram_times_v(x_batch, t_batch)

        assert torch.allclose(result, expected, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantMeanCond
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantMeanCond:
    """Tests for mean_cond(x, t) = Q E[Y(t) | Y(0) = Q^T x] mapped back via _eigenspace_to_fft."""

    @pytest.fixture
    def process(self, x_batch, circulant_callables):
        """Fitted constant-eigenvalue process; closed-form oracle: mean_cond(x, t) = exp(-0.5t) · x."""
        return _make_fitted_constant_process(x_batch, circulant_callables)

    @pytest.fixture
    def x_ones(self, t_sweep):
        """All-ones input matched to t_sweep; simplifies the oracle to a scalar time factor."""
        return torch.ones(t_sweep.shape[0], DATA_DIM)

    def test_mean_cond_output_shape(self, process, x_batch, t_batch):
        """mean_cond(x, t) must return shape (N, M)."""
        result = process.mean_cond(x_batch, t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_mean_cond_at_t0_equals_x(self, process, x_batch, t_zero):
        """mean_cond(x, 0) must equal x.

        atol=1e-5: error from FFT round-trip (Q Q^T ≈ I up to 1e-5) plus
        interpolation rounding at the precomputed grid's left boundary.
        """
        result = process.mean_cond(x_batch, t_zero)

        assert torch.allclose(result, x_batch, atol=1e-5)

    def test_mean_cond_is_linear_in_x(self, process, x_batch, t_batch):
        """mean_cond(α·x, t) must equal α·mean_cond(x, t).

        α=3.7 (non-integer) rules out a pass that is linear only for α=1.
        """
        alpha = 3.7

        result_scaled      = process.mean_cond(alpha * x_batch, t_batch)
        result_then_scaled = alpha * process.mean_cond(x_batch, t_batch)

        assert torch.allclose(result_scaled, result_then_scaled, atol=1e-5, rtol=1e-10)

    def test_mean_cond_uniform_eigenvalues_oracle(self, process, t_sweep, x_ones):
        """With all eigenvalues = -0.5 and x = ones, mean_cond must equal exp(-0.5 t).

        ∫₀ᵗ D du = -0.5t; Q diag(exp(-0.5t)) Q^T · ones = exp(-0.5t) · ones.
        atol=1e-4: dominant error from trapezoid integration (~1e-4, n=1000 points).
        """
        expected = (
            torch.exp(-0.5 * t_sweep)
            .unsqueeze(1)
            .expand(-1, DATA_DIM)
        )

        result = process.mean_cond(x_ones, t_sweep)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_mean_cond_decays_monotonically_for_mean_reverting_eigenvalues(
        self, process, t_sweep, x_ones,
    ):
        """‖mean_cond(x, t)‖₂ must be non-increasing over t for D < 0.

        +1e-6 tolerance absorbs float rounding without masking genuine non-monotone
        behaviour (a sign flip or wrong integration direction produces steps ~1e-3).
        """
        result = process.mean_cond(x_ones, t_sweep)   # (N_sweep, DATA_DIM)
        norms  = result.norm(dim=1)                    # (N_sweep,)

        assert (norms[1:] <= norms[:-1] + 1e-6).all(), (
            f"mean_cond norm is not monotonically non-increasing: "
            f"max upward step = {(norms[1:] - norms[:-1]).clamp(min=0).max().item():.2e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestFourierVsRowInputModes
# ─────────────────────────────────────────────────────────────────────────────

class TestFourierVsRowInputModes:
    """Tests that fourier_*=True and fourier_*=False produce identical observable output.

    mean_cond exercises the D-integral path independently of multiply_sigma and multiply_cov
    (F-integral path). Testing all three ensures neither _precompute_D_integral nor
    _precompute_F_integral silently picks the wrong branch.

    atol=1e-4: the row path introduces an extra FFT (~1e-6 per eigenvalue), well below
    the ~1e-4 trapezoid residual shared by both paths.
    """

    def test_mean_cond_equivalence(self, p_fourier, p_row, x_batch, t_batch):
        """mean_cond must produce the same output for fourier_drift=True and =False."""
        result_fourier = p_fourier.mean_cond(x_batch, t_batch)
        result_row     = p_row.mean_cond(x_batch, t_batch)

        assert torch.allclose(result_fourier, result_row, atol=1e-4)

    def test_multiply_sigma_equivalence(self, p_fourier, p_row, x_batch, t_batch):
        """multiply_sigma must produce the same output for fourier_diffusion=True and =False."""
        result_fourier = p_fourier.multiply_sigma(x_batch, t_batch)
        result_row     = p_row.multiply_sigma(x_batch, t_batch)

        assert torch.allclose(result_fourier, result_row, atol=1e-4)

    def test_multiply_cov_equivalence(self, p_fourier, p_row, x_batch, t_batch):
        """multiply_cov must produce the same output for fourier_diffusion=True and =False."""
        result_fourier = p_fourier.multiply_cov(x_batch, t_batch)
        result_row     = p_row.multiply_cov(x_batch, t_batch)

        assert torch.allclose(result_fourier, result_row, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# TestGetCosineBasis
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCosineBasis:
    """Pure-function tests for _get_cosine_basis(M).

    Covers shape (even/odd M), orthogonality (left/right products, even/odd M),
    and column invariants (DC = 1/sqrt(M), Nyquist = cos(π·i)/sqrt(M) for even M).
    """

    def test_shape_even_m(self):
        """_get_cosine_basis(DATA_DIM) must return shape (DATA_DIM, DATA_DIM)."""
        Q = _get_cosine_basis(DATA_DIM)

        assert Q.shape == (DATA_DIM, DATA_DIM)

    def test_shape_odd_m(self):
        """_get_cosine_basis(9) must return shape (9, 9).

        Odd M skips the Nyquist column insertion branch.
        """
        Q = _get_cosine_basis(9)

        assert Q.shape == (9, 9)

    def test_q_qt_equals_identity_even_m(self):
        """Q @ Q.T must equal I_M for even M = DATA_DIM.

        atol=1e-5: |error| ≤ sqrt(M) · ε₃₂ ≈ 4.8e-7; normalization bug ≈ 0.94.
        """
        Q   = _get_cosine_basis(DATA_DIM)
        eye = torch.eye(DATA_DIM)

        assert torch.allclose(Q @ Q.T, eye, atol=1e-5)

    def test_qt_q_equals_identity_even_m(self):
        """Q.T @ Q must equal I_M for even M = DATA_DIM.

        Tested independently because a transposition error in the returned matrix
        satisfies one product while violating the other.
        """
        Q   = _get_cosine_basis(DATA_DIM)
        eye = torch.eye(DATA_DIM)

        assert torch.allclose(Q.T @ Q, eye, atol=1e-5)

    def test_orthogonality_odd_m(self):
        """Both Q @ Q.T and Q.T @ Q must equal I_9 for odd M = 9.

        atol=1e-5: for M=9, |error| ≤ sqrt(9) · ε₃₂ ≈ 3.6e-7.
        """
        Q   = _get_cosine_basis(9)
        eye = torch.eye(9)

        assert torch.allclose(Q @ Q.T, eye, atol=1e-5), (
            "Q @ Q.T ≠ I₉: orthogonality violated for odd M"
        )
        assert torch.allclose(Q.T @ Q, eye, atol=1e-5), (
            "Q.T @ Q ≠ I₉: orthogonality violated for odd M"
        )

    def test_first_column_is_constant_dc(self):
        """Q[:, 0] must be a constant vector with value 1/sqrt(DATA_DIM).

        atol=1e-6: single float32 division; rounding ≤ ε₃₂/2 ≈ 6e-8.
        """
        Q        = _get_cosine_basis(DATA_DIM)
        expected = torch.full((DATA_DIM,), 1.0 / DATA_DIM**0.5)

        assert torch.allclose(Q[:, 0], expected, atol=1e-6)

    def test_nyquist_column_is_alternating_for_even_m(self):
        """Q[:, DATA_DIM//2] must equal cos(π·i) / sqrt(DATA_DIM) for i = 0, …, M-1.

        The Nyquist frequency k = M/2 produces (+1, −1, +1, …)/sqrt(M) = cos(πi)/sqrt(M).
        Present only for even M. atol=1e-6: two float32 operations; rounding ≈ 2.4e-7.
        """
        Q        = _get_cosine_basis(DATA_DIM)
        indices  = torch.arange(DATA_DIM, dtype=torch.float32)
        expected = torch.cos(torch.pi * indices) / DATA_DIM**0.5

        assert torch.allclose(Q[:, DATA_DIM // 2], expected, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TestFftEigenspaceRoundTrip
# ─────────────────────────────────────────────────────────────────────────────

class TestFftEigenspaceRoundTrip:
    """Tests for _fft_to_eigenspace and _eigenspace_to_fft as a matched pair.

    _fft_to_eigenspace(x) = x @ Q via O(N·M·log M) RFFT.
    _eigenspace_to_fft(y) = y @ Q.T via IRFFT.
    Both must form exact round-trips and agree element-by-element with explicit matmuls.

    atol=1e-5 throughout: each FFT/IFFT pair contributes ≤ log₂(M) · ε₃₂ · ‖x‖∞ ≈ 1.4e-6.
    """

    def test_eigenspace_then_back_is_identity(self, x_batch):
        """_eigenspace_to_fft(_fft_to_eigenspace(x)) must recover x."""
        result = _eigenspace_to_fft(_fft_to_eigenspace(x_batch))

        assert torch.allclose(result, x_batch, atol=1e-5)

    def test_back_then_eigenspace_is_identity(self, x_batch):
        """_fft_to_eigenspace(_eigenspace_to_fft(x)) must recover x.

        PyTorch's RFFT and IRFFT have asymmetric normalisation conventions;
        one composition can be correct while the other is off by a factor of M.
        """
        result = _fft_to_eigenspace(_eigenspace_to_fft(x_batch))

        assert torch.allclose(result, x_batch, atol=1e-5)

    def test__fft_to_eigenspace_matches_q_matmul(self, x_batch):
        """_fft_to_eigenspace(x) must equal x @ Q."""
        Q        = _get_cosine_basis(DATA_DIM)
        expected = x_batch @ Q

        result = _fft_to_eigenspace(x_batch)

        assert torch.allclose(result, expected, atol=1e-5)

    def test__eigenspace_to_fft_matches_q_t_matmul(self):
        """_eigenspace_to_fft(y) must equal y @ Q.T."""
        torch.manual_seed(SEED)
        y        = torch.randn(BATCH_SIZE, DATA_DIM)
        Q        = _get_cosine_basis(DATA_DIM)
        expected = y @ Q.T

        result = _eigenspace_to_fft(y)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_parseval_norm_preservation(self, x_batch):
        """‖_fft_to_eigenspace(x)[n]‖₂ must equal ‖x[n]‖₂ for every sample n.

        x @ Q preserves norms since Q is orthogonal.
        atol=1e-5: two norm calls combined ≤ 9.6e-7.
        """
        norms_original   = x_batch.norm(dim=1)
        norms_eigenspace = _fft_to_eigenspace(x_batch).norm(dim=1)

        assert torch.allclose(norms_eigenspace, norms_original, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# TestDuplicateSymmetricHelpers
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateSymmetricHelpers:
    """Pure function tests for _duplicate_symmetric_eigenvalues and
    _duplicate_symmetric_row_and_get_eigenvalues, in isolation from fit().
    """

    def test_duplicate_eigenvalues_output_shape_even_m(self, t_batch):
        """_duplicate_symmetric_eigenvalues must return shape (N, M) for even M=DATA_DIM."""
        M        = DATA_DIM   # 16
        half_dim = M // 2 + 1 # 9

        lambda_half = lambda t: torch.full((t.shape[0], half_dim), -0.5)
        lambdas     = _duplicate_symmetric_eigenvalues(lambda_half, M)
        result      = lambdas(t_batch)

        assert result.shape == (BATCH_SIZE, DATA_DIM)

    def test_duplicate_eigenvalues_output_shape_odd_m(self, t_batch):
        """_duplicate_symmetric_eigenvalues must return shape (N, M) for odd M=9.

        Odd M uses (M+1)//2 as the flip upper-bound vs M//2 for even M.
        """
        M        = 9
        half_dim = M // 2 + 1  # 5

        lambda_half = lambda t: torch.full((t.shape[0], half_dim), -0.5)
        lambdas     = _duplicate_symmetric_eigenvalues(lambda_half, M)
        result      = lambdas(t_batch)

        assert result.shape == (BATCH_SIZE, M)

    def test_duplicate_eigenvalues_symmetry_property(self, t_batch):
        """The duplicated spectrum must satisfy lambdas[:, k] == lambdas[:, M-k].

        Each position in the half-spectrum is distinct (value j+1 at position j)
        so a wrong flip boundary produces a value mismatch at a specific k.
        torch.equal: the duplicated values are exact copies — no arithmetic, only indexing.
        """
        M        = DATA_DIM   # 16
        half_dim = M // 2 + 1 # 9

        values      = torch.arange(1, half_dim + 1, dtype=torch.float32)
        lambda_half = lambda t: values.unsqueeze(0).expand(t.shape[0], -1)
        lambdas     = _duplicate_symmetric_eigenvalues(lambda_half, M)
        result      = lambdas(t_batch)  # (BATCH_SIZE, M)

        for k in range(1, (M - 1) // 2 + 1):
            assert torch.equal(result[:, k], result[:, M - k]), (
                f"Symmetry violated at k={k}: "
                f"lambdas[:, {k}] != lambdas[:, {M - k}]"
            )

    @pytest.mark.parametrize("M", [DATA_DIM, 9])
    def test_duplicate_row_output_shape_even_and_odd_m(self, t_batch, M):
        """_duplicate_symmetric_row_and_get_eigenvalues must return shape (N, M)."""
        half_dim = M // 2 + 1

        c_half  = lambda t: torch.full((t.shape[0], half_dim), 1.0)
        lambdas = _duplicate_symmetric_row_and_get_eigenvalues(c_half, M)
        result  = lambdas(t_batch)

        assert result.shape == (BATCH_SIZE, M)

    def test_duplicate_row_real_valued(self, t_batch):
        """_duplicate_symmetric_row_and_get_eigenvalues must return dtype float32, not complex64.

        The implementation returns torch.fft.fft(row).real; a missing .real suffix
        yields complex64 whose downstream use in real arithmetic raises dtype errors.
        """
        M        = DATA_DIM   # 16
        half_dim = M // 2 + 1 # 9

        c_half  = lambda t: torch.full((t.shape[0], half_dim), 1.0)
        lambdas = _duplicate_symmetric_row_and_get_eigenvalues(c_half, M)
        result  = lambdas(t_batch)

        assert result.dtype == torch.float32

    def test_duplicate_row_matches_explicit_fft(self, t_batch):
        """Output must equal torch.fft.fft(full_symmetric_row).real.

        Reference: [c_half | flip(c_half[:, 1:(M+1)//2])] then FFT.
        atol=1e-5: per-element rounding from length-16 FFT ≤ 4.3e-6.
        """
        M        = DATA_DIM   # 16
        half_dim = M // 2 + 1 # 9

        values      = torch.arange(1, half_dim + 1, dtype=torch.float32)
        c_half_val  = values.unsqueeze(0).expand(BATCH_SIZE, -1)  # (BATCH_SIZE, 9)
        c_half      = lambda t: values.unsqueeze(0).expand(t.shape[0], -1)

        symmetric_row = torch.cat(
            [c_half_val, torch.flip(c_half_val[:, 1:(M + 1) // 2], dims=[1])],
            dim=1,
        )  # (BATCH_SIZE, 16)
        expected = torch.fft.fft(symmetric_row).real  # (BATCH_SIZE, 16)

        lambdas = _duplicate_symmetric_row_and_get_eigenvalues(c_half, M)
        result  = lambdas(t_batch)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_uniform_row_gives_uniform_eigenvalues(self, t_batch):
        """[c_0, 0, …, 0] half-row must produce all M eigenvalues equal to c_0.

        FFT([c_0, 0, …, 0])[k] = c_0 for all k. Bridges the two input modes:
        the sparse-row convention is equivalent to the constant-eigenvalue convention.
        atol=1e-5: FFT rounding ≤ log₂(16) · ε₃₂ · |c_0| ≈ 2.4e-7.
        """
        M        = DATA_DIM   # 16
        half_dim = M // 2 + 1 # 9

        c_half = lambda t: torch.cat(
            [
                torch.full((t.shape[0], 1), -0.5),
                torch.zeros(t.shape[0], half_dim - 1),
            ],
            dim=1,
        )

        lambdas = _duplicate_symmetric_row_and_get_eigenvalues(c_half, M)
        result  = lambdas(t_batch)  # (BATCH_SIZE, M)

        assert torch.allclose(result, torch.full_like(result, -0.5), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# TestCirculantDeviceConsistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCirculantDeviceConsistency:
    """Tests that CirculantSymmetricMatrixDiffusionProcess propagates the training-data device.

    Two device-sensitive components: q_mat (built with device from _get_cosine_basis)
    and diagonal_process (fitted on _fft_to_eigenspace(x), inheriting x.device).

    Both cpu_process and cuda_process are class-scoped. Each generates its own training
    data inline — pytest forbids class-scoped fixtures from depending on function-scoped
    conftest fixtures. circulant_callables must use device=t.device so the same callables
    work on both devices.
    """

    @pytest.fixture(scope="class")
    def cpu_process(self, circulant_callables):
        """CirculantSymmetricMatrixDiffusionProcess fitted on CPU training data."""
        torch.manual_seed(SEED)
        x_cpu = torch.randn(BATCH_SIZE, DATA_DIM)
        drift_term, diffusion_term = circulant_callables
        p = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            fourier_drift=True,
            fourier_diffusion=True,
        )
        p.fit(x_cpu)
        return p

    @pytest.fixture(scope="class")
    def cuda_process(self, circulant_callables):
        """CirculantSymmetricMatrixDiffusionProcess fitted on CUDA training data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(SEED)
        x_cuda = torch.randn(BATCH_SIZE, DATA_DIM, device="cuda")
        drift_term, diffusion_term = circulant_callables
        p = CirculantSymmetricMatrixDiffusionProcess(
            drift_term=drift_term,
            diffusion_term=diffusion_term,
            fourier_drift=True,
            fourier_diffusion=True,
        )
        p.fit(x_cuda)
        return p

    def test_device_attribute_is_cpu_after_cpu_fit(self, cpu_process):
        """self.device must equal torch.device('cpu') after fitting on CPU data."""
        assert cpu_process.device_ == torch.device("cpu")

    def test_q_mat_is_on_cpu_after_cpu_fit(self, cpu_process):
        """q_mat must reside on CPU after fitting on CPU data."""
        assert cpu_process.q_mat_.device.type == "cpu"

    def test_diagonal_process_grids_are_on_cpu(self, cpu_process):
        """Both precomputed integral grids in diagonal_process must be on CPU.

        Tested with separate assertions so a failure identifies which integral path
        (D or F) dropped the device.
        """
        diag = cpu_process.diagonal_process_

        assert diag.precomputed_d_grid_.device.type == "cpu", (
            "precomputed_d_grid_ must be on cpu — check that _fft_to_eigenspace(x) "
            "preserves x.device before passing to DiagonalDiffusionProcess.fit()"
        )
        assert diag.precomputed_sigma_f_grid_.device.type == "cpu", (
            "precomputed_sigma_f_grid_ must be on cpu — check that _fft_to_eigenspace(x) "
            "preserves x.device before passing to DiagonalDiffusionProcess.fit()"
        )

    def test_mean_cond_output_device_is_cpu(self, cpu_process, x_batch, t_batch):
        """mean_cond output must be on CPU when the process was fitted on CPU data."""
        result = cpu_process.mean_cond(x_batch, t_batch)

        assert result.device.type == "cpu"

    def test_multiply_sigma_output_device_is_cpu(
        self, cpu_process, x_batch, t_batch,
    ):
        """multiply_sigma output must be on CPU when the process was fitted on CPU data."""
        result = cpu_process.multiply_sigma(x_batch, t_batch)

        assert result.device.type == "cpu"

    def test_diffusion_times_v_output_device_is_cpu(
        self, cpu_process, x_batch, t_batch,
    ):
        """diffusion_times_v output must be on CPU when the process was fitted on CPU data."""
        result = cpu_process.diffusion_times_v(x_batch, t_batch)

        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_all_grids_are_on_cuda_after_cuda_fit(self, cuda_process):
        """After fitting on CUDA data, every precomputed tensor must be on CUDA.

        Four attributes checked with individual failure messages to identify which
        link in the device-propagation chain failed.
        """
        diag = cuda_process.diagonal_process_

        assert cuda_process.device_.type == "cuda", (
            "self.device_ must be 'cuda' — check self.device = x.device in fit()"
        )
        assert cuda_process.q_mat_.device.type == "cuda", (
            "q_mat must be on cuda — check _get_cosine_basis device argument in fit()"
        )
        assert diag.precomputed_d_grid_.device.type == "cuda", (
            "precomputed_d_grid_ must be on cuda — check that "
            "_fft_to_eigenspace(x_cuda) preserves x.device"
        )
        assert diag.precomputed_sigma_f_grid_.device.type == "cuda", (
            "precomputed_sigma_f_grid_ must be on cuda — check "
            "_precompute_F_integral device routing in DiagonalDiffusionProcess"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mean_cond_and_multiply_methods_output_device_is_cuda(
        self, cuda_process,
    ):
        """All operator outputs must be on CUDA when the process was fitted on CUDA data."""
        torch.manual_seed(SEED)
        x_cuda = torch.randn(BATCH_SIZE, DATA_DIM, device="cuda")
        t_cuda = torch.linspace(0.1, 0.9, BATCH_SIZE, device="cuda")

        results = {
            "mean_cond":         cuda_process.mean_cond(x_cuda, t_cuda),
            "multiply_sigma":    cuda_process.multiply_sigma(x_cuda, t_cuda),
            "multiply_cov":      cuda_process.multiply_cov(x_cuda, t_cuda),
            "diffusion_times_v": cuda_process.diffusion_times_v(x_cuda, t_cuda),
        }

        for name, result in results.items():
            assert result.device.type == "cuda", (
                f"{name} output must be on cuda; got {result.device}"
            )
            assert result.shape == (BATCH_SIZE, DATA_DIM), (
                f"{name} output shape must be ({BATCH_SIZE}, {DATA_DIM}); "
                f"got {result.shape}"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_device_query_raises_value_error(
        self, cuda_process, x_batch, t_batch,
    ):
        """Calling mean_cond on a CUDA-fitted process with CPU inputs must raise ValueError.

        The diagonal process has an explicit device guard. This test verifies the
        circulant wrapper does not swallow or bypass it.
        """
        with pytest.raises(ValueError):
            cuda_process.mean_cond(x_batch, t_batch)

"""Reusable mixin classes for diffusion process test suites.

None of the classes below start with 'Test', so pytest never collects them
directly. They are inherited by concrete Test* classes in each test file.

Required fixtures per mixin
-----------------------------
    ForwardDiffusionFitTests         → make_process
    ForwardDiffusionCheckpointTests  → make_process
    ForwardDiffusionOperatorTests    → process, x_batch, t_batch
    ScalarCovarianceOperatorTests    → inherits the above
    DiagonalCovarianceOperatorTests  → inherits the above
    ForwardDiffusionSampleLimitTests → make_process

x_batch and t_batch come from conftest.py; make_process is provided by each
concrete class; process is a fitted instance provided by the concrete class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest
import torch
from sklearn.exceptions import NotFittedError

from skfda.ml.generative._diffusion_process import (
    ForwardDiffusionProcess,
)

from ._constants import BATCH_SIZE, CUSTOM_DIM

# ─────────────────────────────────────────────────────────────────────────────
# Mixin 1: fit / internal serialization hooks
# ─────────────────────────────────────────────────────────────────────────────

class ForwardDiffusionFitTests:
    """Mixin verifying fit() and _get_fit_state/_restore_fit_state contract.

    Safe to inherit for any process, including those with non-picklable
    callables (e.g. DiagonalDiffusionProcess with lambdas). For the full
    public checkpoint API see ForwardDiffusionCheckpointTests.

    Subclasses must provide:
        make_process — zero-argument callable returning a fresh unfitted
                       instance.
    """

    @pytest.fixture
    def unfitted_instance(
        self, make_process: Callable[[], ForwardDiffusionProcess],
    ) -> ForwardDiffusionProcess:
        """Fresh unfitted process instance, recreated for every test."""
        return make_process()

    # ── fit() contract ───────────────────────────────────────────────────────

    def test_no_m_before_fit(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """Attribute 'M_' must not exist before fit() is called."""
        assert not hasattr(unfitted_instance, "M_")

    def test_fit_learns_data_dimension(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """After fit(x), self.M_ must equal x.shape[1] (axis 1, not axis 0)."""
        unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        assert unfitted_instance.M_ == CUSTOM_DIM

    def test_fit_returns_self(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """fit() must return self to enable method chaining."""
        result = unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        assert result is unfitted_instance

    # ── _get_fit_state() contract ────────────────────────────────────────────

    def test_get_fit_state_before_fit_raises(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """_get_fit_state() must raise NotFittedError before fit()."""
        with pytest.raises(NotFittedError):
            unfitted_instance._get_fit_state()  # noqa: SLF001

    def test_get_fit_state_contains_m(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """_get_fit_state() must return a dict with key 'M' == CUSTOM_DIM.

        'M' is the only key guaranteed by the base class.
        """
        unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        state = unfitted_instance._get_fit_state()  # noqa: SLF001

        assert state["M"] == CUSTOM_DIM

    # ── _restore_fit_state() contract ────────────────────────────────────────

    def test_restore_fit_state_missing_m_raises(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """_restore_fit_state({}) must raise ValueError when 'M' is absent."""
        with pytest.raises(ValueError, match="'M' is missing"):
            unfitted_instance._restore_fit_state({})  # noqa: SLF001

    def test_restore_fit_state_wrong_m_type_raises(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """_restore_fit_state({'M': '17'}) must raise TypeError."""
        with pytest.raises(TypeError):
            unfitted_instance._restore_fit_state({"M": "17"})  # noqa: SLF001

    def test_restore_fit_state_round_trip(
        self,
        unfitted_instance: ForwardDiffusionProcess,
        make_process: Callable[[], ForwardDiffusionProcess],
    ) -> None:
        """Fit state from _get_fit_state() must restore into a fresh instance.

        Uses two independent instances to confirm genuine transfer.
        Process-specific extra state belongs in each concrete subclass test.
        """
        instance_a = unfitted_instance
        instance_a.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        instance_b = make_process()
        instance_b._restore_fit_state(  # noqa: SLF001
            instance_a._get_fit_state(),  # noqa: SLF001
        )

        assert instance_b.M_ == instance_a.M_


# ─────────────────────────────────────────────────────────────────────────────
# Mixin 2: full public checkpoint API (picklable callables only)
# ─────────────────────────────────────────────────────────────────────────────

class ForwardDiffusionCheckpointTests:
    """Mixin verifying to_checkpoint() / from_checkpoint().

    Inherit only when the process has picklable callables. VP and VE are safe;
    DiagonalDiffusionProcess must use module-level callables in make_process,
    or omit this mixin and add targeted checkpoint tests instead.

    Subclasses must provide:
        make_process — zero-argument callable returning a fresh unfitted
                       instance.
    """

    @pytest.fixture
    def unfitted_instance(
        self, make_process: Callable[[], ForwardDiffusionProcess],
    ) -> ForwardDiffusionProcess:
        """Fresh unfitted process instance, recreated for every test."""
        return make_process()

    def test_to_checkpoint_before_fit_raises(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """to_checkpoint() must raise NotFittedError before fit()."""
        with pytest.raises(NotFittedError):
            unfitted_instance.to_checkpoint()

    def test_to_checkpoint_returns_required_keys(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """to_checkpoint() must include 'class', 'init_kwargs', 'fit_state'."""
        unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = unfitted_instance.to_checkpoint()

        assert {"class", "init_kwargs", "fit_state"} <= checkpoint.keys()

    def test_to_checkpoint_class_is_concrete_type(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """checkpoint['class'] must be the exact runtime type of the instance.

        from_checkpoint() uses this key for dispatch.
        """
        unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = unfitted_instance.to_checkpoint()

        assert checkpoint["class"] is type(unfitted_instance)

    def test_to_checkpoint_init_kwargs_matches_get_params(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """checkpoint['init_kwargs'] must equal get_params()."""
        unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))

        checkpoint = unfitted_instance.to_checkpoint()

        assert checkpoint["init_kwargs"] == unfitted_instance.get_params()

    def test_from_checkpoint_missing_top_level_keys_raises(self) -> None:
        """from_checkpoint({}) must raise ValueError."""
        with pytest.raises(ValueError, match="missing required keys"):
            ForwardDiffusionProcess.from_checkpoint({})

    def test_from_checkpoint_missing_m_in_fit_state_raises(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """from_checkpoint() must raise ValueError when 'M' is absent.

        The 'M' key must be present in fit_state for restoration to succeed.
        """
        checkpoint = {
            "class": type(unfitted_instance),
            "init_kwargs": unfitted_instance.get_params(),
            "fit_state": {},
        }

        with pytest.raises(ValueError, match="'M' is missing"):
            ForwardDiffusionProcess.from_checkpoint(checkpoint)

    def test_from_checkpoint_wrong_m_type_raises(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """from_checkpoint() must raise TypeError when 'M' has wrong type."""
        checkpoint = {
            "class": type(unfitted_instance),
            "init_kwargs": unfitted_instance.get_params(),
            "fit_state": {"M": "17"},
        }

        with pytest.raises(TypeError):
            ForwardDiffusionProcess.from_checkpoint(checkpoint)

    def test_from_checkpoint_continues_saved_rng_stream(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """A restored process must continue the saved RNG stream, not reset it.

        The checkpoint is taken after the generator has already been advanced
        by one draw, so the next sample must match between the original and the
        restored process. A restore that re-seeds from scratch (instead of
        restoring the generator state) would diverge here.
        """
        unfitted_instance.fit(torch.randn(BATCH_SIZE, CUSTOM_DIM))
        unfitted_instance.sample_limit_distribution(n_samples=4)  # advance RNG

        checkpoint = unfitted_instance.to_checkpoint()
        expected = unfitted_instance.sample_limit_distribution(n_samples=4)

        restored = ForwardDiffusionProcess.from_checkpoint(checkpoint)
        actual = restored.sample_limit_distribution(n_samples=4)

        assert torch.equal(expected, actual)


# ─────────────────────────────────────────────────────────────────────────────
# Mixin 3: operator algebraic contracts (any ForwardDiffusionProcess)
# ─────────────────────────────────────────────────────────────────────────────

class ForwardDiffusionOperatorTests:
    """Mixin for operator algebraic contracts of any ForwardDiffusionProcess.

    Shape-agnostic: checks relationships between multiply_sigma, multiply_cov,
    and multiply_inv_sigma without inspecting _cov values. Valid for scalar
    (N,), diagonal (N, M), and full-matrix (N, M, M) covariance structures.

    Subclasses must provide:
        process  — a fitted ForwardDiffusionProcess instance
        x_batch  — input tensor, shape (N, M)         [from conftest.py]
        t_batch  — interior time steps in (0.1, 0.8)  [from conftest.py]
    """

    def test_multiply_cov_equals_multiply_sigma_applied_twice(
        self,
        process: ForwardDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """Cov(h,t) == sigma(sigma(h,t), t): multiply_cov equals sigma twice.

        Verifies that efficient closed-form implementations of multiply_cov
        (e.g. VP's _cov(t)·h) match the definition Cov = sigma·sigmaᵀ.
        """
        result_cov = process.multiply_cov(x_batch, t_batch)
        result_sigma_twice = process.multiply_sigma(
            process.multiply_sigma(x_batch, t_batch),
            t_batch,
        )

        assert torch.allclose(result_cov, result_sigma_twice)

    def test_sigma_inv_sigma_round_trip(
        self,
        process: ForwardDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """multiply_inv_sigma(multiply_sigma(z, t), t) must recover z.

        t_batch uses interior times (0.1, 0.8) to stay above the numerical
        clamp in multiply_inv_sigma for all covariance shapes.
        """
        z_sigma = process.multiply_sigma(x_batch, t_batch)
        result  = process.multiply_inv_sigma(z_sigma, t_batch)

        assert torch.allclose(result, x_batch)


# ─────────────────────────────────────────────────────────────────────────────
# Mixin 4a: value tests for scalar covariance  _cov → (N,)
# ─────────────────────────────────────────────────────────────────────────────

class ScalarCovarianceOperatorTests(ForwardDiffusionOperatorTests):
    """Mixin adding value tests for processes where _cov returns shape (N,).

    Inherits algebraic tests from ForwardDiffusionOperatorTests and adds:

        multiply_sigma(z, t) = sqrt(_cov(t)) · z  [(N,1) broadcast over (N,M)]
        multiply_cov(h, t)   = _cov(t) · h        [(N,1) broadcast over (N,M)]

    Used by VP and VE diffusion processes.
    """

    def test_multiply_sigma_equals_sqrt_cov_scalar_times_vector(
        self,
        process: ForwardDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """multiply_sigma(z, t) must equal sqrt(_cov(t))·z (scalar broadcast).

        _cov returns (N,) — unsqueeze(1) broadcasts over the M dimension.
        """
        sigma_t  = torch.sqrt(process._cov(t_batch)).unsqueeze(1)  # noqa: SLF001
        expected = sigma_t * x_batch                                 # (N, M)

        result = process.multiply_sigma(x_batch, t_batch)

        assert torch.allclose(result, expected)

    def test_multiply_cov_equals_cov_scalar_times_vector(
        self,
        process: ForwardDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """multiply_cov(h, t) must equal _cov(t)·h (scalar broadcast).

        _cov returns (N,) — unsqueeze(1) broadcasts over the M dimension.
        """
        cov_t    = process._cov(t_batch).unsqueeze(1)  # noqa: SLF001
        expected = cov_t * x_batch                      # (N, M)

        result = process.multiply_cov(x_batch, t_batch)

        assert torch.allclose(result, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Mixin 4b: value tests for diagonal covariance  _cov → (N, M)
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalCovarianceOperatorTests(ForwardDiffusionOperatorTests):
    """Mixin adding value tests for processes where _cov returns shape (N, M).

    Inherits algebraic tests from ForwardDiffusionOperatorTests and adds:

        multiply_sigma(z, t) = sqrt(_cov(t)) ⊙ z  element-wise: (N,M) ⊙ (N,M)
        multiply_cov(h, t)   = _cov(t) ⊙ h        element-wise: (N,M) ⊙ (N,M)

    Used by DiagonalDiffusionProcess.
    """

    def test_multiply_sigma_equals_sqrt_cov_diagonal_times_vector(
        self,
        process: ForwardDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """multiply_sigma(z, t) must equal sqrt(_cov(t)) ⊙ z (diagonal).

        _cov returns (N, M) — element-wise sqrt, no unsqueeze needed.
        """
        cov_diag = process._cov(t_batch)           # noqa: SLF001
        expected = torch.sqrt(cov_diag) * x_batch  # (N, M) element-wise

        result = process.multiply_sigma(x_batch, t_batch)

        assert torch.allclose(result, expected)

    def test_multiply_cov_equals_cov_diagonal_times_vector(
        self,
        process: ForwardDiffusionProcess,
        x_batch: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> None:
        """multiply_cov(h, t) must equal _cov(t) ⊙ h (diagonal)."""
        cov_diag = process._cov(t_batch)  # noqa: SLF001
        expected = cov_diag * x_batch     # (N, M) element-wise

        result = process.multiply_cov(x_batch, t_batch)

        assert torch.allclose(result, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Mixin 5: sample_limit_distribution contract
# ─────────────────────────────────────────────────────────────────────────────

class ForwardDiffusionSampleLimitTests:
    """Mixin verifying the sample_limit_distribution contract.

    Covers output shape, device placement, and random-state behavior for any
    concrete process. Statistical properties of the limit distribution (mean,
    variance, parametric form) are process-specific and belong in each
    concrete subclass.

    Subclasses must provide:
        make_process          — zero-argument callable returning a fresh
                                unfitted instance.

        make_process_alt_seed — zero-argument callable returning a fresh
                                unfitted instance with a different seed B ≠ A.
                                Used by the different-seed reproducibility
                                test.
    """

    # ── derived fixtures ─────────────────────────────────────────────────────

    @pytest.fixture
    def unfitted_instance(
        self, make_process: Callable[[], ForwardDiffusionProcess],
    ) -> ForwardDiffusionProcess:
        """Fresh unfitted instance for pre-fit error tests."""
        return make_process()

    @pytest.fixture
    def fitted_process(
        self,
        make_process: Callable[[], ForwardDiffusionProcess],
        x_batch: torch.Tensor,
    ) -> ForwardDiffusionProcess:
        """Fitted process instance ready for sampling (M = DATA_DIM)."""
        instance = make_process()
        instance.fit(x_batch)
        return instance

    # ── pre-fit guard ────────────────────────────────────────────────────────

    def test_sample_before_fit_raises_error(
        self, unfitted_instance: ForwardDiffusionProcess,
    ) -> None:
        """sample_limit_distribution must raise NotFittedError before fit().

        M is only set by fit(); without it the output dimension is unknown.
        """
        with pytest.raises(NotFittedError):
            unfitted_instance.sample_limit_distribution(n_samples=4)

    # ── output shape ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("n_samples", [1, 8, 64], ids=["n1", "n8", "n64"])
    def test_output_shape(
        self, fitted_process: ForwardDiffusionProcess, n_samples: int,
    ) -> None:
        """Output shape must be (n_samples, M).

        Uses fitted_process.M_ rather than DATA_DIM so the test remains
        correct if the process was fitted on a different spatial dimension.
        """
        result = fitted_process.sample_limit_distribution(n_samples)

        assert result.shape == (n_samples, fitted_process.M_)

    # ── device placement ─────────────────────────────────────────────────────

    def test_output_is_on_cpu_by_default(
        self, fitted_process: ForwardDiffusionProcess,
    ) -> None:
        """Samples must be on CPU when no device argument is passed."""
        result = fitted_process.sample_limit_distribution(n_samples=8)

        assert result.device.type == "cpu"

    # ── reproducibility ──────────────────────────────────────────────────────

    def test_generator_advances_between_calls(
        self, fitted_process: ForwardDiffusionProcess,
    ) -> None:
        """Successive calls on the same process must produce different samples.

        The process advances its internal generator on every draw; if two
        consecutive calls returned identical output the generator would not
        be advancing as expected.
        """
        samples_a = fitted_process.sample_limit_distribution(n_samples=32)
        samples_b = fitted_process.sample_limit_distribution(n_samples=32)

        assert not torch.equal(samples_a, samples_b)

    def test_same_seed_gives_identical_first_sample(
        self,
        make_process: Callable[[], ForwardDiffusionProcess],
        x_batch: torch.Tensor,
    ) -> None:
        """Two freshly constructed processes with equal seeds must agree.

        Reproducibility lives in the seed, not in resetting a shared
        generator.  Both instances start from the same RNG state, so their
        first call must be bit-identical.
        """
        p1 = make_process()
        p1.fit(x_batch)
        p2 = make_process()
        p2.fit(x_batch)

        assert torch.equal(
            p1.sample_limit_distribution(n_samples=32),
            p2.sample_limit_distribution(n_samples=32),
        )

    def test_different_seeds_give_different_first_samples(
        self,
        make_process: Callable[[], ForwardDiffusionProcess],
        make_process_alt_seed: Callable[[], ForwardDiffusionProcess],
        x_batch: torch.Tensor,
    ) -> None:
        """Two fresh processes with different seeds give different first draws.

        Verifies that the seed parameter truly distinguishes independent
        generators; a collision here would mean the seed is being ignored.
        """
        p1 = make_process()
        p1.fit(x_batch)
        p2 = make_process_alt_seed()
        p2.fit(x_batch)

        assert not torch.equal(
            p1.sample_limit_distribution(n_samples=32),
            p2.sample_limit_distribution(n_samples=32),
        )

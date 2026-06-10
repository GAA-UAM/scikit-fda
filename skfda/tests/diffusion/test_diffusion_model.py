"""Correctness tests for FunctionalDiffusionGenerator."""

from __future__ import annotations
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from sklearn.exceptions import NotFittedError

from skfda.ml.generative.diffusion_model import (
    FunctionalDiffusionGenerator,
    _fdatagrid_to_tensor_dataset,
)
from skfda.ml.generative.diffusion_process import (
    DiagonalDiffusionProcess,
    VarianceExplodingDiffusionProcess,
    VariancePreservingDiffusionProcess,
)
from skfda.ml.generative.reverse_diffusion import ReverseDiffusionProcess
from skfda.representation.grid import FDataGrid

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

_N_SAMPLES = 20
_N_GRID_POINTS = 32

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_grid() -> FDataGrid:
    """20 identical sin(2πt) samples on 32 grid points.

    Expected values are computed from data_matrix rather than hardcoded so
    the oracle stays independent of grid resolution.
    """
    t = np.linspace(0, 1, _N_GRID_POINTS)
    values = np.sin(2 * np.pi * t)                              # shape (M,)
    data_matrix = np.tile(values, (_N_SAMPLES, 1)).reshape(     # (N, M, 1)
        _N_SAMPLES, _N_GRID_POINTS, 1,
    )
    return FDataGrid(data_matrix=data_matrix, grid_points=[t])


@pytest.fixture(scope="module")
def constant_grid() -> FDataGrid:
    """Every sample is f(t) = 3.0; exercises the _scale_=1.0 clamping guard."""
    t = np.linspace(0, 1, _N_GRID_POINTS)
    data_matrix = np.full((_N_SAMPLES, _N_GRID_POINTS, 1), 3.0)
    return FDataGrid(data_matrix=data_matrix, grid_points=[t])


@pytest.fixture(scope="module")
def labels_array() -> np.ndarray:
    """Alternating labels [0, 1, …] of length _N_SAMPLES, dtype float32.

    float32 matches the .float() cast inside _fdatagrid_to_tensor_dataset.
    """
    return np.array([i % 2 for i in range(_N_SAMPLES)], dtype=np.float32)


@pytest.fixture
def fitted_generator(small_grid: FDataGrid) -> FunctionalDiffusionGenerator:
    """FunctionalDiffusionGenerator fitted on small_grid with max_iter=1.

    Function-scoped so each test gets an independent instance; shared mutable
    state (e.g. RNG advancement) would cause test-order-dependent failures.
    """
    gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)
    gen.fit(small_grid)
    return gen


@pytest.fixture
def tmp_checkpoint(tmp_path):
    """Path to a temporary checkpoint file, fresh per test."""
    return tmp_path / "model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level test helpers
# ─────────────────────────────────────────────────────────────────────────────


class _EmptyModule(torch.nn.Module):
    """Score model stub with no trainable parameters.

    Triggers the 'at least one trainable parameter' guard in fit().
    forward() is never called; fit() raises before the training loop.
    """

    def forward(self, x, t, y=None):  # noqa: ARG002
        return x


class _StubReverseProcess(ReverseDiffusionProcess):
    """Reverse diffusion stub that always returns torch.zeros_like(x_t).

    Attributes:
        call_count: number of times reverse() has been called.
        grad_enabled_during_call: torch.is_grad_enabled() value at each call.
    """

    def __init__(self) -> None:
        # Skip super().__init__(): ReverseDiffusionProcess requires an integrator
        # argument that is irrelevant for a stub.
        self.call_count: int = 0
        self.grad_enabled_during_call: list[bool] = []

    def reverse(self, diff_process, *, score_model, x_t, t_1, y=None, **kwargs):
        self.call_count += 1
        self.grad_enabled_during_call.append(torch.is_grad_enabled())
        return torch.zeros_like(x_t)


class _SpyingVP(VariancePreservingDiffusionProcess):
    """VP subclass that records per-call sample counts passed to fit().

    Delegates to the parent so gen.fit() can complete without error.

    Attribute:
        fit_call_sample_counts: list[int] — per-call sample counts in order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_call_sample_counts: list[int] = []

    def fit(self, x):  # type: ignore[override]
        self.fit_call_sample_counts.append(x.shape[0])
        return super().fit(x)


# ─────────────────────────────────────────────────────────────────────────────
# TestInitialization
# ─────────────────────────────────────────────────────────────────────────────


class TestInitialization:
    """Tests for FunctionalDiffusionGenerator.__init__."""

    # ─────────────────────────────────────────────────────────────────────────
    # Default and custom diff_process
    # ─────────────────────────────────────────────────────────────────────────

    def test_default_diff_process_is_vp(self):
        """Default diff_process must be VariancePreservingDiffusionProcess."""
        gen = FunctionalDiffusionGenerator()

        assert isinstance(gen.diff_process, VariancePreservingDiffusionProcess)

    def test_custom_diff_process_is_stored_by_identity(self):
        """A provided diff_process must be stored as the exact same object."""
        ve = VarianceExplodingDiffusionProcess(
            g_schedule="exponential",
            g_0=0.1,
            g_T=15.0,
        )
        gen = FunctionalDiffusionGenerator(diff_process=ve)

        assert gen.diff_process is ve

    # ─────────────────────────────────────────────────────────────────────────
    # normalize / standardize mutual exclusion
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_and_standardize_both_true_raises(self):
        """normalize=True and standardize=True must raise ValueError."""
        with pytest.raises(ValueError):
            FunctionalDiffusionGenerator(normalize=True, standardize=True)

    def test_normalize_false_standardize_false_accepted(self):
        """Both flags False (raw-data mode) must be accepted without error."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        assert gen.normalize is False
        assert gen.standardize is False

    # ─────────────────────────────────────────────────────────────────────────
    # Verbatim storage of all constructor args
    # ─────────────────────────────────────────────────────────────────────────

    def test_constructor_args_stored_verbatim(self):
        """Every constructor argument must be stored as the exact value passed in.

        Non-default values are used for every parameter to rule out coincidental
        matches with defaults. diff_process and score_model are excluded; they
        are covered by the dedicated identity and None-storage tests.
        """
        gen = FunctionalDiffusionGenerator(
            normalize=False,
            standardize=True,
            max_iter=5,
            batch_size=8,
            n_jobs=2,
            device="cpu",
            seed=7,
        )

        assert gen.normalize is False
        assert gen.standardize is True
        assert gen.max_iter == 5
        assert gen.batch_size == 8
        assert gen.n_jobs == 2
        assert gen.device == "cpu"
        assert gen.seed == 7

    # ─────────────────────────────────────────────────────────────────────────
    # score_model=None
    # ─────────────────────────────────────────────────────────────────────────

    def test_score_model_none_stored_as_none(self):
        """score_model=None must be stored as None; built lazily in fit()."""
        gen = FunctionalDiffusionGenerator(score_model=None)

        assert gen.score_model is None


# ─────────────────────────────────────────────────────────────────────────────
# TestPreprocess
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocess:
    """Tests for FunctionalDiffusionGenerator._preprocess."""

    # ─────────────────────────────────────────────────────────────────────────
    # normalize path — _bias_ and _scale_
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_bias_is_midpoint(self, small_grid):
        """_bias_ must equal (max + min) / 2."""
        gen = FunctionalDiffusionGenerator(normalize=True)
        expected_bias = (
            small_grid.data_matrix.max() + small_grid.data_matrix.min()
        ) / 2

        gen._preprocess(small_grid)

        assert gen._bias_ == pytest.approx(expected_bias)

    def test_normalize_scale_is_half_range(self, small_grid):
        """_scale_ must equal (max − min) / 2."""
        gen = FunctionalDiffusionGenerator(normalize=True)
        expected_scale = (
            small_grid.data_matrix.max() - small_grid.data_matrix.min()
        ) / 2

        gen._preprocess(small_grid)

        assert gen._scale_ == pytest.approx(expected_scale)

    # ─────────────────────────────────────────────────────────────────────────
    # standardize path — _bias_ and _scale_
    # ─────────────────────────────────────────────────────────────────────────

    def test_standardize_bias_is_mean(self, small_grid):
        """_bias_ must equal the global scalar mean."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=True)
        expected_bias = float(small_grid.data_matrix.mean())

        gen._preprocess(small_grid)

        assert gen._bias_ == pytest.approx(expected_bias)

    def test_standardize_scale_is_std(self, small_grid):
        """_scale_ must equal the global std (ddof=0)."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=True)
        expected_scale = float(small_grid.data_matrix.std())

        gen._preprocess(small_grid)

        assert gen._scale_ == pytest.approx(expected_scale)

    # ─────────────────────────────────────────────────────────────────────────
    # no-transform path — _bias_ and _scale_
    # ─────────────────────────────────────────────────────────────────────────

    def test_no_transform_bias_zero_scale_one(self, small_grid):
        """With both flags False, _bias_=0.0 and _scale_=1.0 (identity transform)."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        gen._preprocess(small_grid)

        assert gen._bias_ == 0.0
        assert gen._scale_ == 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Scale clamping guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_constant_data_scale_clamped_to_one(self, constant_grid):
        """When (max − min) / 2 < eps, _scale_ must be clamped to 1.0."""
        gen = FunctionalDiffusionGenerator(normalize=True)

        gen._preprocess(constant_grid)

        assert gen._scale_ == 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Return value — type and length
    # ─────────────────────────────────────────────────────────────────────────

    def test_returns_tensor_dataset(self, small_grid):
        """_preprocess must return a TensorDataset of length n_samples."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        dataset = gen._preprocess(small_grid)

        assert isinstance(dataset, TensorDataset)
        assert len(dataset) == small_grid.n_samples

    # ─────────────────────────────────────────────────────────────────────────
    # Data tensor — dtype and shape
    # ─────────────────────────────────────────────────────────────────────────

    def test_data_tensor_is_float32(self, small_grid):
        """The data tensor must be float32."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        dataset = gen._preprocess(small_grid)

        assert dataset.tensors[0].dtype == torch.float32

    def test_data_tensor_shape_is_n_samples_by_n_points(self, small_grid):
        """Data tensor must have shape (N, M), not (N, M, 1)."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)
        expected_shape = (small_grid.n_samples, len(small_grid.grid_points[0]))

        dataset = gen._preprocess(small_grid)

        assert dataset.tensors[0].shape == expected_shape

    # ─────────────────────────────────────────────────────────────────────────
    # Labeled data — two-tensor dataset
    # ─────────────────────────────────────────────────────────────────────────

    def test_with_labels_dataset_has_two_tensors(self, small_grid, labels_array):
        """Passing y must produce a TensorDataset with exactly two tensors."""
        gen = FunctionalDiffusionGenerator(normalize=False, standardize=False)

        dataset = gen._preprocess(small_grid, y=labels_array)

        assert len(dataset.tensors) == 2
        assert dataset.tensors[1].shape == (small_grid.n_samples,)

    # ─────────────────────────────────────────────────────────────────────────
    # Mutation guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_preprocess_does_not_mutate_original_fdatagrid(self, small_grid):
        """_preprocess must not modify the input FDataGrid in place."""
        data_matrix_before = small_grid.data_matrix.copy()
        gen = FunctionalDiffusionGenerator(normalize=True)

        gen._preprocess(small_grid)

        assert np.array_equal(small_grid.data_matrix, data_matrix_before)


# ─────────────────────────────────────────────────────────────────────────────
# TestFit
# ─────────────────────────────────────────────────────────────────────────────


class TestFit:
    """Tests for FunctionalDiffusionGenerator.fit."""

    # ─────────────────────────────────────────────────────────────────────────
    # scikit-learn convention
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self, small_grid):
        """fit() must return self (scikit-learn chaining convention)."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)

        result = gen.fit(small_grid)

        assert result is gen

    # ─────────────────────────────────────────────────────────────────────────
    # Input validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_raises_for_multidimensional_domain(self):
        """fit() must raise ValueError for dim_domain > 1."""
        t = np.linspace(0, 1, 4)
        data_matrix = np.zeros((2, 4, 4, 1))  # (N, M1, M2, codomain)
        grid_2d = FDataGrid(data_matrix=data_matrix, grid_points=[t, t])
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)

        with pytest.raises(ValueError):
            gen.fit(grid_2d)

    def test_fit_raises_for_multidimensional_codomain(self):
        """fit() must raise ValueError for dim_codomain > 1."""
        t = np.linspace(0, 1, 8)
        data_matrix = np.zeros((2, 8, 2))  # (N, M, codomain=2)
        grid_vector = FDataGrid(data_matrix=data_matrix, grid_points=[t])
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)

        with pytest.raises(ValueError):
            gen.fit(grid_vector)

    # ─────────────────────────────────────────────────────────────────────────
    # Post-fit attributes
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_sets_grid_points_attribute(self, small_grid):
        """fit() must store the training grid as grid_points_."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)

        gen.fit(small_grid)

        assert hasattr(gen, "grid_points_")
        assert np.array_equal(gen.grid_points_, small_grid.grid_points[0])

    def test_fit_sets_score_model_attribute(self, small_grid):
        """fit() must store the trained network as a torch.nn.Module in score_model_."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)

        gen.fit(small_grid)

        assert hasattr(gen, "score_model_")
        assert isinstance(gen.score_model_, torch.nn.Module)

    def test_score_model_in_eval_mode_after_fit(self, small_grid):
        """score_model_ must be in eval mode when fit() returns.

        generate() does not call .eval() itself, so this must be set by fit().
        """
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=42)

        gen.fit(small_grid)

        assert gen.score_model_.training is False

    # ─────────────────────────────────────────────────────────────────────────
    # score_model validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_raises_if_custom_score_model_not_nn_module(self, small_grid):
        """fit() must raise TypeError when score_model is not a torch.nn.Module."""
        gen = FunctionalDiffusionGenerator(score_model="not_a_module")

        with pytest.raises(TypeError):
            gen.fit(small_grid)

    def test_fit_raises_if_score_model_has_no_parameters(self, small_grid):
        """fit() must raise ValueError when the score model has no trainable parameters."""
        gen = FunctionalDiffusionGenerator(score_model=_EmptyModule())

        with pytest.raises(ValueError, match="trainable parameter"):
            gen.fit(small_grid)

    # ─────────────────────────────────────────────────────────────────────────
    # Labeled data
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_with_labels_does_not_raise(self, small_grid, labels_array):
        """fit() must complete without error when y labels are supplied."""
        gen = FunctionalDiffusionGenerator(max_iter=1, seed=0)

        gen.fit(small_grid, y=labels_array)  # must not raise

    # ─────────────────────────────────────────────────────────────────────────
    # diff_process.fit receives all samples
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_diff_process_receives_full_dataset(self, small_grid):
        """diff_process.fit() must be called with all N training samples."""
        spy = _SpyingVP(beta_schedule="linear", beta_min=0.1, beta_max=10.0)
        gen = FunctionalDiffusionGenerator(
            diff_process=spy,
            max_iter=1,
            batch_size=4,
            seed=42,
        )

        gen.fit(small_grid)

        total_samples_seen = sum(spy.fit_call_sample_counts)
        assert total_samples_seen == _N_SAMPLES  # 20; bug yields 4

    # ─────────────────────────────────────────────────────────────────────────
    # Reproducibility
    # ─────────────────────────────────────────────────────────────────────────

    def test_fit_is_reproducible_with_same_seed(self, small_grid):
        """Same seed must produce identical weights after fit()."""
        gen_a = FunctionalDiffusionGenerator(max_iter=1, seed=42)
        gen_b = FunctionalDiffusionGenerator(max_iter=1, seed=42)
        gen_a.fit(small_grid)
        gen_b.fit(small_grid)

        for param_a, param_b in zip(
            gen_a.score_model_.parameters(),
            gen_b.score_model_.parameters(),
        ):
            assert torch.equal(param_a, param_b)

# ─────────────────────────────────────────────────────────────────────────────
# TestLossFunction
# ─────────────────────────────────────────────────────────────────────────────

_LOSS_BATCH = 4  # small batch — cheap while still exercising batch reductions


class TestLossFunction:
    """Tests for FunctionalDiffusionGenerator._loss_function."""

    # ─────────────────────────────────────────────────────────────────────────
    # Output value
    # ─────────────────────────────────────────────────────────────────────────

    def test_loss_is_non_negative(self, fitted_generator):
        """Loss must be non-negative (mean of squared norms)."""
        gen = fitted_generator
        x = torch.randn(_LOSS_BATCH, _N_GRID_POINTS)
        t = torch.full((_LOSS_BATCH,), 0.5)
        generator = torch.Generator()
        generator.manual_seed(0)

        loss = gen._loss_function(gen.score_model_, x, t, None, generator)

        assert loss.item() >= 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Output shape
    # ─────────────────────────────────────────────────────────────────────────

    def test_loss_returns_scalar_tensor(self, fitted_generator):
        """Loss must be a 0-dimensional tensor."""
        gen = fitted_generator
        x = torch.randn(_LOSS_BATCH, _N_GRID_POINTS)
        t = torch.full((_LOSS_BATCH,), 0.5)
        generator = torch.Generator()
        generator.manual_seed(0)

        loss = gen._loss_function(gen.score_model_, x, t, None, generator)

        assert loss.shape == torch.Size([])

    # ─────────────────────────────────────────────────────────────────────────
    # Time-dependence
    # ─────────────────────────────────────────────────────────────────────────

    def test_loss_changes_with_different_t(self, fitted_generator):
        """Loss must differ for different t values with all other inputs fixed."""
        gen = fitted_generator
        x = torch.randn(_LOSS_BATCH, _N_GRID_POINTS)

        t_low  = torch.full((_LOSS_BATCH,), 0.1)
        t_high = torch.full((_LOSS_BATCH,), 0.9)

        # Same seed for both generators so z is identical; t is the only difference.
        gen_low  = torch.Generator()
        gen_low.manual_seed(0)
        gen_high = torch.Generator()
        gen_high.manual_seed(0)

        loss_low  = gen._loss_function(gen.score_model_, x, t_low,  None, gen_low)
        loss_high = gen._loss_function(gen.score_model_, x, t_high, None, gen_high)

        assert loss_low.item() != loss_high.item(), (
            f"Loss was identical at t=0.1 and t=0.9 ({loss_low.item():.6f}), "
            "suggesting multiply_sigma does not depend on t."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerate
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerate:
    """Tests for FunctionalDiffusionGenerator.generate."""

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-fit guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_raises_if_not_fitted(self):
        """generate() must raise NotFittedError when called before fit()."""
        gen = FunctionalDiffusionGenerator()

        with pytest.raises(NotFittedError):
            gen.generate(n_samples=5)

    # ─────────────────────────────────────────────────────────────────────────
    # Return type and structure
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_returns_fdatagrid(self, fitted_generator):
        """generate() must return an FDataGrid."""
        result = fitted_generator.generate(n_samples=5)

        assert isinstance(result, FDataGrid)

    def test_generate_n_samples_respected(self, fitted_generator):
        """generate() must produce exactly n_samples functional observations."""
        result = fitted_generator.generate(n_samples=7)

        assert result.n_samples == 7

    def test_generate_dim_domain_and_codomain(self, fitted_generator):
        """Generated data must have dim_domain=1, dim_codomain=1."""
        result = fitted_generator.generate(n_samples=3)

        assert result.dim_domain == 1
        assert result.dim_codomain == 1

    def test_generate_grid_points_match_training(self, fitted_generator, small_grid):
        """Generated data must be discretized on the same grid as training data."""
        result = fitted_generator.generate(n_samples=3)

        assert np.array_equal(result.grid_points[0], small_grid.grid_points[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Conditional generation
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_with_y_overrides_n_samples(self, fitted_generator):
        """When y is provided, n_samples must equal len(y)."""
        y = np.array([0, 1, 0])

        result = fitted_generator.generate(n_samples=99, y=y)

        assert result.n_samples == len(y)  # 3, not 99

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient tracking
    # ─────────────────────────────────────────────────────────────────────────

    def test_generate_no_gradient_tracking(self, fitted_generator):
        """The reverse pass must run inside torch.no_grad()."""
        stub = _StubReverseProcess()

        fitted_generator.generate(n_samples=3, reverse_process=stub)

        assert stub.call_count >= 1, "reverse() was never called"
        assert not any(stub.grad_enabled_during_call), (
            "Gradient tracking was enabled during at least one reverse() call; "
            "torch.no_grad() context may have been removed from generate()."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Custom reverse process
    # ─────────────────────────────────────────────────────────────────────────

    def test_custom_reverse_process_is_used(self, fitted_generator):
        """generate() must delegate to the caller-supplied reverse_process."""
        stub = _StubReverseProcess()

        fitted_generator.generate(n_samples=3, reverse_process=stub)

        assert stub.call_count == 1

    # ─────────────────────────────────────────────────────────────────────────
    # Inverse transform — normalize path
    # ─────────────────────────────────────────────────────────────────────────

    def test_normalize_inverse_transform_applied(self):
        """generate() must apply the inverse normalization transform.

        Fit on sin(2πt)+5 so _bias_=5.0, _scale_=1.0. Stub returns zeros
        (normalized space), so expected output is 0.0 * 1.0 + 5.0 = 5.0.
        """
        t = np.linspace(0, 1, _N_GRID_POINTS)
        values = np.sin(2 * np.pi * t) + 5.0  # range ≈ [4, 6], _bias_ = 5.0
        data_matrix = np.tile(values, (_N_SAMPLES, 1)).reshape(
            _N_SAMPLES, _N_GRID_POINTS, 1,
        )
        shifted_grid = FDataGrid(data_matrix=data_matrix, grid_points=[t])

        gen = FunctionalDiffusionGenerator(
            normalize=True, max_iter=1, seed=0,
        )
        gen.fit(shifted_grid)

        stub = _StubReverseProcess()  # reverse returns zeros
        result = gen.generate(n_samples=2, reverse_process=stub)

        assert np.allclose(result.data_matrix, gen._bias_, atol=1e-5), (
            f"Expected all values ≈ _bias_={gen._bias_:.4f} after inverse "
            f"normalization, got min={result.data_matrix.min():.4f} "
            f"max={result.data_matrix.max():.4f}. "
            "The inverse transform may not have been applied."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Inverse transform — no-transform path
    # ─────────────────────────────────────────────────────────────────────────

    def test_no_transform_generate_does_not_rescale(self, small_grid):
        """No rescaling when normalize=False and standardize=False."""
        gen = FunctionalDiffusionGenerator(
            normalize=False, standardize=False, max_iter=1, seed=0,
        )
        gen.fit(small_grid)

        stub = _StubReverseProcess()  # reverse returns zeros
        result = gen.generate(n_samples=1, reverse_process=stub)

        assert np.allclose(result.data_matrix, 0.0, atol=1e-6), (
            "Expected all-zero data_matrix when normalize=False and "
            "standardize=False, but values were non-zero. "
            "The rescaling block may have been entered incorrectly."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestGenerateEvolution
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateEvolution:
    """Tests for FunctionalDiffusionGenerator.generate_evolution.

    The result has exactly len(timesteps) snapshots: result[0] is the initial
    noise, result[-1] is the final denoised state.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-fit guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_raises_if_not_fitted(self):
        """generate_evolution() must raise NotFittedError before fit() is called."""
        gen = FunctionalDiffusionGenerator()

        with pytest.raises(NotFittedError):
            gen.generate_evolution(5, timesteps=np.linspace(1.0, 0.0, 5))

    # ─────────────────────────────────────────────────────────────────────────
    # Return type and structure
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_returns_list_of_fdatagrid(self, fitted_generator):
        """generate_evolution() must return a list of FDataGrid objects."""
        result = fitted_generator.generate_evolution(
            3, timesteps=np.linspace(1.0, 0.0, 5),
        )

        assert isinstance(result, list)
        assert all(isinstance(r, FDataGrid) for r in result)

    def test_evolution_length_matches_timesteps(self, fitted_generator):
        """The result list must have exactly len(timesteps) elements."""
        timesteps = np.linspace(1.0, 0.0, 6)

        result = fitted_generator.generate_evolution(3, timesteps=timesteps)

        assert len(result) == len(timesteps)

    def test_evolution_each_fdatagrid_has_correct_n_samples(self, fitted_generator):
        """Every snapshot must contain exactly the requested number of samples."""
        n_samples = 5
        result = fitted_generator.generate_evolution(
            n_samples, timesteps=np.linspace(1.0, 0.0, 4),
        )

        assert all(r.n_samples == n_samples for r in result)

    # ─────────────────────────────────────────────────────────────────────────
    # Inverse transform on post-reverse snapshots
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_inverse_transform_applied_to_all_elements(self):
        """Inverse transform must be applied to every snapshot, not just the last.

        result[0] is the initial noise (before any reverse call) and is
        excluded from the assertion; result[1:] are the post-reverse snapshots.
        """
        t = np.linspace(0, 1, _N_GRID_POINTS)
        values = np.sin(2 * np.pi * t) + 5.0   # range ≈ [4, 6], _bias_ = 5.0
        data_matrix = np.tile(values, (_N_SAMPLES, 1)).reshape(
            _N_SAMPLES, _N_GRID_POINTS, 1,
        )
        shifted_grid = FDataGrid(data_matrix=data_matrix, grid_points=[t])

        gen = FunctionalDiffusionGenerator(
            normalize=True, max_iter=1, seed=0,
        )
        gen.fit(shifted_grid)

        stub = _StubReverseProcess()  # returns zeros for every call
        result = gen.generate_evolution(
            2, timesteps=np.linspace(1.0, 0.0, 3), reverse_process=stub,
        )

        for idx, snapshot in enumerate(result[1:], start=1):
            assert np.allclose(snapshot.data_matrix, gen._bias_, atol=1e-5), (
                f"result[{idx}].data_matrix is not ≈ _bias_={gen._bias_:.4f}; "
                "the inverse transform may not have been applied to this snapshot."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Conditional generation
    # ─────────────────────────────────────────────────────────────────────────

    def test_evolution_with_y_sets_n_samples(self, fitted_generator):
        """When y is provided, every snapshot must have n_samples == len(y)."""
        y = np.array([0, 1])

        result = fitted_generator.generate_evolution(
            99, timesteps=np.linspace(1.0, 0.0, 3), y=y,
        )

        assert all(r.n_samples == len(y) for r in result)  # 2, not 99


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveLoad
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoad:
    """Tests for FunctionalDiffusionGenerator.save and .load."""

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-fit guard
    # ─────────────────────────────────────────────────────────────────────────

    def test_save_raises_if_not_fitted(self, tmp_checkpoint):
        """save() must raise NotFittedError when called before fit()."""
        gen = FunctionalDiffusionGenerator()

        with pytest.raises(NotFittedError):
            gen.save(tmp_checkpoint)

    # ─────────────────────────────────────────────────────────────────────────
    # File creation
    # ─────────────────────────────────────────────────────────────────────────

    def test_save_creates_file(self, fitted_generator, tmp_checkpoint):
        """save() must create a file at the given path."""
        fitted_generator.save(tmp_checkpoint)

        assert tmp_checkpoint.exists()

    # ─────────────────────────────────────────────────────────────────────────
    # Atomic write
    # ─────────────────────────────────────────────────────────────────────────

    def test_save_is_atomic_on_failure(self, fitted_generator, tmp_path, monkeypatch):
        """A failed save must leave neither the target file nor a .tmp file.

        Patches torch.save to write the .tmp file and then raise OSError,
        reproducing a disk-full error after partial I/O.
        """
        target = tmp_path / "model.pt"
        _real_torch_save = torch.save

        def _write_then_fail(obj, path, **kwargs):
            _real_torch_save(obj, path, **kwargs)
            raise OSError("simulated disk full after write")

        monkeypatch.setattr(torch, "save", _write_then_fail)

        with pytest.raises(OSError):
            fitted_generator.save(target)

        assert not target.exists(), "target file must not exist after a failed save"
        assert not target.with_suffix(".tmp").exists(), (
            ".tmp file must be cleaned up after a failed save"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # load() — format validation
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_raises_for_invalid_checkpoint_format(self, tmp_checkpoint):
        """load() must raise ValueError when the file contains a non-dict object."""
        torch.save("not_a_dict", tmp_checkpoint)

        with pytest.raises(TypeError, match="Invalid checkpoint format"):
            FunctionalDiffusionGenerator.load(tmp_checkpoint)

    def test_load_raises_for_missing_keys(self, tmp_checkpoint):
        """load() must raise ValueError when required keys are absent."""
        torch.save({"version": 1}, tmp_checkpoint)

        with pytest.raises(ValueError, match="missing required keys"):
            FunctionalDiffusionGenerator.load(tmp_checkpoint)

    # ─────────────────────────────────────────────────────────────────────────
    # Round-trip — attribute preservation
    # ─────────────────────────────────────────────────────────────────────────

    def test_roundtrip_grid_points_preserved(self, fitted_generator, tmp_checkpoint):
        """grid_points_ must survive a save → load round-trip unchanged."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        assert np.array_equal(gen2.grid_points_, fitted_generator.grid_points_)

    def test_roundtrip_bias_and_scale_preserved(self, fitted_generator, tmp_checkpoint):
        """_bias_ and _scale_ must survive a save → load round-trip unchanged."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        assert gen2._bias_ == fitted_generator._bias_
        assert gen2._scale_ == fitted_generator._scale_

    def test_roundtrip_normalize_flag_preserved(self, fitted_generator, tmp_checkpoint):
        """The normalize flag must survive a save → load round-trip unchanged."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        assert gen2.normalize is True

    # ─────────────────────────────────────────────────────────────────────────
    # Round-trip — generate() works after load
    # ─────────────────────────────────────────────────────────────────────────

    def test_roundtrip_can_generate_after_load(self, fitted_generator, tmp_checkpoint):
        """A loaded generator must be able to call generate() without error."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        result = gen2.generate(n_samples=3)

        assert isinstance(result, FDataGrid)
        assert result.n_samples == 3

    # ─────────────────────────────────────────────────────────────────────────
    # Device override
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_device_override(self, fitted_generator, tmp_checkpoint):
        """load(device='cpu') must place all score model parameters on CPU."""
        fitted_generator.save(tmp_checkpoint)
        gen2 = FunctionalDiffusionGenerator.load(tmp_checkpoint, device="cpu")

        for param in gen2.score_model_.parameters():
            assert param.device.type == "cpu", (
                f"Parameter on {param.device} after load(device='cpu'); "
                "device override may not have been applied to the score model."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # RNG state reproducibility
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_restores_rng_state_for_reproducibility(
        self, fitted_generator, tmp_checkpoint,
    ):
        """Two loads from the same checkpoint must yield identical outputs."""
        fitted_generator.save(tmp_checkpoint)

        gen_a = FunctionalDiffusionGenerator.load(tmp_checkpoint)
        gen_b = FunctionalDiffusionGenerator.load(tmp_checkpoint)

        result_a = gen_a.generate(n_samples=4)
        result_b = gen_b.generate(n_samples=4)

        assert np.array_equal(result_a.data_matrix, result_b.data_matrix), (
            "generate() produced different outputs from two loads of the same "
            "checkpoint. The RNG state stored in the checkpoint may not have "
            "been correctly restored."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # diff_process instance injection
    # ─────────────────────────────────────────────────────────────────────────

    def test_load_with_diff_process_instance_restores_state(
        self, small_grid, tmp_checkpoint,
    ):
        """load(diff_process=...) must restore fitted state into the supplied instance.

        DiagonalDiffusionProcess stores a non-picklable callable, so load() must
        graft the checkpoint's numerical state onto the fresh instance.
        """
        drift_term = lambda t: torch.ones(t.shape[0], _N_GRID_POINTS)
        diff_term = lambda t: torch.ones(t.shape[0], _N_GRID_POINTS)
        proc = DiagonalDiffusionProcess(drift_term=drift_term, diffusion_term=diff_term)
        gen = FunctionalDiffusionGenerator(
            diff_process=proc, max_iter=1, seed=0,
        )
        gen.fit(small_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen.save(tmp_checkpoint)

        # Reconstruct with a fresh process carrying the same callable.
        fresh_proc = DiagonalDiffusionProcess(drift_term=drift_term, diffusion_term=diff_term)
        gen2 = FunctionalDiffusionGenerator.load(
            tmp_checkpoint, diff_process=fresh_proc,
        )

        x_test = torch.randn(2, gen2.diff_process.M)
        t_test = torch.full((2,), 0.5)
        gen2.diff_process.mean_cond(x_test, t_test)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# TestFromFDataGridToTensorDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestFromFDataGridToTensorDataset:
    """Tests for the module-level utility _fdatagrid_to_tensor_dataset."""

    # ─────────────────────────────────────────────────────────────────────────
    # Number of tensors in the dataset
    # ─────────────────────────────────────────────────────────────────────────

    def test_without_labels_returns_single_tensor_dataset(self, small_grid):
        """Without y, the dataset must contain exactly one tensor."""
        ds = _fdatagrid_to_tensor_dataset(small_grid)

        assert len(ds.tensors) == 1

    def test_with_labels_returns_two_tensor_dataset(self, small_grid, labels_array):
        """With y, the dataset must contain exactly two tensors: data and labels."""
        ds = _fdatagrid_to_tensor_dataset(small_grid, y=labels_array)

        assert len(ds.tensors) == 2

    # ─────────────────────────────────────────────────────────────────────────
    # Data tensor — shape and dtype
    # ─────────────────────────────────────────────────────────────────────────

    def test_data_tensor_shape(self, small_grid):
        """Data tensor must have shape (N, M), not (N, M, 1)."""
        ds = _fdatagrid_to_tensor_dataset(small_grid)

        expected_shape = (small_grid.n_samples, len(small_grid.grid_points[0]))
        assert ds.tensors[0].shape == expected_shape

    def test_data_tensor_dtype_float32(self, small_grid):
        """The data tensor must be float32."""
        ds = _fdatagrid_to_tensor_dataset(small_grid)

        assert ds.tensors[0].dtype == torch.float32

    # ─────────────────────────────────────────────────────────────────────────
    # Label tensor — values and shape
    # ─────────────────────────────────────────────────────────────────────────

    def test_labels_tensor_matches_input(self, small_grid, labels_array):
        """The label tensor must contain exactly the values passed in y."""
        ds = _fdatagrid_to_tensor_dataset(small_grid, y=labels_array)

        expected = torch.from_numpy(labels_array).float()
        assert torch.allclose(ds.tensors[1], expected)

    def test_labels_tensor_shape(self, small_grid, labels_array):
        """The label tensor must have shape (N,), not (N, 1) or (1, N)."""
        ds = _fdatagrid_to_tensor_dataset(small_grid, y=labels_array)

        assert ds.tensors[1].shape == (small_grid.n_samples,)

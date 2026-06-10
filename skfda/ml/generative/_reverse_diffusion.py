"""Reverse diffusion processes and integrators for score-based models."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import Protocol

import torch
from torch import Tensor

from ._diffusion_process import (
    CirculantSymmetricMatrixDiffusionProcess,
    CustomDiffusionProcess,
    DiffusionProcess,
    ForwardDiffusionProcess,
    _eigenspace_to_fft,
    _fft_to_eigenspace,
)
from ._score_model import ScoreModel


class ReverseDiffusionProcess(ABC):
    """Protocol for the reverse process of a diffusion model.

    It defines the method `reverse` that takes a diffusion process,
    a score model, a noisy sample at time t_1, its label and it returns
    the denoised sample at time t_0.
    """
    _T0_SAFE: float = 1e-3

    def _tweedie_denoise(
        self,
        diff_process: ForwardDiffusionProcess,
        score_model: ScoreModel,
        x_t0: Tensor,
        x_t: Tensor,
        t_safe: Tensor | float,
        y: Tensor | None = None,
    ) -> Tensor:
        r"""Apply Tweedie's formula as a final denoising step.

        .. math::

            \mathbb{E}[x_0 \mid x_t] =
            \frac{x_t + \Sigma_t\,s_\theta(x_t,\,t)}{a(t)}

        where :math:`\Sigma_t = \operatorname{Cov}[X(t) \mid X(0)]` and
        :math:`a(t)` is the diagonal mean-scaling from
        :py:meth:`~ForwardDiffusionProcess.mean_cond`.

        Only valid for processes where :py:meth:`mean_cond` is diagonal in the
        canonical basis; circulant processes should project to eigenspace
        before calling this via their registered overload.

        Args:
            diff_process: The forward diffusion process.
            score_model: The score model :math:`s_\theta(x, t)`.
            x_t0: The noisy sample at time ``t_safe``, shape (N, M).
            x_t: Sample used to infer the batch size for the time tensor.
            t_safe: The (safe, non-zero) time at which denoising is applied.
            y: Optional conditioning labels, shape (N, C).

        Returns:
            The Tweedie-denoised estimate of :math:`x_0`, shape (N, M).
        """
        t_tensor = torch.ones_like(x_t[:, 0]) * t_safe
        score = score_model(x_t0, t_tensor, y)
        cov_score = diff_process.multiply_cov(score, t_tensor)
        ones = torch.ones_like(x_t0)
        mu_t0_scale = diff_process.mean_cond(ones, t_tensor)
        return (x_t0 + cov_score) / mu_t0_scale.clamp(min=1e-5)

    @abstractmethod
    def reverse(
        self,
        diffusion_process: ForwardDiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        t_0: Tensor | float = 1e-3,
        y: Tensor | None = None,
    ) -> Tensor:
        """Reverse the diffusion process from time t_1 to time t_0.

        Args:
            diffusion_process: The diffusion process defining the forward SDE.
            score_model: The score-based model used to estimate the score
            function.
            x_t: The noisy sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The time step of the noisy sample, shape (N,).
            t_0: The time step to integrate back to, default is 1e-3.
            y: Optional labels for conditional generation, shape (N, C).

        Returns:
            The denoised sample at time t_0, shape (N, M) or (N, 1, M).
        """

class SDEIntegrator(Protocol):
    """Protocol for SDE integration methods."""
    def __call__(
        self,
        diff_process: DiffusionProcess,
        x_0: Tensor,
        t_0: Tensor | float,
        t_1: Tensor | float,
    ) -> Tensor:
        """Integrate the SDE from time t_1 to time t_0.

        Args:
            diff_process: The diffusion process defining the SDE.
            x_0: The initial sample at time t_0, shape (N, M) or (N, 1, M).
            t_0: The initial time step float or tensor, shape (N,).
            t_1: The final time step to integrate to.

        Returns:
            The integrated sample at time t_1, shape (N, M) or (N, 1, M).
        """
        ...

class ODEIntegrator(Protocol):
    """Protocol for ODE integration methods."""
    def __call__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_0: Tensor | float,
        t_1: Tensor | float,
    ) -> Tensor:
        """Integrate the ODE defined by the drift function from t_1 to t_0.

        Args:
            f: The function defining the ODE, takes (t,x) and returns dx/dt.
            x_0: The initial sample at time t_0, shape (N, M) or (N, 1, M).
            t_0: The initial time step float or tensor, shape (N,).
            t_1: The final time step to integrate to.
        """
        ...


class SDEReverseDiffusionProcess(ReverseDiffusionProcess):
    """Implementation of the reverse process using SDE integration."""
    def __init__(self, integrator: SDEIntegrator) -> None:
        self.integrator = integrator

    @singledispatchmethod
    def reverse(
        self,
        diff_process: ForwardDiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        t_0: Tensor | float = 0.,
        y: Tensor | None = None,
    ) -> Tensor:
        r"""Integrate the reverse process using the SDE integrator.

        The reverse SDE drift is:

        .. math::

            dx = \bigl[f(x,t) - G(t)\,G(t)^\top \nabla_x \log p_t(x)\bigr]\,dt
                 + G(t)\,d\bar{W}

        where :math:`\bar{W}` is a Brownian motion running backwards in time.
        A final Tweedie denoising step is applied when ``t_0`` is below
        ``_T0_SAFE``, where score estimates become unreliable near zero.

        Args:
            diff_process: The diffusion process defining the forward SDE.
            score_model: The score-based model used to estimate the score
                function.
            x_t: The noisy sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The time step of the noisy sample, shape (N,).
            t_0: The time step to integrate back to, default is 0.
            y: Optional labels for conditional generation, shape (N,).

        Returns:
            The denoised sample at time t_0, shape (N, M) or (N, 1, M).
        """
        def backward_drift(x: Tensor, t: Tensor) -> Tensor:
            score = score_model(x, t, y)
            drift = diff_process.drift(x, t)
            return drift - diff_process.diffusion_gram_times_v(score, t)

        reverse_process = CustomDiffusionProcess(
            drift=backward_drift,
            diffusion=diff_process.diffusion,
        )

        t_0_safe = max(t_0, self._T0_SAFE)
        x_t0 = self.integrator(reverse_process, x_t, t_1, t_0_safe)
        if t_0 < t_0_safe:
            return self._tweedie_denoise(
                diff_process, score_model, x_t0, x_t, t_0_safe, y,
            )
        return x_t0


    @reverse.register
    def _reverse_circulant(
            self,
            diff_process: CirculantSymmetricMatrixDiffusionProcess,
            score_model: ScoreModel,
            x_t: Tensor,
            t_1: Tensor | float,
            t_0: Tensor | float = 0.,
            y: Tensor | None = None,
    ) -> Tensor:
        r"""Integrate the reverse SDE in eigenspace using FFT projections.

        Projects x_t to eigenspace in :math:`O(NM\log M)`, runs the reverse
        SDE on the diagonal process in :math:`O(NM)` per step, then projects
        back in :math:`O(NM\log M)`.

        Args:
            diff_process: The circulant diffusion process.
            score_model: The score model (operates in original space).
            x_t: The noisy sample at time t_1, shape (N, M).
            t_1: The initial time step.
            t_0: The final time step, default 0.
            y: Optional conditioning labels, shape (N,).

        Returns:
            The denoised sample at time t_0, shape (N, M).
        """
        x_t_diag = _fft_to_eigenspace(x_t)

        def score_model_diag(
            x_diag: Tensor, t: Tensor, y: Tensor | None,
        ) -> Tensor:
            score = score_model(_eigenspace_to_fft(x_diag), t, y)
            return _fft_to_eigenspace(score)

        x_0_diag = self.reverse(
            diff_process.diagonal_process,
            score_model_diag, x_t_diag, t_1, t_0, y,
        )
        return _eigenspace_to_fft(x_0_diag)

class ProbabilityFlowODEReverseProcess(ReverseDiffusionProcess):
    """Implementation of the reverse process using ODE integration."""
    def __init__(self, integrator: ODEIntegrator) -> None:
        self.integrator = integrator

    @singledispatchmethod
    def reverse(
        self,
        diff_process: ForwardDiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        t_0: Tensor | float = 0.,
        y: Tensor | None = None,
    ) -> Tensor:
        r"""Integrate the reverse process using the ODE integrator.

        Uses the probability-flow ODE, a deterministic equivalent to the
        reverse SDE that shares the same marginal distributions:

        .. math::

            \frac{dx}{dt} = f(x,t)
            - \tfrac{1}{2}\,G(t)\,G(t)^\top \nabla_x \log p_t(x)

        The factor :math:`\tfrac{1}{2}` (vs. 1 in the SDE) removes the
        Brownian term while preserving the score-weighted drift.

        Args:
            diff_process: The diffusion process defining the forward SDE.
            score_model: The score-based model used to estimate the score
                function.
            x_t: The noisy sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The time step of the noisy sample, shape (N,).
            t_0: The time step to integrate back to, default is 0.
            y: Optional labels for conditional generation, shape (N,).

        Returns:
            The denoised sample at time t_0, shape (N, M) or (N, 1, M).
        """
        def backward_drift(t: Tensor, x: Tensor) -> Tensor:
            score = score_model(x, t, y)
            drift = diff_process.drift(x, t)
            return drift - 0.5 * diff_process.diffusion_gram_times_v(score, t)

        t_0_safe = max(t_0, self._T0_SAFE)
        x_t0 = self.integrator(backward_drift, x_t, t_1, t_0_safe)

        if t_0 < t_0_safe:
            return self._tweedie_denoise(
                diff_process, score_model, x_t0, x_t, t_0_safe, y,
            )
        return x_t0
    @reverse.register
    def _reverse_circulant(
            self,
            diff_process: CirculantSymmetricMatrixDiffusionProcess,
            score_model: ScoreModel,
            x_t: Tensor,
            t_1: Tensor | float,
            t_0: Tensor | float = 0.,
            y: Tensor | None = None,
    ) -> Tensor:
        r"""Integrate the reverse ODE in eigenspace using FFT projections.

        Projects x_t to eigenspace in :math:`O(NM\log M)`, runs the reverse
        ODE on the diagonal process in :math:`O(NM)` per step, then projects
        back in :math:`O(NM\log M)`.

        Args:
            diff_process: The circulant diffusion process.
            score_model: The score model (operates in original space).
            x_t: The noisy sample at time t_1, shape (N, M).
            t_1: The initial time step.
            t_0: The final time step, default 0.
            y: Optional conditioning labels, shape (N,).

        Returns:
            The denoised sample at time t_0, shape (N, M).
        """
        x_t_diag = _fft_to_eigenspace(x_t)

        def score_model_diag(
            x_diag: Tensor, t: Tensor, y: Tensor | None,
        ) -> Tensor:
            score = score_model(_eigenspace_to_fft(x_diag), t, y)
            return _fft_to_eigenspace(score)

        x_0_diag = self.reverse(
            diff_process.diagonal_process,
            score_model_diag, x_t_diag, t_1, t_0, y,
        )
        return _eigenspace_to_fft(x_0_diag)


class RK4Integrator(ODEIntegrator):
    """Classical 4th-order Runge-Kutta ODE integrator.

    Follows the ``ODEIntegrator`` protocol: ``__call__(f, x_t, t_1, t_0)``
    where ``f(t, x)`` returns *dx/dt*.
    """

    def __init__(self, n_steps: int = 1000) -> None:
        self.n_steps = n_steps

    def __call__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        x_0: Tensor,
        t_0: Tensor | float,
        t_1: Tensor | float,
    ) -> Tensor:
        """Integrate the ODE using the classical RK4 method.

        Args:
            f: The function defining the ODE, takes ``(t, x)`` and
                returns *dx/dt*.
            x_0: The initial sample at time *t_0*, shape ``(N, M)``.
            t_0: The initial time step.
            t_1: The final time step to integrate to.

        Returns:
            The integrated sample at time *t_1*, shape ``(N, M)``.
        """
        n = x_0.shape[0]
        device = x_0.device
        times = torch.linspace(
            float(t_0), float(t_1), self.n_steps + 1, device=device,
        )
        dt = times[1] - times[0]
        x = x_0.clone()
        for step in range(self.n_steps):
            t_step = times[step].item()
            t_mid = t_step + 0.5 * dt.item()
            t_next = t_step + dt.item()

            t_vec = torch.full((n,), t_step, device=device)
            t_mid_vec = torch.full((n,), t_mid, device=device)
            t_next_vec = torch.full((n,), t_next, device=device)

            k1 = f(t_vec, x)
            k2 = f(t_mid_vec, x + 0.5 * dt * k1)
            k3 = f(t_mid_vec, x + 0.5 * dt * k2)
            k4 = f(t_next_vec, x + dt * k3)

            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x


class EulerMaruyamaIntegrator(SDEIntegrator):
    """Simple Euler-Maruyama method for SDE integration."""
    def __init__(
        self,
        n_steps: int = 1000,
        device: torch.device | str = "cpu",
        seed: int | None = None,

    ) -> None:
        """Initialize the Euler-Maruyama integrator.

        Args:
            n_steps: The number of integration steps to take from t_1 to t_0.
            device: The device on which to perform the integration.
                    Default is "cpu".
            seed: The random seed for reproducibility. Default is None.
        """
        self.n_steps = n_steps
        self.seed = seed
        self.device = torch.device(device)

        # Resolve missing CUDA indices (e.g., 'cuda' -> 'cuda:0')
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())


    def _get_generator(self) -> torch.Generator:
        """Return a seeded torch.Generator, creating it on first call."""
        if not hasattr(self, "generator_"):
            self.generator_ = torch.Generator(device=self.device)
            if self.seed is not None:
                self.generator_.manual_seed(self.seed)
        return self.generator_

    def __call__(
        self,
        diff_process: DiffusionProcess,
        x_0: Tensor,
        t_0: Tensor | float,
        t_1: Tensor | float,
    ) -> Tensor:
        """Integrate the SDE using Euler-Maruyama method.

        Args:
            diff_process: The diffusion process defining the SDE.
            x_0: The initial sample at time t_0, shape (N, M).
            t_0: The initial time step as a float.
            t_1: The final time step to integrate to as a float.

        Returns:
            The integrated sample at time t_1, shape (N, M).
        """
        n, m = x_0.shape
        device = x_0.device
        if x_0.device != self.device:
            msg = (
                f"Input data is on {x_0.device} but integrator is on "
                f"{self.device}. Move the data or create the integrator "
                f"with device='{x_0.device}'."
            )
            raise ValueError(
                msg,
            )

        times = torch.linspace(t_0, t_1, self.n_steps + 1, device=device)
        dt = times[1] - times[0]  # negative when integrating backwards
        # std of dW is :math:`\sqrt{|dt|}`; variance is always positive
        sqrt_dt = torch.sqrt(torch.abs(dt))
        # :math:`dW \sim \mathcal{N}(0,\,|dt|\,I)`
        dw = torch.randn(
            (self.n_steps, n, m),
            dtype=x_0.dtype,
            device=device,
            generator=self._get_generator(),
        ) * sqrt_dt

        x_t = x_0.clone()

        for step in range(self.n_steps):
            t = torch.full((n,), times[step].item(), device=device)

            drift_t = diff_process.drift(x=x_t, t=t)
            diff_dw = diff_process.diffusion_times_v(v=dw[step], t=t)

            x_t = x_t + drift_t * dt + diff_dw

        return x_t

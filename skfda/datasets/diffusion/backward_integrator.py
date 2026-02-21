

from abc import ABC, abstractmethod

from typing import Callable, Protocol

import torch
from torch import Tensor

from .diffusion_process import DiffusionProcess 
from .score_model import ScoreModel


class BackwardIntegrator(ABC):
    """Integrator for the reverse process of a diffusion model.

        It defines the method `integrate` that takes a diffusion process,
        a score model, a noisy sample at time t_1, its label and it returns
        the denoised sample at time t_0.
    """
    @abstractmethod
    def bkwrd_integrate(
        self,
        diff_process: DiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        y: Tensor | None = None,
        t_0: Tensor | float = 0,
    ) -> Tensor:
        """Integrate the reverse process from time t_1 to time t_0.

        Args:
            diff_process: The diffusion process defining the forward SDE.
            score_model: The score-based model used to estimate the score
            function.
            x_t: The noisy sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The time step of the noisy sample, shape (N,).
            y: Optional labels for conditional generation, shape (N, C).
            t_0: The time step to integrate back to, default is 0.

        Returns:
            The denoised sample at time t_0, shape (N, M) or (N, 1, M).
        """


class EulerMaruyamaBackwardIntegrator(BackwardIntegrator):
    r"""Implementation of the Euler-Maruyama method.

    Given a diffusion process given by the SDE:
    `:math:`dx = f(x, t) dt + g(t) dW_t`

    It integrates the reverse-time SDE:
    `:math:`dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) d\bar{W}_t`
    where `:math:`s(x, t)` is the score function estimated by the score model.
    """
    def __init__(self, n_steps: int = 1000):
        self.n_steps = n_steps

    # TODO: Decide whether set a method to integrate
    # And another to reverse_process or just this method that does both things.
    # The method to integrate can be used for other purposes
    # not only for the reverse process.
    def bkwrd_integrate(
        self,
        diff_process: DiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        y: Tensor | None = None,
        t_0: Tensor | float = 0,
    ) -> Tensor:
        """Integrate the reverse process using the Euler-Maruyama method.

        Args:
            diff_process: The diffusion process defining the forward SDE.
            score_model: The score-based model used to estimate the score
            function.
            x_t: The noisy sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The time step of the noisy sample, shape (N,).
            y: Optional labels for conditional generation, shape (N,).
            t_0: The time step to integrate back to, default is 0.

        Returns:
            The denoised sample at time t_0, shape (N, M) or (N, 1, M).
        """
        if isinstance(t_1, (int, float)):
            t_1 = torch.tensor(t_1, device=x_t.device)  # Shape (1,)
        if isinstance(t_0, (int, float)):
            t_0 = torch.tensor(t_0, device=x_t.device)  # Shape (1,)

        def backward_drift(x: Tensor, t: Tensor) -> Tensor:
            score = score_model(x, t, y)
            diff_term = diff_process.diffusion(t)

            if diff_term.dim() == 1:
                # Scalar shape (N,)
                # Reshape to (N, 1) for broadcasting
                diff_term = diff_term.unsqueeze(1)
            if diff_term.dim() in (0, 1, 2):
                # Scalar, (N,1) or (N,M) shape
                # Element-wise multiplication
                diff_score = diff_term**2 * score
            else:
                # Full (N, M, M) shape
                # Matrix-vector multiplication
                diff_sqr = torch.einsum("bij,bkj->bik", diff_term, diff_term)
                diff_score = torch.einsum("bi,bij->bj", score, diff_sqr)

            return diff_process.drift(x, t) - diff_score

        return euler_maruyama_integration(
            x_t,
            t_1,
            t_0,
            backward_drift,
            diff_process.diffusion,
            n_steps=self.n_steps,
        )


def euler_maruyama_integration(
    x: Tensor,
    t_0: Tensor,
    t_end: Tensor,
    drift: Callable[[Tensor, Tensor], Tensor],
    diffusion: Callable[[Tensor], Tensor],
    n_steps: int,
) -> Tensor:
    """Performs Euler-Maruyama integration of an SDE.

    Args:
        x: The initial data at time t as a tensor, shape (N, M)
        t_0: The initial time steps as a tensor, shape (N,)
        t_end: The final time steps as a tensor, shape (N,)
        drift: The drift function of the SDE.
        diffusion: The diffusion coefficient function of the SDE.
        n_steps: The number of integration steps.

    Returns:
        The integrated data at time t_0 as a tensor, shape (N, M)
    """
    N, M = x.shape
    device = x.device
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]  # Can be negative
    # Note: sqrt(|dt|) because variance is always positive
    sqrt_dt = torch.sqrt(torch.abs(dt))

    # Brownian increment: dW ~ N(0, |dt|)
    dw = torch.randn((n_steps, N, M), dtype=x.dtype, device=device) * sqrt_dt
    x_t = x.clone()
    for n in range(n_steps):
        t = torch.full((N,), times[n].item(), device=device)

        drift_t = drift(x=x_t, t=t)  # Shape (N, M)
        diffusion_t = diffusion(t=t)

        # Handle different shapes of diffusion_t: (N,), (N, M) or (N, M, M)
        if diffusion_t.dim() == 1:
            # Scalar shape (N,)
            # Reshape to (N, 1) for broadcasting
            diffusion_t = diffusion_t.unsqueeze(1)
        if diffusion_t.dim() in (0, 1, 2):
            # Scalar, (N,1) or (N,M) shape
            # Element-wise multiplication
            diff_dW = diffusion_t * dw[n]
        else:
            # Full (N, M, M) shape
            # Matrix-vector multiplication
            diff_dW = torch.einsum("bi,bij->bj", dw[n], diffusion_t)
        x_t = x_t + drift_t * dt + diff_dW  # Shape (N, M)

    return x_t

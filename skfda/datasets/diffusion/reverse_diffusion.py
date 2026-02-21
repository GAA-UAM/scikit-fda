
from typing import Callable
from typing_extensions import Protocol

from torch import Tensor
import torch

from .diffusion_process import CustomDiffusionProcess, DiffusionProcess
from .score_model import ScoreModel


class ReverseDiffusionProcess(Protocol):
    """Protocol for the reverse process of a diffusion model.

    It defines the method `reverse` that takes a diffusion process, 
    a score model, a noisy sample at time t_1, its label and it returns
    the denoised sample at time t_0.
    """

    def reverse(
        self,
        diffusion_process: DiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        y: Tensor | None = None,
        t_0: Tensor | float = 0,
    ) -> Tensor:
        """Reverse the diffusion process from time t_1 to time t_0.

        Args:
            diffusion_process: The diffusion process defining the forward SDE.
            score_model: The score-based model used to estimate the score
            function.
            x_t: The noisy sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The time step of the noisy sample, shape (N,).
            y: Optional labels for conditional generation, shape (N, C).
            t_0: The time step to integrate back to, default is 0.

        Returns:
            The denoised sample at time t_0, shape (N, M) or (N, 1, M).
        """

class SDEIntegrator(Protocol):
    """Protocol for SDE integration methods."""
    def __call__(
        self,
        diff_process: DiffusionProcess,
        x_t: Tensor,
        t_1: Tensor | float,
        t_0: Tensor | float,
    ) -> Tensor:
        """Integrate the SDE from time t_1 to time t_0.

        Args:
            diff_process: The diffusion process defining the SDE.
            x_t: The initial sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The initial time step, shape (N,).
            t_0: The final time step to integrate to.

        Returns:
            The integrated sample at time t_0, shape (N, M) or (N, 1, M).
        """
        ...

class ODEIntegrator(Protocol):
    """Protocol for ODE integration methods."""
    def __call__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        x_t: Tensor,
        t_1: Tensor | float,
        t_0: Tensor | float,
    ) -> Tensor:
        """Integrate the ODE defined by the drift function from time t_1 to time t_0.

        Args:
            f: The function defining the ODE, takes (x,t) and returns dx/dt.
            x_t: The initial sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The initial time step, shape (N,).
            t_0: The final time step to integrate to.
        """
        ...


class SDEReverseDiffusionProcess(ReverseDiffusionProcess):
    """Implementation of the reverse process using SDE integration."""
    def __init__(self, integrator: SDEIntegrator):
        self.integrator = integrator

    def reverse(
        self,
        diff_process: DiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        y: Tensor | None = None,
        t_0: Tensor | float = 0,
    ) -> Tensor:
        """Integrate the reverse process using the SDE integrator.

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
        def backward_drift(x: Tensor, t: Tensor) -> Tensor:
            score = score_model(x, t, y)
            drift = diff_process.drift(x, t)
            diff = diff_process.diffusion(t)

            if diff.dim() == 1:
                diff = diff.unsqueeze(1)  # Shape (N, 1) for broadcasting

            if diff.dim() < 3:  # Diagonal case
                diff_score = diff**2 * score
            else:  # General case
                diff_sqr = torch.einsum("bij,bkj->bik", diff, diff)
                diff_score = torch.einsum("bi,bij->bj", score, diff_sqr)

            return drift - diff_score

        reverse_process = CustomDiffusionProcess(
            drift=backward_drift,
            diffusion=diff_process.diffusion,
        )
        return self.integrator(reverse_process, x_t, t_1, t_0)


class ODEReverseDiffusionProcess(ReverseDiffusionProcess):
    """Implementation of the reverse process using ODE integration."""
    def __init__(self, integrator: ODEIntegrator):
        self.integrator = integrator

    def reverse(
        self,
        diff_process: DiffusionProcess,
        score_model: ScoreModel,
        x_t: Tensor,
        t_1: Tensor | float,
        y: Tensor | None = None,
        t_0: Tensor | float = 0,
    ) -> Tensor:
        """Integrate the reverse process using the ODE integrator.

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
        def backward_drift(t: Tensor, x: Tensor) -> Tensor:
            score  = score_model(x, t, y)
            drift = diff_process.drift(x, t)
            diff = diff_process.diffusion(t)

            if diff.dim() == 1:
                diff = diff.unsqueeze(1)  # Shape (N, 1) for broadcasting

            if diff.dim() < 3:  # Diagonal case
                diff_score = diff**2 * score
            else:  # General case
                diff_sqr = torch.einsum("bij,bkj->bik", diff, diff)
                diff_score = torch.einsum("bi,bij->bj", score, diff_sqr)

            return drift - diff_score

        return self.integrator(backward_drift, x_t, t_1, t_0)



class EulerMaruyamaIntegrator(SDEIntegrator):
    """Simple implementation of the Euler-Maruyama method for SDE integration."""
    def __init__(self, n_steps: int = 1000):
        self.n_steps = n_steps

    def __call__(
        self,
        diff_process: DiffusionProcess,
        x_t: Tensor,
        t_1: Tensor | float,
        t_0: Tensor | float,
    ) -> Tensor:
        """Integrate the SDE using Euler-Maruyama method.

        Args:
            diff_process: The diffusion process defining the SDE.
            x_t: The initial sample at time t_1, shape (N, M) or (N, 1, M).
            t_1: The initial time step, shape (N,).
            t_0: The final time step to integrate to.

        Returns:
            The integrated sample at time t_0, shape (N, M) or (N, 1, M).
        """
        return euler_maruyama_integration(
            x=x_t,
            t_0=t_1,
            t_end=t_0,
            drift=diff_process.drift,
            diffusion=diff_process.diffusion,
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
        if diffusion_t.dim() < 3:  # Diagonal case
            # Scalar, (N,1) or (N,M) shape
            # Element-wise multiplication
            diff_dW = diffusion_t * dw[n]
        else:
            # Full (N, M, M) shape
            # Matrix-vector multiplication
            diff_dW = torch.einsum("bi,bij->bj", dw[n], diffusion_t)
        x_t = x_t + drift_t * dt + diff_dW  # Shape (N, M)

    return x_t

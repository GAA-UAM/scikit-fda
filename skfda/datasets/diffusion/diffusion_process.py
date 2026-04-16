from typing import Callable, Protocol, Final, Literal

from ...typing._base import RandomStateLike

from ..._utils._sklearn_adapter import BaseEstimator
from .torch_adapter import make_torch_generator

import torch
from torch import Tensor
import numpy as np

from abc import ABC, abstractmethod

class DiffusionProcess(Protocol):
    """Protocol for a diffusion process used in generative models.

    It defines the drift and diffusion terms of an SDE.
    """
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t."""
        ...

    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t."""
        ...


class CustomDiffusionProcess(DiffusionProcess):
    """Common interface of a diffusion process to be used in generative models.

    It defines the drift and diffusion terms of an SDE.
    """
    def __init__(self,
                 drift: Callable[[Tensor, Tensor], Tensor],
                 diffusion: Callable[[Tensor], Tensor]):
        """Initializes the diffusion process with the given drift and diffusion functions.

        Args:
            drift: A function that takes the current state x and time t, 
            and returns the drift term of the SDE, shape (N, M) or (N,).
            diffusion: A function that takes the current time t and 
            returns the diffusion term of the SDE, shape (N,), (N, M) or (N, M, M).
        """
        self.f = drift
        self.g = diffusion

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t.

        Args:
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The drift term as a tensor Shape (N, M) or (N,)
        """
        return self.f(x, t)

    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t.

        Output shape can be (N, M, M), (N, M) or (N,).

        If (N,) is returned, it represents a scalar
        multiple of the identity matrix. Hence, when used
        its behavior should be the same as if a scalar
        multiple of the identity matrix is returned.

        If (N, M) is returned, it represents only the diagonal
        of the diffusion matrix. Hence, when used its behavior
        should be the same as if a diagonal matrix is returned.

        Args:
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The diffusion term as a tensor Shape (N,), (N, M) or (N, M, M)
        """
        return self.g(t)


# TODO(): Decide whether an abstrac class or Protocol is better for this
class ForwardDiffusionProcess(DiffusionProcess, BaseEstimator):
    """Defines the forward diffusion process of a generative model.

    This method represents a diffusion process with gaussian
    conditional distributions. It defines the methods
    mean_cond and sigma_cond of the conditional distribution at time t.
    where mean is mean of the distribution and sigma is a square root
    of the covariance matrix.

    It also defines a method to compute the inverse of the square root
    of the covariance matrix at time t.

    Also defines the method to sample from the limiting distribution.
    """
    def fit(self, x: Tensor) -> "ForwardDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method can be used to fit the parameters of the
        diffusion process to the data.For the most basic, it learns the
        dimensionality of the data for sampling from limit distribution.
        More advanced diffusion process can learn other parameters such as
        the beta(t) schedule for variance-preserving.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        self.M = x.shape[1]  # Learn the dimension of the data
        return self

    def serialize_fit_data(self) -> dict:
        """Serializes the data learned in the `fit` method to a dictionary.

        This method should return a dictionary with the data necessary to
        store the parameters learned in the `fit` method. This is useful for
        saving the fitted diffusion process to disk or for transferring the
        learned parameters to another instance of the same class.

        Returns:
            A dictionary with the data learned in the `fit` method.
        """
        if not hasattr(self, "M"):
            raise ValueError("The diffusion process must be fitted to the data before serializing the fit data. Call the fit method with the training data.")
        return {"M": self.M}

    def deserialize_fit_data(self, data: dict) -> None:
        """Deserializes the data learned in the `fit` method from a dictionary.

        This method should load the parameters learned in the `fit` method
        from a dictionary, which is the format returned by the
        `serialize_fit_data` method.
        This method should be such that using `fit` and then
        `serialize_fit_data` returns a dictionary that when
        loaded into a new instance of the class with
        `deserialize_fit_data` leaves the new diffusion process in an equivalent
        state to what the original had after calling `fit`.

        Args:
            data: A dictionary with the data learned in the `fit` method.
        """
        if " M " not in data:
            raise ValueError("The key 'M' is missing from the data dictionary. This key is required to load the parameters of the diffusion process.")
        if not isinstance(data["M"], int):
            raise ValueError(f"The value of 'M' in the data dictionary should be an integer representing the dimension of the data. Got {type(data['M'])} instead.")
        self.M = data["M"]

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t."""
        ...

    @abstractmethod
    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t."""
        ...

    @abstractmethod
    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the mean of the conditional distribution at time t.

        It computes the mean of the distribution of X(t) given X(0) = x.

        Args:
            x: The current functional data as a tensor, shape (N, M)
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The mean of the conditional distribution at time t, shape (N, M)
        """
        ...

    @abstractmethod
    def sigma_cond(self, t: Tensor) -> Tensor:
        """Computes the square root of the covariance matrix of the conditional distribution at time t.

        It computes the square root of the covariance matrix of the
        distribution of X(t) given X(0) = x.

        Output shape can be (N, M, M), (N, M) or (N,).

        If (N,) is returned, it represents a scalar
        multiple of the identity matrix. Hence, when used
        its behavior should be the same as if a scalar
        multiple of the identity matrix is returned.

        If (N, M) is returned, it represents only the diagonal
        of the diffusion matrix. Hence, when used its behavior
        should be the same as if a diagonal matrix is returned.

        Args:
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The square root of the covariance matrix of the conditional
            distribution at time t, shape (N,), (N, M) or (N, M, M)
        """
        ...

    @abstractmethod
    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        """Computes the inverse of the square root of the covariance matrix of the conditional distribution at time t.

        It computes the inverse of the square root of the covariance matrix of the
        distribution of X(t) given X(0) = x.

        Output shape can be (N, M, M), (N, M) or (N,).

        If (N,) is returned, it represents a scalar
        multiple of the identity matrix. Hence, when used
        its behavior should be the same as if a scalar
        multiple of the identity matrix is returned.

        If (N, M) is returned, it represents only the diagonal
        of the diffusion matrix. Hence, when used its behavior
        should be the same as if a diagonal matrix is returned.

        Args:
            t: The current time steps as a tensor, shape (N,)

        Returns:
            The inverse of the square root of the covariance matrix of the conditional
            distribution at time t, shape (N,), (N, M) or (N, M, M)
        """
        ...

    @abstractmethod
    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Samples data from the limit distribution of the diffusion process.

        The output shape must be (n_samples, M) where M is the dimension of the data.
        It is learned at the `fit` method.

        Args:
            n_samples: The number of samples to generate.
            device: The device on which to create the samples. Default is "cpu".

        Returns:
            X: Tensor with samples from the limit distribution, shape (n_samples, M)
        """
        ...





class VariancePreservingDiffusionProcess(ForwardDiffusionProcess):
    r"""Implements a variance-preserving diffusion process.

    Based on the following forward SDE equation:
    :math:`d\mathbf{X}(t) = -\frac{1}{2}\beta(t)\mathbf{X}(t)dt + \sqrt{\beta(t)}d\mathbf{W}(t)`

    where :math:`\beta(t)` is a time-dependent function controlling
    the noise level.

    The linear and cosine schedules for :math:`\beta(t)` are implemented.
    In addition the :math:`\beta(0)` and :math:`\beta(T)` values can be set.
    In this class :math;`T` is taken to be 1.

    When using `cosine` schedule the values of :math:`\beta(0)` and
    :math:`\beta(T)` are used to clamp the values of :math:`\beta(t)` to be between
    them. This is done to avoid extremely large values of :math:`\beta(t)` at 
    the end of the diffusion process which can cause instability on the reverse process.
    """
    # TODO(): Decide whether to allow other T values
    T: Final[int] = 1

    def __init__(
        self,
        beta_schedule: Literal["linear", "cosine"] = "cosine",
        beta_min: float = 0.,
        beta_max: float = 20.,
    ):
        """Initializes the variance-preserving diffusion process.

        Args:
            beta_schedule: The schedule for beta(t). Can be 'linear' or
                        'cosine'. Default is 'cosine'.
            beta_min: The value of beta(0). Used only if `beta_schedule`
                    is 'linear'. Default is 0.
            beta_max: The value of beta(T). Used only if `beta_schedule`
                    is 'linear'. Default is 10.0.
            random_state: The random state to use for reproducible results. Default is None.
        """
        # TODO() Decide what to do with the device
        if beta_schedule not in ("linear", "cosine"):
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        self.beta_schedule = beta_schedule
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.M = None  # Will be set in fit method
        self.s = torch.tensor(1e-3)  # Small constant for numerical stability in cosine schedule

    def _beta_t(self, t: Tensor) -> Tensor:
        """Computes the value of beta(t) at time t.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            The value of beta(t) as a tensor, shape (N,)
        """
        if self.beta_schedule == "linear":
            beta_t = self.beta_min + (self.beta_max - self.beta_min) * t / self.T
        elif self.beta_schedule == "cosine":
            beta_t = (torch.pi / (self.T * (self.s + 1)) *
                      torch.tan(
                                torch.pi * 0.5 *
                                (t / self.T + self.s) / (self.s + 1),
                                )
                      )
            beta_t = torch.clamp(beta_t, min=self.beta_min, max=self.beta_max)  # Clamp beta_t to be between 0 and beta_max
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        return beta_t

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the drift term of the SDE at time t.

        For the variance-preserving diffusion process, the drift term is given by:
        :math:`\mathbf{f}(\mathbf{X}(t), t) = -\frac{1}{2} \beta(t) \mathbf{X}(t)`
        """
        return -0.5 * self._beta_t(t).unsqueeze(1) * x

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        For the variance-preserving diffusion process, the diffusion term is given by:
        :math:`g(t) = \sqrt{\beta(t)}`
        """
        return torch.sqrt(self._beta_t(t))

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the mean of the conditional distribution at time t.

        For the variance-preserving diffusion process, the mean of the
        conditional distribution is given by:
        :math:`\mu_t = \exp(-0.5 * \int_0^t \beta(s) ds) * x`

        Args:
            x: The data to condition on as a tensor, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            mu_t: The mean of the diffusion process at time t, shape (N,M)
        """
        if self.beta_schedule == "linear":

            integral_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2) / self.T

            mu_t = torch.exp(-0.5 * integral_beta)  # Shape (N,)

        elif self.beta_schedule == "cosine":
            f_t = torch.cos(((t / self.T + self.s) / (1 + self.s)) * (torch.pi / 2)) ** 2
            f_0 = torch.cos((self.s / (1 + self.s)) * (torch.pi / 2)) ** 2

            mu_t = torch.sqrt(f_t / f_0)  # Shape (N,)
        return mu_t.unsqueeze(-1) * x

    def sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the square root of the covariance matrix at time t.

        For the variance-preserving diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 =\int_0^t \beta(s) \exp(-\int_s^t\beta(u) du) ds =
        \left(1 - \exp(-\int_0^t \beta(s) ds)\right)`

        In the case of a linear schedule for beta(t), the integral can be computed
        analytically as:
        :math:`\int_0^t \beta(s) ds = \beta_min * t + \frac{(beta_max - beta_min) * t^2}{2T}`
        Hence sigma_t^2 can be computed as:
        :math:`\sigma_t^2 = \left(1 - \exp(-(\beta_min * t +  \frac{(beta_max - beta_min) * t^2}{2T}))\right)`

        In the case of a cosine schedule for beta(t), the integral is given by:
        :math:`\int_0^t \beta(s) ds = -\ln\left(\frac{f(t)}{f(0)}\right)`
        where :math:`f(t) = \cos^2\left(\frac{\pi}{2}\frac{t/s + 1}{1 + s}\right)`
        Hence sigma_t^2 can be computed as:
        :math:`\sigma_t^2 = \left(1 - \frac{f(t)}{f(0)}\right)`

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        if self.beta_schedule == "linear":
            integral_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2) / self.T

            return torch.sqrt(1 - torch.exp(-integral_beta))  # Shape (N,)

        if self.beta_schedule == "cosine":
            f_t = torch.cos(((t / self.T + self.s) / (1 + self.s)) * (torch.pi / 2)) ** 2
            f_0 = torch.cos((self.s / (1 + self.s)) * (torch.pi / 2)) ** 2

            sigma_t_squared = 1 - (f_t / f_0)  # Shape (N,)

            return torch.sqrt(sigma_t_squared)  # Shape (N,)

        raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the inverse of the square root of the covariance matrix at time t.

        For the variance-preserving diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 =\int_0^t \beta(s) \exp(\int_s^t\beta(u) du) ds =
        \left(1 - \exp(-\int_0^t \beta(s) ds)\right)`

        Hence, the inverse of the square root of the covariance matrix is
        given by the square root of the inverse of :math:`\sigma_t^2`.

        For stability reasons, we clamp the values of :math:`\sigma_t` to be
        at least 1e-3 to avoid division by zero or extremely large
        values of inv_sigma_t, which can cause instability in the
        training process.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        # Compute the integral of beta from 0 to t using the trapezoidal rule
        sigma_t = self.sigma_cond(t)  # Shape (N,)
        sigma_t_safe = torch.clamp(sigma_t, min=1e-3)  # Avoid division by zero
        return 1.0 / sigma_t_safe  # Shape (N,)

    def sample_limit_distribution(
            self,
            n_samples: int,
            device: torch.device | str = "cpu",
            random_state: RandomStateLike = None,
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-preserving diffusion process, the final distribution
        at time T is given by a standard normal distribution:
        :math:`\mathbf{X}(T) \sim \mathcal{N}(0, I)`

        Args:
            n_samples: The number of samples to generate.
            grid_size: The number of discretization points of the functional data.
                          Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".
            random_state: The random state to use for reproducible results.
                     Default is None.

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)
        """
        # TODO(): Decide wheteher to check if M is None or if it has an attribute.
        if not hasattr(self, "M"):
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        if self.M is None:
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        generator = make_torch_generator(random_state, device=device)  # Use a different generator for sampling from the limit distribution to avoid affecting the generator used for training
        return torch.randn((n_samples, self.M), device=device, generator=generator)

class VarianceExplodingDiffusionProcess(ForwardDiffusionProcess):
    r"""Implements a variance-exploding diffusion process.

    Based on the following forward SDE equation:
    :math:`d\mathbf{X}(t) = \sqrt{g(t)}d\mathbf{W}(t)`

    where :math:`g(t)` is a time-dependent function controlling
    the noise level.

    The linear and exponential schedules for :math:`g(t)` are implemented.
    In addition the :math:`g(0)` and :math:`g(T)` values can be set.
    In this class :math;`T` is taken to be 1.
    """
    # TODO(): Decide whether to allow other T values
    T: Final[int] = 1

    def __init__(
        self,
        g_schedule: Literal["linear", "exponential"] = "exponential",
        g_0: float = 0.1,
        g_T: float = 15.,
    ):
        """Initializes the variance-exploding diffusion process.

        Args:
            g_schedule: The schedule for g(t). Can be 'linear' or
                        'exponential'. Default is 'linear'.
            g_0: The value of g(0).
            g_T: The value of g(T).
            random_state: The random state to use for reproducible results. Default is None.

        """
        if g_schedule not in ("linear", "exponential"):
            raise ValueError(f"Unknown g schedule: {g_schedule}")
        
        self.g_schedule = g_schedule
        self.g_0 = g_0
        self.g_T = g_T


        self.M = None  # Will be set in fit method

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the drift term of the SDE at time t.

        For the variance-exploding diffusion process, the drift term is zero:
        :math:`\mathbf{f}(\mathbf{X}(t), t) = 0`
        """
        return torch.zeros_like(x)

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        For the variance-exploding diffusion process, the diffusion term is given by:
        :math:`g(t) = g(0) + (g(T) - g(0)) * t / T` for linear schedule
        :math:`g(t) = g(0) * (g(T) / g(0)) ** (t / T)` for exponential schedule
        """
        if self.g_schedule == "linear":
            return self.g_0 + (self.g_T - self.g_0) * t / self.T
        elif self.g_schedule == "exponential":
            return self.g_0 * torch.pow(self.g_T / self.g_0, t / self.T)
        else:
            raise ValueError(f"Unknown g schedule: {self.g_schedule}")

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the mean of the conditional distribution at time t.

        For the variance-exploding diffusion process, the mean of the
        conditional distribution is given by:
        :math:`\mu_t = x`

        Args:
            x: The data to condition on as a tensor, shape (N, M)
            t: The time steps as a tensor, shape (N,)

        Returns:
            mu_t: The mean of the diffusion process at time t, shape (N,M)
        """
        return x

    def sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the square root of the covariance matrix at time t.

        For the variance-exploding diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 = \int_0^t g^2(s) ds`

        Hence for the linear schedule, sigma_t^2 can be computed as:
        :math: `g^2(t) = (g(0) + (g(T) - g(0)) * t / T)^2 =
        g(0)^2 + 2 * g(0) * (g(T) - g(0)) * t / T + ((g(T) - g(0))^2 * t^2) / T^2`
        :math:`\sigma_t^2 = g(0)^2 * t^2 + g(0) * (g(T) - g(0)) * t^2 / T + ((g(T) - g(0))^2 * t^3) / (3 * T^2)`
        For the exponential schedule, sigma_t^2 can be computed as:
        :math:`g^2(t) = (g(0) * (g(T) / g(0)) ** (t / T))^2 = g(0)^2 * (g(T) / g(0)) ** (2t / T)`
        :math:`\sigma_t^2 = (g(0)^2 * T / (2 \ln(g(T) / g(0)))) * ((g(T) / g(0)) ** (2 * t / T) - 1)`

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            sigma_t: The square root of the covariance matrix at time t, shape (N,)
        """
        if self.g_schedule == "linear":
            return torch.sqrt(
                self.g_0**2 * t +
                self.g_0 * (self.g_T - self.g_0) * t ** 2 / self.T +
                (self.g_T - self.g_0)**2 * t ** 3 / (3 * self.T ** 2),
                )
        if self.g_schedule == "exponential":
            if self.g_0 != self.g_T:

                log_term = np.log(self.g_T / self.g_0)
                return torch.sqrt(
                    ((self.g_0 ** 2) * self.T / (2 * log_term)) *
                    (((self.g_T / self.g_0) ** (2 * t / self.T)) - 1),
                    )
            return torch.zeros_like(t)
        else:
            raise ValueError(f"Unknown g schedule: {self.g_schedule}")

    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the inverse of the square root of the covariance matrix at time t.

        For the variance-exploding diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 = \int_0^t g^2(s) ds`

        Hence, the inverse of the square root of the covariance matrix is
        given by the square root of the inverse of :math:`\sigma_t^2`.

        For stability reasons, we clamp the values of :math:`\sigma_t` to be
        at least 1e-3 to avoid division by zero or extremely large
        values of inv_sigma_t, which can cause instability in the
        training process.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        sigma_t = self.sigma_cond(t)  # Shape (N,)
        sigma_t_safe = torch.clamp(sigma_t, min=1e-3)  # Avoid division by zero
        return 1.0 / sigma_t_safe  # Shape (N,)

    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
        random_state: RandomStateLike = None,
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-exploding diffusion process, the final distribution
        at time T is given by a normal distribution with mean 0 and covariance
        matrix given by the integral of g(s) from 0 to T.

        Args:
            n_samples: The number of samples to generate.
            grid_size: The number of discretization points of the functional data.
                          Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".
            random_state: The random state to use for reproducible results.
                     Default is None.

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)
        """
        if not hasattr(self, "M"):
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        if self.M is None:
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        genetator = make_torch_generator(random_state, device=device)  # Use a different generator for sampling from the limit distribution to avoid affecting the generator used for training
        sigma_T = self.sigma_cond(torch.tensor([self.T], device=device)).item()  # Shape (1,)
        return torch.randn((n_samples, self.M), device=device, generator=genetator) * sigma_T

# Helper function for interpolation of the integrals of D(t) and F(t) in
# the DiagonalDiffusionProcess class.
def batch_linear_interp_1d(t: Tensor, t_grid: Tensor, f_grid: Tensor) -> Tensor:
    """Vectorized 1D linear interpolation with stability handling.

    Args:
        t: query points, shape (N,) - MUST be within [t_grid[0], t_grid[-1]]
        t_grid: time coordinates of data points, shape (K,) - must be sorted
        f_grid: y-coordinates for M functions, shape (K, M)

    Returns:
        Interpolated values, shape (N, M)
    """
    if t_grid.is_contiguous() is False:
        raise ValueError("t_grid must be a contiguous tensor.")
    if t.is_contiguous() is False:
        raise ValueError("t must be a contiguous tensor.")
    indices = torch.searchsorted(t_grid, t, right=False)
    indices = torch.clamp(indices, 1, len(t_grid) - 1)
    t0 = t_grid[indices - 1]
    t1 = t_grid[indices]
    y0 = f_grid[indices - 1, :]
    y1 = f_grid[indices, :]

    # Compute interpolation with stability
    dt = t1 - t0
    eps = 1e-10  # Adjust based on your data scale

    # Avoid division by zero: when dt is tiny, use midpoint value
    weight = torch.where(
        torch.abs(dt) > eps,
        (t - t0) / dt,
        torch.tensor(0.5, dtype=t.dtype, device=t.device),
    )
    # Linear interpolation: y = y0 + weight * (y1 - y0)
    result = y0 + weight.unsqueeze(-1) * (y1 - y0)
    return result  # (N, M)


class DiagonalDiffusionProcess(ForwardDiffusionProcess):
    """Implements a diffusion process with a diagonal drift and diffusion term.

    The drift term is given by a linear term with a diagonal matrix D(t) and 
    the diffusion term is given by a diagonal matrix g(t). All the computations
    are done numerically. In order to reduce computations, the integration is done
    once in a fixed grid of time points and then the values are interpolated 
    for the given time steps. This allows to have a reasonable computational cost
    when using this diffusion process as the forward process of a generative model.
    """
    # TODO(): Decide whether to allow other T values
    T: Final[int] = 1
    def __init__(
            self,
            D_t: Callable[[Tensor], Tensor],
            g_t: Callable[[Tensor], Tensor],
            n_integration_points: int = 1000,
    ):
        """Initializes the diagonal diffusion process.

        Args:
            D_t: A function that takes the current time t and returns the value
                of D(t) as a tensor, shape (N, M) or (N,)
            g_t: A function that takes the current time t and returns the
                value of g(t), de diffusion term as a tensor, shape (N, M) or (N,).
                If None, it is assumed that the variance preserving case is used,
                where g(t) = sqrt(-2 * D(t)).
            n_integration_points: The number of points to use for numerical
                integration when computing the mean and covariance of the 
                conditional distribution. Default is 1000.
        """
        # TODO(): Decide whether to validate D_t and g_t here or in the fit method.
        self.D_t = D_t
        self.g_t = g_t
        # TODO(): Decide whether to check if D_t and g_t return a tensor of shape (N, M).
        self.n_integration_points = n_integration_points
        self.M = None  # Will be set in fit method

    def fit(self, x: Tensor) -> "DiagonalDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method is used to learn the dimensionality of the data
        in order to sample from the limit distribution.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        super().fit(x)  # Learn the dimension of the data
        self.device = x.device  # Learn the device of the data
        self._precompute_D_integral()
        self._precompute_F_integral()
        return self

    def serialize_fit_data(self) -> dict:
        """Serializes the data learned in the `fit` method to a dictionary.

        This method should return a dictionary with the data necessary to
        store the parameters learned in the `fit` method. This is useful for
        saving the fitted diffusion process to disk or for transferring the
        learned parameters to another instance of the same class.

        Returns:
            A dictionary with the data learned in the `fit` method.
        """
        data = super().serialize_fit_data()
        data["device"] = self.device
        return data

    def deserialize_fit_data(self, data: dict) -> None:
        """Deserializes the data learned in the `fit` method from a dictionary.

        This method should load the parameters learned in the `fit` method
        from a dictionary, which is the format returned by the
        `serialize_fit_data` method.
        This method should be such that using `fit` and then
        `serialize_fit_data` returns a dictionary that when
        loaded into a new instance of the class with
        `deserialize_fit_data` leaves the new diffusion process in an equivalent
        state to what the original had after calling `fit`.

        Args:
            data: A dictionary with the data learned in the `fit` method.
        """
        if "M" not in data:
            raise ValueError("The key 'M' is missing from the data dictionary.")
        if not isinstance(data["M"], int):
            raise ValueError(f"The value of 'M' in the data dictionary must be an integer, but got {type(data['M'])}.")
        if "device" not in data:
            raise ValueError("The key 'device' is missing from the data dictionary.")
        if not isinstance(data["device"], torch.device) or data["device"] not in ("cpu", "cuda"):
            raise ValueError(f"The value of 'device' in the data dictionary must be a torch.device with type 'cpu' or 'cuda', but got {data['device']}.")
        self.M = data["M"]
        self.device = data["device"]
        self._precompute_D_integral()
        self._precompute_F_integral()

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the drift term of the SDE at time t.

        For this process, the drift term is given by D(t) * x, where D(t) is a diagonal matrix.
        """
        return self.D_t(t) * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Computes the diffusion term of the SDE at time t.

        For this process, the diffusion term is given by g(t), where g(t) is a diagonal matrix.
        """
        return self.g_t(t)

    def _precompute_D_integral(self):
        """Precomputes the integral of D(t) on a grid from 0 to T (1.0)."""
        t = torch.linspace(0, self.T, self.n_integration_points,
                           device=self.device)
        # Evaluate D(t): shape (n_integration_points, M) or (n_integration_points, 1)
        Dt = self.D_t(t)
        # Handle case where D_t returns (N, 1) and broadcast to (N, M)
        if Dt.dim() == 1:
            Dt = Dt.unsqueeze(-1)  # (N,) -> (N, 1)
        if Dt.shape[-1] == 1:
            Dt = Dt.expand(-1, self.M)  # (N, 1) -> (N, M)

        D_int = torch.cumulative_trapezoid(Dt, t, dim=0)
        # Prepend zeros at t=0
        D_int = torch.cat([D_int.new_zeros((1, Dt.shape[1])), D_int], dim=0)
        self.precomputed_Dgrid_T = t  # shape (n_integration_points,)
        self.precomputed_Dgrid = D_int  # shape (n_integration_points + 1, M)

    def _integrate_D(self, t: Tensor) -> Tensor:
        """
        It computes ∫[0 to t] D(u) du for diagonal D(t).

        Assumes t ∈ (0, 1], so all query times are within a known bounded
        interval.

        Parameters:
        t (Tensor): The current time, shape (N,) or (N, 1), values in (0, 1]

        Returns:
        Tensor:  The diagonal elements of exp(∫[0 to t] D(u) du),
                shape (N, M)
        """
        # Interpolate D integral for each time point
        # T_grid shape (integration_steps,)
        # D_grid shape (integration_steps, M)
        return batch_linear_interp_1d(
                    t.reshape(-1),
                    self.precomputed_Dgrid_T,
                    self.precomputed_Dgrid,
        )

    def _precompute_F_integral(self):
        """Precomputes the integral part of the sigma_t calculation on a grid from 0 to T (1.0).

        We compute F_i(t) = ∫[0 to t] g(s)^2 * exp(-2 * ∫[0 to s] D_i(u) du) ds

        Parameters:
        t_max (float): The maximum time to compute.
        device (torch.device): The device to perform computations on.
        """
        t = torch.linspace(0, self.T, self.n_integration_points,
                           device=self.device)

        # Evaluate g(s): shape (n_integration_points,) or (n_integration_points, 1)
        gt = self.g_t(t)
        if isinstance(gt, (int, float)):
            gt = torch.full_like(t, gt)
        if gt.dim() == 1:
            gt = gt.unsqueeze(-1)  # shape (n_integration_points, 1)
        # Handle case where g_t returns (N, 1) and broadcast to (N, M)
        if gt.shape[-1] == 1:
            gt = gt.expand(-1, self.M)  # (N, 1) -> (N, M)
        # h_i(s) for all i: shape (n_integration_points, M)
        h = (gt**2) * torch.exp(
                            -2 * self._integrate_D(t),
                            )
        # Cumulative integration: F_i(s) for all i,
        # shape (n_integration_points, M)
        F = torch.cumulative_trapezoid(h, t, dim=0)
        # Prepend zeros at s=0
        F = torch.cat([F.new_zeros((1, F.shape[1])), F], dim=0)
        self.precomputed_sigma_Tgrid = t  # shape (n_integration_points,)
        self.precomputed_sigma_Fgrid = F  # shape (n_integration_points + 1, M)

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the mean of the conditional distribution at time t.

        Is given by exp(-∫[0 to t] D(u) du) * x, where the exponential is taken
        elementwise since D(t) is diagonal.
        """
        return  torch.exp(self._integrate_D(t)) * x

    def sigma_cond(self, t: Tensor) -> Tensor:
        """Computes the square root of the covariance matrix at time t.

        Since the covariance matrix is diagonal,
        we return only the square root of the diagonal elements.

        Cov[X(t)] =
        Σ_0 @ exp(2 * ∫[0 to t] D(u) du) +
        ∫[0 to t] g(s)^2 Q(t)^T * Q(t) * exp(2 * ∫[s to t] D(u) du) ds

        To do it efficiently, we factor out the exp(2 * ∫[0 to t] D(u) du) term
        and compute the integral via numerical integration and interpolation.

        Let F_i(t) = ∫[0 to t] g(s)^2 * exp(-2 * ∫[0 to s] D_i(u) du) ds

        Cov[X_i(t)] = exp(2 * ∫[0 to t] D_i(u) du) * ( Σ_0[i] + F_i(t) )

        Then we integrate till t_max (1.0) and
        interpolate F_i(t) for each required t.

        Parameters:
        t (Tensor): The current time, shape (N,) or (N, 1)

        Returns:
        Tensor: The square root of the diagonal of Cov[X(t)], shape (N, M)
        """
        F_t = batch_linear_interp_1d(
                    t.reshape(-1),
                    self.precomputed_sigma_Tgrid,
                    self.precomputed_sigma_Fgrid,
        )  # shape (N, M)
        # TODO(): Decide what to do with the initial covariance Σ_0. 
        # For now we assume it is 0, but maybe we should add it as a parameter of the class 
        # and add it to the F_t term.
        exp_2Dt = torch.exp(2 * self._integrate_D(t))  # shape (N, M)
        cov = exp_2Dt * (F_t)  # shape (N, M)

        return torch.sqrt(cov.clamp(min=0))  # shape (N, M)

    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        r"""Computes the inverse of the square root of the covariance matrix at time t.

        For the variance-exploding diffusion process, the covariance matrix
        at time t is given by:
        :math:`\sigma_t^2 = \int_0^t g^2(s) ds`

        Hence, the inverse of the square root of the covariance matrix is
        given by the square root of the inverse of :math:`\sigma_t^2`.

        For stability reasons, we use a pseudoinverse approach: when
        :math:`\\sigma_t` is near zero (below 1e-3), we set the inverse to
        zero instead of a large value. This avoids numerical instability
        and is correct for eigenvalue directions where no noise is injected
        (e.g. positive eigenvalues in SpatialCoupling with VP clamping).

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            inv_sigma_t: The inverse of the square root of the covariance
                         matrix at time t, shape (N,)
        """
        sigma_t = self.sigma_cond(t)  # Shape (N,) or (N, M)
        # Pseudoinverse: set inverse to 0 where sigma is near zero.
        # When VP clamping makes g=0 for some eigenvalue directions
        # (e.g. positive eigenvalues in SpatialCoupling), sigma_i ≈ 0
        # in those directions. Using 1/clamp(sigma, 1e-3) would give
        # inv_sigma = 1000, breaking sigma * inv_sigma = I. Instead,
        # we set inv_sigma = 0 for those directions, effectively
        # ignoring them in the loss (correct for directions with no noise).
        mask = sigma_t > 1e-3
        inv_sigma = torch.zeros_like(sigma_t)
        inv_sigma[mask] = 1.0 / sigma_t[mask]
        return inv_sigma  # Shape (N,) or (N, M)

    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
        random_state: RandomStateLike = None,
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-preserving diffusion process, the final distribution
        at time T is given by a normal distribution with mean 0 and covariance
        matrix given by the integral of g(s) from 0 to T.

        Args:
            n_samples: The number of samples to generate.
            grid_size: The number of discretization points of the functional data.
                          Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".
            random_state: The random state to use for reproducible results.
                    Default is None.

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)
        """
        if not hasattr(self, "M"):
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        if self.M is None:
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        generator = make_torch_generator(random_state, device=device)
        sigma_T = self.sigma_cond(torch.tensor([self.T], device=device))  # Shape (1, M)
        return torch.randn((n_samples, self.M), device=device, generator=generator) * sigma_T  # Shape (n_samples, M)


# Helper function to compute the eigen decomposition of the matrix defined by K and rho.
def get_eigen_decomposition(K: Callable[[Tensor], Tensor| float],
                            ro: Callable[[Tensor], Tensor| float],
                            M: int,
                            device: torch.device | str = 'cpu',
                            ) -> tuple[Tensor, Tensor]:
    """Compute the eigen decomposition of the matrix defined by K and rho functions on a grid of given size.

    K(t)   ρ  ρ² ρ³ ... ρ^(M/2) ρ^(M/2-1) ... ρ² ρ
    ρ  K(t)   ρ  ρ² ... ρ^(M/2) ρ^(M/2-1) ... ρ³ ρ²
    .
    .
    .
    ρ² ρ³ ... ρ^(M/2) ρ^(M/2-1) ... ρ² ρ  K(t) ρ
    ρ  ρ² ... ρ^(M/2) ρ^(M/2-1) ... ρ³ ρ² ρ  K(t)

    ρ is also time dependent, but the eigenvectors are not.
    
    Parameters:
    -----------
    K: Callable[[Tensor], Tensor| float]
        Diagonal elements of B(t) function.
    ro: Callable[[Tensor], Tensor| float]
        No diagonal elements of B(t) function.
        Following the power law structure.
    M: int
        Size of the matrix.
    device: torch.device | str
        Device to run the computations on. Default is 'cpu'.
    Returns:
        Q: Eigenvectors matrix of shape (M, M).
        lambdas: Eigenvalues Callable[[Tensor], Tensor] of shape (M,).
    """
    with torch.no_grad():
            # Ensure M is an integer tensor for calculations
            m_tensor = torch.tensor(M, device=device, dtype=torch.float32)
            indices = torch.arange(m_tensor, device=device, dtype=torch.float32)

            Q = torch.zeros((M, M), device=device)

            # --- 1. DC Component (k=0) ---
            # Corresponds to lambda[0]
            Q[:, 0] = 1.0 / torch.sqrt(m_tensor)

            # --- 2. Harmonic Frequencies (Pairs) ---
            # We iterate k from 1 up to (M-1)//2.
            # This generates matching pairs at columns k and M-k.
            # Example M=8: k=1, 2, 3. (Cols 1&7, 2&6, 3&5 filled).
            # Example M=9: k=1, 2, 3, 4. (Cols 1&8, 2&7, 3&6, 4&5 filled).

            limit = (M - 1) // 2
            if limit > 0:
                k_vals = torch.arange(1, limit + 1, device=device, dtype=torch.float32)
                # Outer product: indices (rows) x k_vals (freqs)
                # Shape: (M, num_freqs)
                angles = (2 * torch.pi * indices.unsqueeze(1) * k_vals.unsqueeze(0)) / m_tensor
                # Flip angles son M-1 is filled with k=1, M-2 with k=2, etc.
                flip_angles = torch.flip(angles, dims=[1])

                normalization = torch.sqrt(2.0 / m_tensor)

                # Fill the 'left' side (k) with Cosines
                Q[:, 1:limit+1] = normalization * torch.cos(angles)

                # Fill the 'right' side (M-k) with Sines
                # We map k=1 to col M-1, k=2 to col M-2, etc.
                Q[:, M - limit:] = normalization * torch.sin(flip_angles)

            # --- 3. Nyquist Component (k=M/2) ---
            # Only exists if M is even. Corresponds to lambda[M/2].
            if M % 2 == 0:
                # This is the alternating vector [1, -1, 1, -1...]
                Q[:, M // 2] = ((-1) ** indices) / torch.sqrt(m_tensor)

    def lambdas(t: Tensor) -> Tensor:
        # 1. Evaluate the parameter functions at time t
        k_val = K(t) # Shape (N, M) or (N,)

        r_val = ro(t) # Shape (N,)
        # 2. Construct the first row based on Power Law: rho^dist
        # dist is the circular distance: min(|i-j|, M-|i-j|)
        dists = torch.minimum(indices, m_tensor - indices)

        # Avoid potential error if r_val is 0 (0^0=1, but 0^n=0)
        # We assume 0 <= rho < 1.
        r_val_expanded = r_val.unsqueeze(-1) if r_val.ndim > 0 else r_val
        row = torch.pow(r_val_expanded, dists)

        row[..., 0] = 0.0  # Overwrite diagonal with K(t)

        # 3. Compute eigenvalues via FFT
        # Real symmetric circulant -> eigenvalues are Real(FFT(row))
        lambdas = torch.fft.fft(row).real

        return lambdas + k_val.unsqueeze(-1)  # Shape (N, M)
    return Q, lambdas


#Helper function to build the B(t) matrix from K(t) and rho(t).
def build_B_t_from_K_rho(
        K: Callable[[Tensor], Tensor| float],
        rho: Callable[[Tensor], Tensor| float],
        M: int,
        device: torch.device | str = 'cpu',
) -> Callable[[Tensor], Tensor]:
    """Builds the B(t) matrix from K(t) and rho(t) functions.

    K(t)   ρ  ρ² ρ³ ... ρ^(M/2) ρ^(M/2-1) ... ρ² ρ
    ρ  K(t)   ρ  ρ² ... ρ^(M/2) ρ^(M/2-1) ... ρ³ ρ²
    .
    .
    .
    ρ² ρ³ ... ρ^(M/2) ρ^(M/2-1) ... ρ² ρ  K(t) ρ
    ρ  ρ² ... ρ^(M/2) ρ^(M/2-1) ... ρ³ ρ² ρ  K(t)

    Parameters:
    K: Callable[[Tensor], Tensor| float]
        Diagonal elements of B(t) function.
    rho: Callable[[Tensor], Tensor| float]
        No diagonal elements of B(t) function.
        Following the power law structure.
    M: int
        Size of the matrix.
    device: torch.device | str
        Device to run the computations on. Default is 'cpu'.
    
    Returns:
        B_t: Callable[[Tensor], Tensor] that takes time t and returns the B(t) matrix of shape (N, M, M).
    """
    def B_t(t: Tensor) -> Tensor:
        k_val = K(t).unsqueeze(-1).unsqueeze(-1) # Shape (N,1,1)
        r_val = rho(t).unsqueeze(-1).unsqueeze(-1) # Shape (N,1,1)

        indices = torch.arange(M, device=device)
        dists = torch.minimum(indices, M - indices)
        row = torch.pow(r_val, dists).unsqueeze(0).expand(t.shape[0], -1, -1)
        # Set diagonal to zero
        row[..., indices, indices] = k_val.squeeze(-1)

        # Build circulant matrix for each batch element
        # TODO(): This is inefficient since we are building matrices every call.
        # However, I think is the only safe way to do it for parallel processing.
        B_matrices = torch.zeros((t.shape[0], M, M), device=device)
        for i in range(M):
            B_matrices += torch.roll(row, shifts=i, dims=2)

        return B_matrices
    return B_t

def _duplicate_symmetric_eigenvalues(
        lambda_half: Callable[[Tensor], Tensor],
        M: int,
        device: torch.device | str = 'cpu',
    ) -> Callable[[Tensor], Tensor]:
    """Duplicates the eigenvalues of a symmetric matrix given the first M/2+1 eigenvalues.

    Parameters:
    ------------
        lambda_half: Callable[[Tensor], Tensor| float]
            The first M/2+1 eigenvalues.
        M: int
            Size of the matrix.
        device: torch.device | str
            Device to run the computations on. Default is 'cpu'.

    Returns:
            lambdas: Eigenvalues Callable[[Tensor], Tensor] of shape (N,M).
    """
    def lambdas(t: Tensor) -> Tensor:
        N = t.shape[0]
        lambda_half_val = lambda_half(t)

        lambdas_val = torch.zeros((N, M), device=device)
        lambdas_val[:, :M//2 + 1] = lambda_half_val if lambda_half_val.ndim == 2 else lambda_half_val.unsqueeze(-1)

        # Fixed symmetry slicing for Odd/Even M
        lambdas_val[:, M//2 + 1:] = torch.flip(lambdas_val[:, 1:(M + 1)//2], dims=[1])

        return lambdas_val
    return lambdas

def _duplicate_symmetric_row_and_get_eigenvalues(c_half: Callable[[Tensor], Tensor], M: int, device: torch.device | str = 'cpu') -> Callable[[Tensor], Tensor]:
    """Compute the eigen decomposition of a circulant matrix defined by a half row function.

        c_0(t)  c_1(t)  c_2(t) ... c_(M/2)(t) c_(M/2-1)(t) ... c_2(t) c_1(t)
        c_1(t)  c_0(t)  c_1(t)   c_2(t)  ... c_(M/2-1)(t)  c_(M/2)(t) ... c_3(t)  c_2(t)
        .
        .
        .
        c_2(t) c_3(t) ... c_(M/2)(t) c_(M/2-1)(t) ... c_2(t) c_1(t)
        c_1(t)  c_2(t)  c_3(t)   c_4(t)  ... c_(M/2)(t)  c_(M/2-1)(t) ... c_1(t)  c_0(t)

    Parameters:
    -----------
    c: Callable[[Tensor], Tensor]
        The M/2+1 first row elements of the circulant matrix.
    M: int
        Size of the matrix.
    device: torch.device | str
        Device to run the computations on. Default is 'cpu'.

    Returns:
        lambdas: Eigenvalues Callable[[Tensor], Tensor] of shape (N,M).
    """
    def lambdas(t: Tensor) -> Tensor:
        N = t.shape[0]
        c_val = c_half(t)

        row = torch.zeros((N, M), device=device)
        row[:, :M//2 + 1] = c_val if c_val.ndim == 2 else c_val.unsqueeze(-1)
        row[:, M//2 + 1:] = torch.flip(row[:, 1:(M + 1)//2], dims=[1])

        return torch.fft.fft(row).real
    return lambdas

def get_cosine_basis(M: int, device: torch.device | str = 'cpu') -> Tensor:
    """Computes the cosine basis for a given size M.

    The cosine basis is an orthogonal matrix that diagonalizes
    circulant symmetric matrices. It has the following structure:
        c_0  c_1  c_2 ... c_(M/2) c_(M/2-1) ... c_2 c_1
        c_1  c_0  c_1   c_2  ... c_(M/2-1)  c_(M/2) ... c_3  c_2
        .
        .
        .
        c_2 c_3 ... c_(M/2) c_(M/2-1) ... c_2 c_1
        c_1  c_2  c_3   c_4  ... c_(M/2)  c_(M/2-1) ... c_1  c_0

    where the first column is the DC component, the next columns are the harmonic frequencies,
    and if M is even, the middle column is the Nyquist frequency.

    Args:
        M: The size of the matrix.
        device: The device on which to create the matrix. Default is 'cpu'.
    Returns:
        Q: The real part of the Fourier basis, shape (M, M).
    """
    with torch.no_grad():
        # Ensure M is an integer tensor for calculations
        m_tensor = torch.tensor(M, device=device, dtype=torch.float32)
        indices = torch.arange(m_tensor, device=device, dtype=torch.float32)

        Q = torch.zeros((M, M), device=device)

        # --- 1. DC Component (k=0) ---
        # Corresponds to lambda[0]
        Q[:, 0] = 1.0 / torch.sqrt(m_tensor)

        # --- 2. Harmonic Frequencies (Pairs) ---
        # We iterate k from 1 up to (M-1)//2.
        # This generates matching pairs at columns k and M-k.
        # Example M=8: k=1, 2, 3. (Cols 1&7, 2&6, 3&5 filled).
        # Example M=9: k=1, 2, 3, 4. (Cols 1&8, 2&7, 3&6, 4&5 filled).

        limit = (M - 1) // 2
        if limit > 0:
            k_vals = torch.arange(1, limit + 1, device=device, dtype=torch.float32)
            # Outer product: indices (rows) x k_vals (freqs)
            # Shape: (M, num_freqs)
            angles = (2 * torch.pi * indices.unsqueeze(1) * k_vals.unsqueeze(0)) / m_tensor
            # Flip angles son M-1 is filled with k=1, M-2 with k=2, etc.
            flip_angles = torch.flip(angles, dims=[1])

            normalization = torch.sqrt(2.0 / m_tensor)

            # Fill the 'left' side (k) with Cosines
            Q[:, 1:limit+1] = normalization * torch.cos(angles)

            # Fill the 'right' side (M-k) with Sines
            # We map k=1 to col M-1, k=2 to col M-2, etc.
            Q[:, M - limit:] = normalization * torch.sin(flip_angles)

        # --- 3. Nyquist Component (k=M/2) ---
        # Only exists if M is even. Corresponds to lambda[M/2].
        if M % 2 == 0:
            # This is the alternating vector [1, -1, 1, -1...]
            Q[:, M // 2] = torch.cos(torch.pi * indices) / torch.sqrt(m_tensor)
    return Q

class CirculantSymmetricMatrixDiffusionProcess(ForwardDiffusionProcess):
    """Implements a diffusion process with a circulant symmetric drift and diffusion term.

    A circulant symmetric matrix has the following structure:
        c_0(t)  c_1(t)  c_2(t) ... c_(M/2)(t) c_(M/2-1)(t) ... c_2(t) c_1(t)
        c_1(t)  c_0(t)  c_1(t)   c_2(t)  ... c_(M/2-1)(t)  c_(M/2)(t) ... c_3(t)  c_2(t)
        .
        .
        .
        c_2(t) c_3(t) ... c_(M/2)(t) c_(M/2-1)(t) ... c_2(t) c_1(t)
        c_1(t)  c_2(t)  c_3(t)   c_4(t)  ... c_(M/2)(t)  c_(M/2-1)(t) ... c_1(t)  c_0(t)


    This matrix can be diagonalized by a constant real orthogonal matrix, given by the
    real part of the Fourier basis. Hences, the computations are done in the eigenvector space
    where the drift and diffusion are diagonal.
    """
    T: Final[int] = 1

    def __init__(
            self,
            drift_term: Callable[[Tensor], Tensor],
            diffusion_term: Callable[[Tensor], Tensor],
            fourier_drift: bool = False,
            fourier_diffusion: bool = False,
            n_integration_points: int = 1000,
    ):
        """Initializes the circulant symmetric matrix diffusion process.

        Args:
            drift_term: A function that takes the current time t and returns
                the value of M/2 + 1 coefficients needed to build the circulant
                symmetric matrix. Depending on the value of fourier drift, the
                coefficients are either the eigenvalues of the drift term in
                the Fourier basis (if fourier_drift is True) or the first
                M/2 + 1 elements of the first row of the circulant matrix
                (if fourier_drift is False).
            diffusion_term: A function that takes the current time t and returns
                the value of M/2 + 1 coefficients needed to build the circulant
                symmetric matrix for the diffusion term. Depending on the value of
                fourier_diffusion, the coefficients are either the eigenvalues of
                the diffusion term in the Fourier basis (if fourier_diffusion is True)
                or the first M/2 + 1 elements of the first row of the circulant
                matrix for the diffusion term (if fourier_diffusion is False).
            fourier_drift: If True, the drift_term function returns the eigenvalues of the
                drift term in the Fourier basis. If False, the drift_term function returns
                the first M/2 + 1 elements of the first row of the circulant matrix
                for the drift term. Default is False.
            fourier_diffusion: If True, the diffusion_term function returns the eigenvalues of the
                diffusion term in the Fourier basis. If False, the diffusion_term function returns
                the first M/2 + 1 elements of the first row of the circulant matrix
                for the diffusion term. Default is False.
            n_integration_points: The number of points to use for numerical
                integration when computing the mean and covariance of the
                conditional distribution. Default is 1000.
        """
        self.random_state = random_state
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.fourier_drift = fourier_drift
        self.fourier_diffusion = fourier_diffusion
        self.n_integration_points = n_integration_points
        self.M = None  # Will be set in fit method


    def fit(self, x: Tensor) -> "CirculantSymmetricMatrixDiffusionProcess":
        """Fits the parameters of the diffusion process to the data.

        This method is used to learn the dimensionality of the data
        in order to sample from the limit distribution and
        to compute the matrix B(t) of the drift term.

        Args:
            x: The input functional data as a tensor, shape (N, M)

        Returns:
            self: The fitted diffusion process.
        """
        super().fit(x)  # Learn the dimension of the data
        self.device = x.device  # Learn the device of the data
        self.Q = get_cosine_basis(self.M, self.device)  # Shape (M, M)
        if self.fourier_drift:
            self.lambdas = _duplicate_symmetric_eigenvalues(
                lambda_half=self.drift_term,
                M=self.M,
                device=self.device
            )
        else:
             self.lambdas = _duplicate_symmetric_row_and_get_eigenvalues(
                c_half=self.drift_term,
                M=self.M,
                device=self.device,
            )

        if self.fourier_diffusion:
            self.diag_gt = _duplicate_symmetric_eigenvalues(
                lambda_half=self.diffusion_term,
                M=self.M,
                device=self.device,
            )
        else:
            self.diag_gt = _duplicate_symmetric_row_and_get_eigenvalues(
                c_half=self.diffusion_term,
                M=self.M,
                device=self.device,
            )

        self.QT = self.Q.T

        self.diagonal_process = DiagonalDiffusionProcess(
            D_t=self.lambdas,
            g_t=self.diag_gt,
            n_integration_points=self.n_integration_points,
            random_state=self.random_state,
        ).fit(x @ self.Q)  # Fit in the eigenbasis

        self.B_t = lambda t: self.Q @ torch.diag_embed(self.lambdas(t)) @ self.QT

        # Shape (M, M) @ (N, M, M) -> (N, M, M)
        self.G = lambda t: self.Q @ torch.diag_embed(self.diag_gt(t))
        return self

    def serialize_fit_data(self) -> dict:
        """Serializes the data learned in the `fit` method to a dictionary.

        This method should return a dictionary with the data necessary to
        store the parameters learned in the `fit` method. This is useful for
        saving the fitted diffusion process to disk or for transferring the
        learned parameters to another instance of the same class.

        Returns:
            A dictionary with the data learned in the `fit` method.
        """
        data = super().serialize_fit_data()
        data["device"] = self.device
        return data

    def deserialize_fit_data(self, data: dict) -> None:
        """Deserializes the data learned in the `fit` method from a dictionary.

        This method should load the parameters learned in the `fit` method
        from a dictionary, which is the format returned by the
        `serialize_fit_data` method.
        This method should be such that using `fit` and then
        `serialize_fit_data` returns a dictionary that when
        loaded into a new instance of the class with
        `deserialize_fit_data` leaves the new diffusion process in an equivalent
        state to what the original had after calling `fit`.

        Args:
            data: A dictionary with the data learned in the `fit` method.
        """
        if "M" not in data:
            raise ValueError("The key 'M' is missing from the data dictionary.")
        if not isinstance(data["M"], int):
            raise ValueError(f"The value of 'M' in the data dictionary must be an integer, but got {type(data['M'])}.")
        if "device" not in data:
            raise ValueError("The key 'device' is missing from the data dictionary.")
        if not isinstance(data["device"], torch.device) or data["device"] not in ("cpu", "cuda"):
            raise ValueError(f"The value of 'device' in the data dictionary must be a torch.device with type 'cpu' or 'cuda', but got {data['device']}.")
        self.M = data["M"]
        self.device = data["device"]
        x_toy = torch.zeros((1, self.M), device=self.device)
        self.fit(x_toy)
        # This will compute the eigen decomposition and precompute the
        # matrices needed for the drift and diffusion terms, which are the
        # main parameters learned in the fit method.


    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Computes the drift term of the SDE at time t.
        
        For the spatial coupling diffusion process, the drift term is given by:
        :math:`\mathbf{f}(\mathbf{X}(t), t) = B(t) \mathbf{X}(t)`
        where B(t) is the matrix defined by K(t) and rho(t).
        """
        B_t = self.B_t(t)  # Shape (N, M, M)
        return torch.einsum('nij,nj->ni', B_t, x)
        # TODO(): Check if I can optimize this by chaning basis:

    def diffusion(self, t: Tensor) -> Tensor:
        r"""Computes the diffusion term of the SDE at time t.

        For the spatial coupling diffusion process, the diffusion term is given by:
        :math:`\mathbf{g}(t) = g(t)` if diagonal_basis_g_t is False, where g(t) is a function of time.
        If diagonal_basis_g_t is True, the diffusion term is given by:
        :math:`\mathbf{g}(t) = Q g(t)`, where g(t) is a function of time and I is the identity matrix.

        Args:
            t: The time steps as a tensor, shape (N,)

        Returns:
            g_t: The diffusion term at time t, shape (N, M, M), (N, M) or (N,)
        """
        return self.G(t)

    def mean_cond(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes the mean of the conditional distribution at time t.

        Using the eigen decomposition of the drift term,
        we can compute the mean of the conditional distribution in the
        eigenvector space where the drift is diagonal and then transform
        it back to the original space.

        With Y_0 = Q^T @ X_0
        E[X(t) | X_0] = E[Q @ Y(t) | X_0] = Q @ E[Y(t) | Y_0]

        Args:
            x : The initial condition X_0, shape (N, M)
            t : The current time, shape (N,)

        Returns:
            mean_t (Tensor): The mean of the conditional distribution at time t, shape (N, M).
        """
        # We can compute the mean of the conditional distribution in the eigenvector space where the drift is diagonal and then transform it back to the original space.
        y = x @ self.Q  # Shape (N, M) @ (M, M) -> (N, M)
        mean_eigen = self.diagonal_process.mean_cond(y, t)  # Shape (N, M)
        return mean_eigen @ self.QT  # Shape (N, M) @ (M, M) -> (N, M)


    def sigma_cond(self, t: Tensor) -> Tensor:
        """Computes a square root of the covariance matrix of X(t).

        As Cov[X(t)] = Q @ Cov[Y(t)] @ Q^T,
        And Cov[Y(t)] is diagonal and Q is orthogonal,
        We want to return the square root of Cov[X(t)].
        Hence if Sigma_Y(t) is the square root of the diagonal of Cov[Y(t)],
        Sigma_X(t) = Q @ diag(Sigma_Y(t)) @ Q^T

        Args:
            t : The current time, shape (N,) or (N, 1)

        Returns:
            Tensor: The square root of the diagonal of Cov[X(t)], shape (N, M, M)
        """
        # Get diagonal std from diagonal diffusion process
        sigma_Y_t = self.diagonal_process.sigma_cond(t)  # shape (N, M)

        # Build diag(Sigma_Y(t)) matrices
        sigma_Y_t_diag = torch.diag_embed(sigma_Y_t)  # shape (N, M, M)
        # Compute Sigma_X(t) = Q @ diag(Sigma_Y(t)) @ Q^T (in row convention)
        return self.Q @ sigma_Y_t_diag @ self.QT  # shape (N, M, M)

    def inv_sigma_cond(self, t: Tensor) -> Tensor:
        """It computes the inverse of the square root of the covariance matrix of X(t).

        Since Cov[X(t)] = Q @ Cov[Y(t)] @ Q^T,
        And Cov[Y(t)] is diagonal and Q is orthogonal,
        Cov[X(t)]^{-1/2} = Q @ Cov[Y(t)]^{-1/2} @ Q^T
        Where Cov[Y(t)]^{-1/2} is diagonal with elements 1 / sigma_Y_i(t)
        """
        # Get diagonal std from diagonal diffusion process
        inv_sigma_Y_t = self.diagonal_process.inv_sigma_cond(t)  # shape (N, M)
        # Build diag(Cov[Y(t)]^{-1/2}) matrices
        inv_sigma_Y_t_diag = torch.diag_embed(inv_sigma_Y_t)  # shape (N, M, M)
        # Compute Cov[X(t)]^{-1/2} = Q @ diag(Cov[Y(t)]^{-1/2}) @ Q^T
        return self.Q @ inv_sigma_Y_t_diag @ self.QT  # shape (N, M, M)

    def sample_limit_distribution(
        self,
        n_samples: int,
        device: torch.device | str = "cpu",
        random_state: RandomStateLike = None,
    ) -> Tensor:
        r"""Samples data from the final distribution of the diffusion process.

        For the variance-preserving diffusion process, the final distribution
        at time T is given by a normal distribution with mean 0 and covariance
        matrix given by the integral of g(s) from 0 to T.

        Args:
            n_samples: The number of samples to generate.
            grid_size: The number of discretization points of the functional data.
                          Refers to the M dimension of the data.
            device: The device on which to create the samples.
                    Default is "cpu".
            random_state: The random state to use for reproducible results.
                    Default is None.

        Returns:
            X: Tensor with samples from the final distribution.
            Shape (n_samples, grid_size)
        """
        if not hasattr(self, "M"):
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        if self.M is None:
            raise ValueError("The diffusion process must be fitted to the data before sampling from the final distribution. Call the fit method with the training data.")
        sigma_T = self.sigma_cond(torch.tensor([self.T], device=device))  # Shape (1, M, M)
        generator = make_torch_generator(random_state, device=device)
        noise = torch.randn((n_samples, self.M), device=device, generator=generator)  # Shape (n_samples, M)
        # TODO(): Check if I can optimize this by chaning basis:
        # Y = self.diagonal_process.sample_limit_distribution(n_samples, device)  # Shape (n_samples, M)
        # X = Y @ self.QT  # Shape (n_samples, M) @ (M, M) -> (n_samples, M)
        return noise @ sigma_T.squeeze(0)  # Shape (n_samples, M)

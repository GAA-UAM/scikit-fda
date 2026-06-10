"""Score network models for score-based diffusion."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .diffusion_process import ForwardDiffusionProcess

ScoreStateType = int | float | Tensor | torch.device
ScoreCheckpointDict = TypedDict(
    "ScoreCheckpointDict",
    {
        "class": type["ScoreModel"],
        "init_kwargs": dict[str, Any],
        "fit_state": dict[str, ScoreStateType],
    },
)

class ScoreModel(nn.Module, ABC):
    """Abstract base class for time-dependent score-based models."""
    _init_kwargs: dict[str, Any]

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        r"""Forward pass of the score-based model.

        Args:
          x: The input functional data as a tensor, shape (N, M) or (N, 1, M).
          t: The time steps as a tensor, shape (N,).
          y: Optional tensor with the class labels of the data, shape (N,).
            Default is `None`.

        Returns:
          Approximates the score :math:`\nabla_x \log p_t(x)` (or a
          noise-scaled version when ``multiply_inv_sigma`` is set),
          shape (N, M).
        """
        ...

    def to_checkpoint(self) -> "ScoreCheckpointDict":
        """Serialize the full model state to a self-describing checkpoint.

        The checkpoint contains everything needed to reconstruct this instance
        from scratch via :meth:`from_checkpoint`: the concrete class object,
        the constructor arguments, and the weights state dict.  Subclasses
        achieve custom serialization by setting ``self._init_kwargs`` in
        their ``__init__``; this method should not be overridden.

        Returns:
            ScoreCheckpointDict: A dictionary with three keys:

            ``"class"``
                The concrete class of this instance.
            ``"init_kwargs"``
                Constructor keyword arguments set by the subclass in
                ``self._init_kwargs``.
            ``"fit_state"``
                PyTorch state dict of the model weights (all tensors moved
                to CPU so the checkpoint is device-independent).

        Raises:
            AttributeError: If the subclass did not set ``self._init_kwargs``
                in its ``__init__``.
        """
        if not hasattr(self, "_init_kwargs"):
            msg = (
                f"{type(self).__name__} must assign self._init_kwargs in "
                "__init__ to support checkpointing. "
                "Example: self._init_kwargs = {'n_points': n_points, ...}"
            )
            raise AttributeError(
                msg,
            )
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
        }
        return {
            "class": type(self),
            "init_kwargs": self._init_kwargs,
            "fit_state": state_dict,
        }


    def _restore_fit_state(self, data: dict[str, ScoreStateType]) -> None:
        """Restore model weights from a serialized state dictionary.

        Calls :meth:`torch.nn.Module.load_state_dict` with ``strict=True``.
        Subclasses almost never need to override this.

        Args:
            data: The ``"fit_state"`` sub-dictionary from a checkpoint
                produced by :meth:`to_checkpoint` (i.e. a PyTorch state dict).

        Raises:
            ValueError: If the state dict is incompatible with the current
                model architecture.
        """
        try:
            self.load_state_dict(data, strict=True)
        except RuntimeError as exc:
            msg = (
                "Checkpoint state dict is incompatible with the current score "
                "model architecture. Ensure the same architecture was used "
                "when saving."
            )
            raise ValueError(
                msg,
            ) from exc

    def on_load(self, diff_process: "ForwardDiffusionProcess") -> None:
        """Hook called by FunctionalDiffusionGenerator.load after restoring.

        Subclasses that depend on a callable from the diffusion process
        (e.g. multiply_inv_sigma) should override this to re-attach it.
        The default implementation does nothing.

        Args:
            diff_process: The fully restored diffusion process.
        """

    @classmethod
    def from_checkpoint(cls, data: ScoreCheckpointDict) -> "ScoreModel":
        """Reconstruct a fitted score model from a checkpoint dictionary.

        The concrete subclass is read directly from ``data["class"]`` (a
        class object stored by pickle), so the correct type is always
        instantiated regardless of which class this method is called on.

        Args:
            data: A checkpoint dictionary as produced by :meth:`to_checkpoint`.

        Returns:
            ScoreModel: A fully reconstructed instance in ``eval`` mode,
            on CPU.  Move to the desired device with ``.to(device)`` after
            calling this method.

        Raises:
            ValueError: If required keys are missing or the stored class is
                not a subclass of :class:`ScoreModel`.
        """
        required = {"class", "init_kwargs", "fit_state"}
        missing = required - data.keys()
        if missing:
            msg = f"Checkpoint is missing required keys: {sorted(missing)}."
            raise ValueError(
                msg,
            )

        model_cls = data["class"]
        if not (
            isinstance(model_cls, type) and issubclass(model_cls, ScoreModel)
        ):
            msg = (
                "Expected 'class' to be a subclass of ScoreModel, "
                f"got {model_cls!r}."
            )
            raise TypeError(
                msg,
            )

        model = model_cls(**data["init_kwargs"])
        model._restore_fit_state(data["fit_state"])  # noqa: SLF001 # Ignore private method
        model.eval()
        return model



class Swish(nn.Module):
    """Module form of swish activation to keep the model serializable."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply swish activation."""
        return x * torch.sigmoid(x)

class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time steps."""

    def __init__(
        self,
        n_points: int,
        scale: float = 30.0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        # Fixed RFF weights: sampled once at init, not updated during training.
        self.rff_weights = nn.Parameter(
            torch.randn(n_points // 2, device=device) * scale,
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Compute random Fourier feature embeddings for scalar inputs.

        .. math::

            \varphi(t) = \bigl[\sin(2\pi w_1 t),\ldots,\sin(2\pi w_k t),\;
                                \cos(2\pi w_1 t),\ldots,\cos(2\pi w_k t)\bigr]

        where :math:`w_i \sim \mathcal{N}(0,\,\text{scale}^2)` are the
        fixed weights stored in ``rff_weights``.
        """
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * torch.pi
        return  torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize a dense layer."""
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Linear layer; appends a size-1 trailing dim for broadcasting."""
        result: Tensor = self.dense(x)
        return result[..., None]

class UNetScoreModel(ScoreModel):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        multiply_inv_sigma: Callable[[Tensor, Tensor], Tensor] | None = None,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (9, 9, 9, 9),
        n_groups: tuple[int, ...] = (4, 32, 32, 32),
        n_points: int = 100,
        device: str | torch.device = "cpu",
    ) -> None:
        r"""Initialize a time-dependent score-based network.

        Args:
          multiply_inv_sigma: A callable ``(h, t) -> h_scaled`` that applies
              the inverse square-root covariance operator to ``h``. For
              diagonal/scalar processes this is pointwise multiplication;
              for circulant processes it applies
              :math:`Q\,\operatorname{diag}(\sigma_t^{-1})\,Q^\top` via FFT.
              If ``None``, no scaling is applied. Default is ``None``.
          channels: The number of channels for feature maps of each resolution.
          n_points: The dimensionality of Gaussian random Fourier feature
          embeddings.
          kernel_sizes: The kernel sizes for the convolutional layers.
          n_groups: The number of groups for group normalization in each layer.
          device: The device on which to initialize model parameters.
        """
        super().__init__()

        self._init_kwargs = {
        "channels": channels,
        "kernel_sizes": kernel_sizes,
        "n_groups": n_groups,
        "n_points": n_points,
        "device": "cpu",           # device-agnostic; caller moves after load
        "multiply_inv_sigma": None, # not picklable; restored via on_load
        }


        self.device = device
        self.n_points = n_points
        self.channels = channels
        self.n_groups = n_groups
        # Gaussian random Fourier feature embedding layer for time
        n_points_even = n_points + (n_points % 2)  # must be even for sin/cos
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(
                n_points=n_points_even, device=device,
            ),
            nn.Linear(n_points_even, n_points),
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv1d(
            1,
            channels[0],
            kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0]//2,
            bias=False,
        )
        self.dense1 = Dense(n_points, channels[0])
        self.gnorm1 = nn.GroupNorm(n_groups[0], num_channels=channels[0])
        self.conv2 = nn.Conv1d(
            channels[0],
            channels[1],
            kernel_sizes[1],
            stride=4,
            padding=kernel_sizes[1]//2,
            bias=False,
        )
        self.dense2 = Dense(n_points, channels[1])
        self.gnorm2 = nn.GroupNorm(n_groups[1], num_channels=channels[1])
        self.conv3 = nn.Conv1d(
            channels[1],
            channels[2],
            kernel_sizes[2],
            stride=4,
            padding=kernel_sizes[2]//2,
            bias=False,
        )
        self.dense3 = Dense(n_points, channels[2])
        self.gnorm3 = nn.GroupNorm(n_groups[2], num_channels=channels[2])
        self.conv4 = nn.Conv1d(
            channels[2],
            channels[3],
            kernel_sizes[3],
            stride=4,
            padding=kernel_sizes[3]//2,
            bias=False,
        )
        self.dense4 = Dense(n_points, channels[3])
        self.gnorm4 = nn.GroupNorm(n_groups[3], num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose1d(
            channels[3],
            channels[2],
            kernel_size=kernel_sizes[3],
            padding=kernel_sizes[3]//2,
            stride=kernel_sizes[3]//2,
            bias=False,
            output_padding=kernel_sizes[3]//2-1,
        )
        self.dense5 = Dense(n_points, channels[2])
        self.tgnorm4 = nn.GroupNorm(n_groups[3], num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose1d(
            channels[2] + channels[2],
            channels[1],
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2]//2,
            stride=kernel_sizes[2]//2,
            bias=False,
            output_padding=kernel_sizes[2]//2-1,
        )
        self.dense6 = Dense(n_points, channels[1])
        self.tgnorm3 = nn.GroupNorm(n_groups[2], num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose1d(
            channels[1] + channels[1],
            channels[0],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1]//2,
            stride=kernel_sizes[1]//2,
            bias=False,
            output_padding=kernel_sizes[1]//2-1,
        )
        self.dense7 = Dense(n_points, channels[0])
        self.tgnorm2 = nn.GroupNorm(n_groups[1], num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose1d(
            channels[0] + channels[0],
            1,
            kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            stride=1,
            bias=False,
        )

        # Keep activation as an nn.Module so checkpoints remain serializable.
        self.act = Swish()
        self.multiply_inv_sigma = multiply_inv_sigma
        self.to(device)

    def on_load(self, diff_process: "ForwardDiffusionProcess") -> None:
        """Re-attach multiply_inv_sigma from the restored diffusion process.

        This is called by FunctionalDiffusionGenerator.load after both the
        diff process and the score model have been reconstructed. It restores
        the callable that was excluded from the checkpoint to avoid pickling
        problems with non-picklable diff processes.
        """
        self.multiply_inv_sigma = diff_process.multiply_inv_sigma

    def forward(self, x:Tensor, t:Tensor, y:Tensor | None = None) -> Tensor:  # noqa: ARG002 # Ignore unused y
        """Forward pass of the score-based model.

        Args:
          x: The input functional data as a tensor, shape (N, M) or (N, 1, M).
          t: The time steps as a tensor, shape (N,).
          y: Optional tensor with the class labels of the data, shape (N,).
             Default is `None`.

        Returns:
          The output of the score-based model, shape (N, M).
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        embed = self.act(self.embed(t))
        # Add channel dimension if input does not have it
        x_in = x.unsqueeze(1) if x.dim() == 2 else x  # noqa: PLR2004 # Ignore magic number

        h1 = self.conv1(x_in)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)

        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        h = self._match_size(h, h3)
        # Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self._match_size(h, h2)

        # Skip connection from the encoding path
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self._match_size(h, h1)

        # Skip connection from the encoding path
        h = self.tconv1(torch.cat([h, h1], dim=1))
        # remove the channel dim added at input
        h = h.squeeze(1)

        if self.multiply_inv_sigma is not None:
            # whiten: pre-condition by the inverse noise std
            h = self.multiply_inv_sigma(h, t)

        if h.isnan().any():
            msg = "NaN values encountered in score network output."
            raise ValueError(msg)

        # Add channel dimension back if input had it
        out: Tensor = h.unsqueeze(1) if x.dim() != h.dim() else h
        return out

    def _match_size(self, x: Tensor, target: Tensor) -> Tensor:
        """Crop or pad x to match target's spatial dimensions."""
        if x.shape[-1] != target.shape[-1]:
            # Center crop or pad
            diff = x.shape[-1] - target.shape[-1]
            if diff > 0:
                # Crop from center
                start = diff // 2
                return x[..., start:start + target.shape[-1]]

            # Pad symmetrically
            pad_total = -diff
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return nn.functional.pad(x, (pad_left, pad_right))
        return x

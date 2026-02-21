"""
Created on Wed Feb  4 10:04:58 2026.

Code adapted by alberto.suarez@uam.es from
https://yang-song.net/blog/2021/score/


"""

import torch
from torch import nn
from torch import Tensor
from typing import Callable


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time steps."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * torch.pi
        gaus =  torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return gaus


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)[..., None]


class ScoreModel(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        inv_sigma_t: Callable[[Tensor], Tensor],
        channels: tuple[int] = (32, 64, 128, 256),
        embed_dim: int = 100, device: str | torch.device = "cpu",
    ):
        """Initialize a time-dependent score-based network.

        Args:
          inv_sigma_t: A function that takes time t and gives the inverse of
                       the square root of the covariance matrix at time t.
                       As a tensor with shapes (N, M, M) for a batch size
                       of N and data dimension M, (N,M) in which case it is
                       assumed to be a (N,M,M) diagonal matrix where only the
                       main diagonal is returned or a scalar in which case
                       it is assumed to be a multiple of the identity matrix.
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random Fourier feature
          embeddings.
        """
        kernel_sizes = (17, 9, 9, 5)
        super().__init__()
        self.device = device
        # Gaussian random Fourier feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
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
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv1d(
            channels[0],
            channels[1],
            kernel_sizes[1],
            stride=4,
            padding=kernel_sizes[1]//2,
            bias=False,
        )
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv1d(
            channels[1],
            channels[2],
            kernel_sizes[2],
            stride=4,
            padding=kernel_sizes[2]//2,
            bias=False,
        )
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv1d(
            channels[2],
            channels[3],
            kernel_sizes[3],
            stride=4,
            padding=kernel_sizes[3]//2,
            bias=False,
        )
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose1d(
            channels[3],
            channels[2],
            9,
            padding=3,
            stride=4,
            bias=False,
            output_padding=3,
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose1d(
            channels[2] + channels[2],
            channels[1],
            9,
            padding=3,
            stride=4,
            bias=False,
            output_padding=3,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose1d(
            channels[1] + channels[1],
            channels[0],
            9,
            padding=3,
            stride=4,
            bias=False,
            output_padding=3,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose1d(
            channels[0] + channels[0], 1, 9, padding=4, stride=1,
        )

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.inv_sigma_t = inv_sigma_t
        self.to(device)

    def forward(self, x:Tensor, t:Tensor, y:Tensor | None = None) -> Tensor:
        """Forward pass of the score-based model.

        Args:
          x: The input functional data as a tensor, shape (N, M) or (N, 1, M).
          t: The time steps as a tensor, shape (N,).
          y: Optional tensor with the class labels of the data, shape (N,).
             Default is `None`.

        Returns:
          The output of the score-based model, shape (N, M).
        """
        # Obtain the Gaussian random Fourier feature embedding for t
        embed = self.act(self.embed(t))
        # Add channel dimension if input does not have it
        if x.dim() == 2:
            x_in = x.unsqueeze(1)
        else:
            x_in = x

        h1 = self.conv1(x_in)

        # Incorporate information from t
        h1 += self.dense1(embed)
        # Group normalization
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
        # Normalize output
        # Remove channel dimension (N,1,M) -> (N,M)
        h = h.squeeze(1)
        p = 1.
        if self.inv_sigma_t is not None:
            p = self.inv_sigma_t(t)

        if p.dim() == 1:
            # (N,) shape
            p = p.unsqueeze(1)  # Shape (N,1)

        if p.dim() in (0, 2):
            # Scalar, (N,1) or (N,M) shape
            # Element-wise multiplication
            h = h * p
        elif p.dim() == 3:
            # (N, M, M) shape
            # Batched matrix vector product
            h = torch.einsum("nij,ni->nj", p, h)

        # TODO(): Consider if we want to raise an error if NaN values are encountered in the output
        if h.isnan().any():
            raise ValueError("NaN values encountered in score network output.")
        # Add channel dimension back if input had it
        return h.unsqueeze(1) if x.dim() != h.dim() else h

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
        # Sizes match
        return x

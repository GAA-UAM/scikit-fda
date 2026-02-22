from .backward_integrator import EulerMaruyamaBackwardIntegrator

from ..._utils._sklearn_adapter import BaseEstimator
from ...representation.grid import FDataGrid
from ...typing._numpy import NDArrayFloat

from .diffusion_process import ForwardDiffusionProcess, VariancePreservingDiffusionProcess
from .score_model import ScoreModel, ScoreModelBig
from .reverse_diffusion import ReverseDiffusionProcess, SDEReverseDiffusionProcess, EulerMaruyamaIntegrator

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import tqdm # TODO(): Delete this when done testing


class FDataGenerator(BaseEstimator):
    """Class that implements a diffusion model for functional data.

    Using a score-based diffusion model, it is possible to generate new
    samples of functional data.

    For now, only 1D functional data is supported. The model can be trained
    on any 1D functional data, but the sampling process is not conditioned on
    class labels or any other information about the data. The generated samples
    will be from the same distribution as the training data, but there is no
    guarantee on the quality of the generated samples.

    """
    # TODO(): How should I set the default diffusion process?
    # TODO(): Decide what to do with device
    def __init__(
        self,
        diff_process: ForwardDiffusionProcess = VariancePreservingDiffusionProcess(),
        normalize: bool = True,
        scale_by_inv_sigma: bool = True,
        max_iter: int = 200,
        n_jobs: int = 0,
        device: torch.device | str = "cpu",
    ):
        """Initializes the diffusion model for functional data.

        Args:
            diff_process: An instance of a DiffusionProcess that defines
                the diffusion process to be used. It defaults to
                VariancePreservingDiffusionProcess.
            normalize: Whether to normalize the data to [-1, 1] before training
                the diffusion model. Default is True.
            scale_by_inv_sigma: Whether to scale the score by the inverse of the
                noise level sigma_t during training. This can help stabilize
                training when the noise level is very small. Default is True.
            max_iter: The maximum number of iterations (epochs) to train the
                score model. Default is 200.
            n_jobs: The number of worker processes to use for data loading.
                Default is 0 (the main process will be used). If set to -1, the
                number of workers will be set to the number of CPU cores available.
            device: The device to use for training and sampling. Default is "cpu".
        """
        super().__init__()
        self.diff_process = diff_process
        self.normalize = normalize
        self.scale_by_inv_sigma = scale_by_inv_sigma
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.device = device
        self._bias = 0.0  # Normalization bias
        self._scale = 1.0  # Normalization scale

        self.batch_size = 32  # TODO(): Decide whether to set as attribute or argument

    def fit(
        self,
        X: FDataGrid,
        y: NDArrayFloat | None = None,
    ) -> "FDataGenerator":
        """Fits the diffusion model to the provided functional data.

        Args:
            X: The input functional data as an FDataGrid object.
            y: Optional tensor with the class labels of the data.
               Default is `None`.
            n_epochs: The number of epochs to train the model. Default is 200.
            n_workers: Number of workers to use for data loading.
                Default is 0 (The main process will be used).
                If set to -1, the number of workers will be set to
                the number of CPU cores available.

        Returns:
            The fitted FDataGenerator instance.
        """
        # Check that data is 1D
        if X.dim_domain != 1 or X.dim_codomain != 1:
            raise ValueError("FDataGenerator only supports 1D functional data.")
        # Find the number of discretization points
        # As this is fo 1D data, there is only one grid_points array
        self.grid_points = X.grid_points[0]
        grid_size = len(self.grid_points)

        self.score_model = ScoreModelBig(
            embed_dim=grid_size,
            channels=(32, 64, 128, 256),
            n_groups=(4, 32, 32, 32),
            inv_sigma_t=self.diff_process.inv_sigma_cond if self.scale_by_inv_sigma else None,
            device=self.device,
        )
        # Preprocess the data and create a DataLoader for training
        dataset = self._preprocess(X, y)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
        )
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(
            self.score_model.parameters(),
            lr=learning_rate,
        )

        # Fit the diffusion process
        # TODO: Cambiar esto para que no se ajuste solo a un batch si no a
        # todo el dataset
        # Decide if the FDataGrid is passed ot a preprocess data.
        data_elem= next(iter(data_loader))
        x = data_elem[0] if isinstance(data_elem, (list, tuple)) else data_elem
        self.diff_process.fit(x)

        # Add tqdm for progress bar
        # TODO(): DELETE THIS WHEN DONE TESTING
        tqdm_epoch = tqdm.trange(self.max_iter)

        eps = 1e-3
        # Training loop
        for _ in tqdm_epoch:
            # Epoch metrics
            avg_loss = 0.0
            num_items = 0
            for x in data_loader:
                # Train step
                # TODO(): When decide if to use tensor dataset or custom dataset,
                x = x[0] if isinstance(x, (list, tuple)) else x
                # Unpack if dataset returns (data, labels) or just data
                # Change this to unpack
                x_dev = x.to(self.device)

                # Generate a random time step for each sample in the batch
                # TODO(): Is this the best way to sample, I don't like to use T here
                # Sample from uniform(0, T-eps)
                t = (
                        torch.rand(x_dev.shape[0], device=self.device) *
                        (self.diff_process.T - eps)
                    ) + eps

                loss = self._loss_function(self.score_model, x_dev, t)
                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.score_model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                # Update epoch metrics
                avg_loss += loss.item() * x_dev.shape[0]
                num_items += x_dev.shape[0]

            tqdm_epoch.set_description(
                            f"Average Loss: {avg_loss / num_items:5f}",
                                     )
        return self

    def _preprocess(
            self,
            X: FDataGrid,
            y: NDArrayFloat | None = None,
    ) -> Tensor:
        """Preprocesses the functional data before training or sampling.

        Args:
            X: The input functional data as an FDataGrid object.
            y: Optional tensor with the class labels of the data.
                Default is `None`.

        Returns:
            A TorchDataset with the preprocessed data.
        """
        if self.normalize:
            eps = 1e-2 # TODO(): Decide how to handle small scale values
            # Maybe set the minimum scale to 1.0 directly

            # Normalize the data to [-1, 1]
            max_fd = X.data_matrix.max()
            min_fd = X.data_matrix.min()
            self._bias = (max_fd + min_fd) / 2
            self._scale = ((max_fd - min_fd) / 2)

            if self._scale < eps:
                self._scale = 1.0

            X_normalized = (X - self._bias) / self._scale
            dataset = from_FDataGrid_to_TensorDataset(X_normalized, y)

            #dataset = FDataGridDataset(X_normalized, y)
        else:
            dataset = from_FDataGrid_to_TensorDataset(X, y)
            #dataset = FDataGridDataset(X, y)
        return dataset

    def _loss_function(
            self,
            score_model: ScoreModel,
            x: Tensor,
            t: Tensor,
    ) -> Tensor:
        """Computes the loss function for training the score-based model.

        Args:
            score_model: The score-based model to be trained.
            x: The input functional data as a tensor, shape (N, M) or
               (N, 1, M).
            t: The time steps for each sample in the batch, shape (N,).

        Returns:
            The computed loss as a tensor.
        """
        mu_t = self.diff_process.mean_cond(x, t)
        sigma_t = self.diff_process.sigma_cond(t)
        z = torch.randn_like(x)  # Shape (N, M)
        # TODO(): Create a function that does this
        if sigma_t.dim() == 1:
            # (N,) shape
            sigma_t = sigma_t.unsqueeze(1)  # Shape (N,1)

        if sigma_t.dim() in (0, 2):
            # Scalar, (N,1) or (N,M) shape
            # Element-wise multiplication
            score_time_z = sigma_t * z
        elif sigma_t.dim() == 3:
            # (N, M, M) shape
            # Batched matrix vector product
            score_time_z = torch.einsum("nij,ni->nj", sigma_t, z)

        x_t = mu_t + score_time_z
        score = score_model(x_t, t)  # Shape (N, M)

        if sigma_t.dim() == 1:
            # (N,) shape
            sigma_t = sigma_t.unsqueeze(1)  # Shape (N,1)

        if sigma_t.dim() in (0, 2):
            # Scalar, (N,1) or (N,M) shape
            # Element-wise multiplication
            score_time_sigma = score * sigma_t
        elif sigma_t.dim() == 3:
            # (N, M, M) shape
            # Batched matrix vector product
            score_time_sigma = torch.einsum("nij,ni->nj", sigma_t, score)

        return torch.mean(
            torch.sum(
                (score_time_sigma + z) ** 2,
                dim=1,
            ),
        )

    def generate(
            self,
            n_samples: int,
            reverse_process: ReverseDiffusionProcess = SDEReverseDiffusionProcess(EulerMaruyamaIntegrator()),
            y: NDArrayFloat | None = None,
    ) -> FDataGrid:
        """Generates new samples of functional data using the diffusion model.

        If class labels are provided, the generated samples will be conditioned
        on those labels and n_samples will be ignored.

        If the model has not been fitted yet, a RuntimeError is raised.

        Args:
            n_samples: The number of samples to generate.
            reverse_process: The reverse diffusion process to use for generating samples.
            y: Optional tensor with the class labels of the data.
               Default is `None`.


        Returns:
            An FDataGrid object containing the generated functional data.
        """
        if not hasattr(self, "score_model"):
            raise RuntimeError("The model must be fitted before generating samples.")

        if y is not None:
            n_samples = len(y)
        # Sample from the final distribution of the diffusion process
        x_t = self.diff_process.sample_limit_distribution(
            n_samples,
            self.device,
        )
        # Reverse diffusion process to generate samples
        # No gradients needed during generation; this avoids storing
        # the computation graph and significantly reduces memory usage.
        with torch.no_grad():
            x_0 = reverse_process.reverse(
                self.diff_process,
                self.score_model,
                x_t,
                self.diff_process.T,
                y,
            )
            
        # Denormalize the data if normalization was applied

        # Convert the generated samples to an FDataGrid object
        data_matrix = x_0.detach().cpu().numpy().reshape(n_samples, -1, 1)

        f_data = FDataGrid(
            data_matrix=data_matrix,
            grid_points=self.grid_points,
        )

        if self.normalize:
            f_data = f_data * self._scale + self._bias
        return f_data

# TODO(): Decide whether a TensorDataset is better or a custom Dataset class
def from_FDataGrid_to_TensorDataset(X: FDataGrid, y: NDArrayFloat | None = None) -> torch.utils.data.TensorDataset:
    r"""Converts an FDataGrid object to a PyTorch TensorDataset.

    The functional data must be a :math:`\mathbb{R}^1`-valued function.

    Args:
        X: The input functional data as an FDataGrid object.
        y: Optional tensor with the class labels of the data.
           Default is `None`.

    Returns:
        A PyTorch TensorDataset containing the functional data.
    """
    # Convert the FDataGrid data to a PyTorch tensor
    data_tensor = torch.from_numpy(X.data_matrix[..., 0]).float()
    if y is not None:
        labels_tensor = torch.from_numpy(y).float()
        # Create a TensorDataset with data and labels
        dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    else:
        dataset = torch.utils.data.TensorDataset(data_tensor)
    return dataset

class FDataGridDataset(torch.utils.data.Dataset):
    r"""PyTorch Dataset for FDataGrid objects.

    The functional data must be a :math:`\mathbb{R}^1`-valued function.

    Args:
        X: The input functional data as an FDataGrid object.
    """
    def __init__(self, X: FDataGrid, y:NDArrayFloat | None = None):
        """Initializes the FDataGridDataset."""
        # Convert the FDataGrid data to a PyTorch tensor
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self,
                    idx: int,
                    ) -> Tensor | tuple[Tensor, Tensor]:
        """Returns the sample at the given index."""
        tensor = torch.from_numpy(self.X.data_matrix[idx, ..., 0]).float()
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return tensor, label
        return tensor

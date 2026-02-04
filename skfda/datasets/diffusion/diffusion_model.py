from ..._utils._sklearn_adapter import BaseEstimator
from ...representation.grid import FDataGrid
from ...typing._numpy import NDArrayFloat

from diffusion_process import DiffusionProcess, VariancePreservingDiffusionProcess
from score_model import ScoreNet

import torch
from torch.utils.data import DataLoader

import tqdm # TODO(): Delete this when done testing


class FData1DGenerator(BaseEstimator):
    """Class that implements a diffusion model for 1D functional data.

    Using a score-based diffusion model, it is possible to generate new
    samples of functional data.
    """
    #TODO(): How should I set the default diffusion process?
    # TODO(): Decide what to do with device
    def __init__(self, 
                 diff_process: DiffusionProcess = VariancePreservingDiffusionProcess(),
                 normalize: bool = True,
                 device: torch.device | str = "cpu",
                ):
        """Initializes the diffusion model for 1D functional data.

        Args:
            diff_process: An instance of a DiffusionProcess that defines
                the diffusion process to be used. It defaults to
                VariancePreservingDiffusionProcess.
            normalize: Whether to normalize the data to [-1, 1] before training
                the diffusion model. Default is True.
        """
        super().__init__()
        self.diff_process = diff_process
        self.normalize = normalize
        self.device = device

        self._bias = 0.0 #Normalization bias
        self._scale = 1.0 #Normalization scale

        self.batch_size = 32 # TODO(): Decide whether to set as attribute or argument

    # TODO(): Decide if n_epochs should be an argument of fit or a class 
    # attribute also decide the the name of the atribute, sklearn uses 
    # max_iter and is a class attribute
    def fit(self, X: FDataGrid, y:NDArrayFloat | None = None, n_epochs: int = 200, n_workers: int = 0,) -> "FData1DGenerator":
        """Fits the diffusion model to the provided functional data.

        Args:
            X: The input functional data as an FDataGrid object.
            y: Optional tensor with the class labels of the data.
               Default is `None`.
            n_workers: Number of workers to use for data loading.
                Default is 0 (The main process will be used).
                If set to -1, the number of workers will be set to 
                the number of CPU cores available.

        Returns:
            The fitted FData1DGenerator instance.
        """
        # Check that data is 1D
        if X.dim_domain != 1 or X.dim_codomain != 1:
            raise ValueError("FData1DGenerator only supports 1D functional data.")
        # Find the number of discretization points
        self.grid_points = X.grid_points
        grid_size = len(X.grid_points[0])
        self.score_model = ScoreNet(embed_dim=grid_size,
                                     inv_std=self.diff_process.inv_sigma_t,
                                     device=self.device)
        dataset = self._preprocess(X, y)
        data_loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=n_workers)
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.score_model.parameters(), lr=learning_rate)
        
        # Add tqdm for progress bar
        # TODO(): DELETE THIS WHEN DONE TESTING
        tqdm_epoch = tqdm.trange(n_epochs)

        eps = 1e-9
        # Training loop
        for _ in tqdm_epoch:
            # Epoch metrics
            avg_loss = 0.0
            num_items = 0
            for x in data_loader:
                # Train step
                x = x.to(self.device)
                # Generate a random time step for each sample in the batch
                # TODO(): Is this the best way to sample, I don't like to use T here
                # Sample from uniform(0, T-eps)
                t = torch.rand(x.shape[0], device=self.device) * (self.diff_process.T - eps)
                loss = self._loss_function(self.score_model, x, t)
                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Update epoch metrics
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            tqdm_epoch.set_description(
                            "Average Loss: {:5f}".format(avg_loss / num_items)
                                     )
        return self
    
    def _preprocess(self, X: FDataGrid, y:NDArrayFloat | None = None) -> torch.Tensor:
        """Preprocesses the functional data before training or sampling.

        Args:
            X: The input functional data as an FDataGrid object.
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

            dataset = FDataGridDataset(X_normalized, y)
        else:
            dataset = FDataGridDataset(X, y)
        return dataset

    def _loss_function(self, score_model: ScoreNet, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the loss function for training the score-based model.

        Args:
            score_model: The score-based model to be trained.
            x: The input functional data as a tensor, shape (N, M) or (N, 1, M).

        Returns:
            The computed loss as a tensor.
        """
        x_t_state = self.diff_process.forward(x, t)
        x_t = x_t_state.x_t  # Shape (N, M)
        z = x_t_state.z  # Shape (N, M)
        sigma_t = x_t_state.sigma_t  # Shape (N,)

        score = score_model(x_t, t)  # Shape (N, M)

        return torch.mean(
            torch.sum(
                (score * sigma_t + z) ** 2,
                dim=1,
            ),
        )

    def generate(self, n_samples: int, y: NDArrayFloat | None = None) -> FDataGrid:
        """Generates new samples of functional data using the diffusion model.

        If class labels are provided, the generated samples will be conditioned
        on those labels and n_samples will be ignored.

        If the model has not been fitted yet, a RuntimeError is raised.

        Args:
            n_samples: The number of samples to generate.
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
        x_t = self.diff_process.sample_final_distribution(n_samples,
                                                          self.device)

        # Reverse diffusion process to generate samples
        x_0_state = self.diff_process.backward(self.score_model, x_t, y)

        # Denormalize the data if normalization was applied
        if self.normalize:
            x_0 = x_0_state.x * self._scale + self._bias
        else:
            x_0 = x_0_state.x
        # Convert the generated samples to an FDataGrid object
        data_matrix = x_0.cpu().numpy().reshape(n_samples, -1, 1)

        return FDataGrid(data_matrix=data_matrix,
                         grid_points=self.grid_points)

# TODO(): Decide whether a TensorDataset is better or a custom Dataset class
def from_FDataGrid_to_TensorDataset(X: FDataGrid, y:NDArrayFloat | None = None) -> torch.utils.data.TensorDataset:
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

    def __getitem__(self, idx: int):
        """Returns the sample at the given index."""
        tensor = torch.from_numpy(self.X.data_matrix[idx, ..., 0]).float()
        if self.y is not None:
            label = torch.from_numpy(self.y[idx]).float()
            return tensor, label
        return tensor

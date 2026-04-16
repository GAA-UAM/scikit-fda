from skfda.datasets.diffusion.torch_adapter import make_torch_generator

from ..._utils._sklearn_adapter import BaseEstimator
from ...representation.grid import FDataGrid
from ...typing._numpy import NDArrayFloat
from ...typing._base import RandomStateLike

from .diffusion_process import ForwardDiffusionProcess, VariancePreservingDiffusionProcess
from .score_model import ScoreModel, ScoreModelConv
from .reverse_diffusion import ReverseDiffusionProcess, SDEReverseDiffusionProcess, EulerMaruyamaIntegrator, ProbabilityFlowODEReverseProcess, RK4Integrator

from ...exploratory.stats import std, mean


from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.utils.validation import check_is_fitted

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
        diff_process: ForwardDiffusionProcess | None = None,
        score_model: torch.nn.Module | None = None,
        normalize: bool = True,
        standardize: bool = False,
        max_iter: int = 200,
        batch_size: int = 32,
        n_jobs: int = 0,
        device: torch.device | str = "cpu",
        random_state: RandomStateLike = None,
    ):
        """Initializes the diffusion model for functional data.

        Args:
            diff_process: An instance of a DiffusionProcess that defines
                the diffusion process to be used. It defaults to
                VariancePreservingDiffusionProcess.
            score_model: Optional score model to train. It must implement
                ``forward(x, t, y=None)`` and return a tensor with the same
                shape as ``x``. If ``None``, a default ``ScoreModel`` is built
                during ``fit``.
            normalize: Whether to normalize the data to [-1, 1] before training
                the diffusion model. Default is True.
            standardize: Whether to standardize the data to have zero mean and unit variance
                before training the diffusion model. Default is False.
            max_iter: The maximum number of iterations (epochs) to train the
                score model. Default is 200.
            batch_size: The number of samples per batch to load at each training 
                step. Default is 32.
            n_jobs: The number of worker processes to use for data loading.
                Default is 0 (the main process will be used). If set to -1, the
                number of workers will be set to the number of CPU cores available.
            device: The device to use for training and sampling. Default is "cpu".
            random_state: The random state to use for reproducible results. Default is None.
        """
        super().__init__()
        self.diff_process = (
            diff_process
            if diff_process is not None
            else VariancePreservingDiffusionProcess()
        )
        self.score_model = score_model
        self.normalize = normalize
        self.standardize = standardize
        if self.normalize and self.standardize:
            raise ValueError("normalize and standardize cannot both be True.")
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        self.random_state = random_state
        self.torch_generator = make_torch_generator(random_state=random_state, device=device)


    def _build_score_model(self, embed_dim: int) -> torch.nn.Module:
        """Create a score model using the current estimator configuration."""
        if self.score_model is not None:
            if not isinstance(self.score_model, torch.nn.Module):
                raise TypeError("score_model must be a torch.nn.Module or None.")
            return self.score_model.to(self.device)


        return ScoreModelConv(
            embed_dim=embed_dim,
            inv_sigma_t=self.diff_process.inv_sigma_cond,
            device=self.device,
            random_state=self.random_state,
        )

    def _validate_score_model_output(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor | None = None,
    ) -> None:
        """Validate score model compatibility using a lightweight dry run."""
        try:
            out = self.score_model_(x, t, y)
        except TypeError:
            out = self.score_model_(x, t)

        if not isinstance(out, Tensor):
            raise TypeError("score_model must return a torch.Tensor.")

        if out.shape != x.shape:
            raise ValueError(
                "score_model output shape must match input x shape "
                f"({tuple(x.shape)}), got {tuple(out.shape)}.",
            )

    def _ensure_diff_process_fitted_for_dimension(self, n_features: int) -> None:
        """Fit the diffusion process if needed, using only dimensional info."""
        if getattr(self.diff_process, "M", None) == n_features:
            return

        x_probe = torch.zeros((1, n_features), device=self.device)
        self.diff_process.fit(x_probe)

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
        self.grid_points_ = X.grid_points[0]
        self.grid_points = self.grid_points_
        grid_size = len(self.grid_points_)

        self.score_model_ = self._build_score_model(embed_dim=grid_size)
        self.score_model = self.score_model_

        params = list(self.score_model_.parameters())
        if not params:
            raise ValueError(
                "score_model must define at least one trainable parameter.",
            )

        self.score_model_.train()

        # Preprocess the data and create a DataLoader for training
        dataset = self._preprocess(X, y)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            generator=self.torch_generator,
        )
        data_elem = next(iter(data_loader))
        x = data_elem[0] if isinstance(data_elem, (list, tuple)) else data_elem
        x_dev = x.to(self.device)
        t_val = torch.full((x_dev.shape[0],), 0.5, device=self.device)
        y_val = None
        if isinstance(data_elem, (list, tuple)) and len(data_elem) > 1:
            y_val = data_elem[1].to(self.device)

        self._validate_score_model_output(x_dev, t_val, y_val)

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(
            self.score_model_.parameters(),
            lr=learning_rate,
        )

        # Fit the diffusion process
        # TODO: Cambiar esto para que no se ajuste solo a un batch si no a
        # todo el dataset
        # Decide if the FDataGrid is passed ot a preprocess data.
        self.diff_process.fit(x_dev)

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
                # Sample from uniform(eps, T)
                t = (
                        torch.rand(x_dev.shape[0], device=self.device, generator=self.torch_generator) *
                        (self.diff_process.T - eps)
                    ) + eps

                loss = self._loss_function(self.score_model_, x_dev, t)
                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.score_model_.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                # Update epoch metrics
                avg_loss += loss.item() * x_dev.shape[0]
                num_items += x_dev.shape[0]

            tqdm_epoch.set_description(
                            f"Average Loss: {avg_loss / num_items:5f}",
                                     )
        self.score_model_.eval()
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
            self._bias_ = (max_fd + min_fd) / 2
            self._scale_ = ((max_fd - min_fd) / 2)

            if self._scale_ < eps:
                self._scale_ = 1.0
                self._scale = self._scale_

            X_normalized = (X - self._bias_) / self._scale_
            dataset = from_FDataGrid_to_TensorDataset(X_normalized, y)

            #dataset = FDataGridDataset(X_normalized, y)
        elif self.standardize:
            eps = 1e-2 # TODO(): Decide how to handle small scale values
            # Maybe set the minimum scale to 1.0 directly

            # Standardize the data to have zero mean and unit variance
            mean_fd = mean(X)
            std_fd = std(X)
            self._bias_ = mean_fd
            self._scale_ = std_fd
            unsafe_mask = self._scale_.data_matrix < eps
            self._scale_.data_matrix[unsafe_mask] = 1.0


            X_standardized = (X - self._bias_) / self._scale_
            dataset = from_FDataGrid_to_TensorDataset(X_standardized, y)

            #dataset = FDataGridDataset(X_standardized, y)
        else:
            dataset = from_FDataGrid_to_TensorDataset(X, y)
            #dataset = FDataGridDataset(X, y)
        return dataset

    def _loss_function(
            self,
            score_model: torch.nn.Module,
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
        z = torch.randn_like(x, generator=self.torch_generator)  # Shape (N, M)
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
            reverse_process: ReverseDiffusionProcess | None = None,
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
        check_is_fitted(self, attributes=["score_model_", "grid_points_"])

        if reverse_process is None:
            reverse_process = SDEReverseDiffusionProcess(EulerMaruyamaIntegrator(random_state=self.random_state))

        if y is not None:
            n_samples = len(y)
        # Sample from the final distribution of the diffusion process
        x_t = self.diff_process.sample_limit_distribution(
            n_samples,
            self.device,
            random_state=self.random_state,
        )
        # Reverse diffusion process to generate samples
        # No gradients needed during generation; this avoids storing
        # the computation graph and significantly reduces memory usage.
        with torch.no_grad():
            x_0 = reverse_process.reverse(
                diff_process=self.diff_process,
                score_model=self.score_model_,
                x_t=x_t,
                t_1=self.diff_process.T,
                y=y,
            )


        # Convert the generated samples to an FDataGrid object
        data_matrix = x_0.detach().cpu().numpy().reshape(n_samples, -1, 1)

        f_data = FDataGrid(
            data_matrix=data_matrix,
            grid_points=self.grid_points_,
        )

        if self.normalize or self.standardize:
            bias = getattr(self, "_bias_", 0.0)
            scale = getattr(self, "_scale_", 1.0)
            f_data = f_data * scale + bias
        return f_data

    def generate_evolution(
        self,
        n_samples: int,
        timesteps: NDArrayFloat,
        reverse_process: ReverseDiffusionProcess | None = None,
        y: NDArrayFloat | None = None,
    ) -> list[FDataGrid]:
        """Generates the evolutionary trajectory of functional data samples.

        Tracks and returns the state of the generated functional data at each 
        specified time step during the reverse diffusion process.

        If class labels are provided, the generated samples will be conditioned
        on those labels and n_samples will be ignored.

        If the model has not been fitted yet, a RuntimeError is raised.

        Args:
            timesteps: An array of timesteps (usually descending from T to 0)
                    at which to extract the intermediate data states.
            n_samples: The number of samples to generate.
            reverse_process: The reverse diffusion process to use for 
            generation. If `None`, defaults to `SDEReverseDiffusionProcess`
            with `EulerMaruyamaIntegrator` with 1000 steps.
            y: Optional tensor with the class labels of the data. Default is
            `None`.

        Returns:
            A list of FDataGrid objects representing the functional data
            at each timestep.
        """
        check_is_fitted(self, attributes=["score_model_", "grid_points_"])

        if reverse_process is None:
            reverse_process = SDEReverseDiffusionProcess(
                EulerMaruyamaIntegrator(random_state=self.random_state),
            )

        if y is not None:
            n_samples = len(y)

        # Sample from the final distribution of the diffusion process (noise)
        x_current = self.diff_process.sample_limit_distribution(
            n_samples,
            self.device,
            random_state=self.random_state
        )

        evolution_tensors = []

        # Reverse diffusion process piece-wise to capture intermediate samples
        with torch.no_grad():
            for i in range(len(timesteps) - 1):
                t_current = timesteps[i]
                t_next = timesteps[i + 1]

                # Store the state at the current timestep
                evolution_tensors.append(x_current.clone())

                x_current = reverse_process.reverse(
                    self.diff_process,
                    self.score_model_,
                    x_current,
                    t_current,
                    y=y,
                    t_0=t_next,
                )

            # Append the final state (usually at t=0)
            evolution_tensors.append(x_current.clone())


        # Pre-calculate bias and scale outside the loop to optimize performance
        bias = getattr(self, "_bias_", 0.0) if (self.normalize or self.standardize) else 0.0
        scale = getattr(self, "_scale_", 1.0) if (self.normalize or self.standardize) else 1.0

        f_data_evolution = []

        # Convert all tracked tensors into FDataGrid objects
        for x_step in evolution_tensors:
            data_matrix = x_step.detach().cpu().numpy().reshape(n_samples, -1, 1)

            f_data = FDataGrid(
                data_matrix=data_matrix,
                grid_points=self.grid_points_,
            )

            # Apply transformations if necessary
            if self.normalize or self.standardize:
                f_data = f_data * scale + bias
                
            f_data_evolution.append(f_data)

        return f_data_evolution

    def save_score_model(self, file_path: str | Path) -> None:
        """Persist score model and generation metadata using torch.save."""
        check_is_fitted(self, attributes=["score_model_", "grid_points_"])

        state_dict = {
            key: value.detach().cpu()
            for key, value in self.score_model_.state_dict().items()
        }

        checkpoint = {
            "score_model_state_dict": state_dict,
            "grid_points": np.asarray(self.grid_points_).tolist(),
            "M": int(len(self.grid_points_)),
            "normalize": self.normalize,
            "standardize": self.standardize,
            "bias": float(getattr(self, "_bias_", 0.0)),
            "scale": float(getattr(self, "_scale_", 1.0)),
            "diffusion_process_state_dict": self.diff_process.serialize_fit_data(),
        }
        torch.save(checkpoint, Path(file_path))

    def load_score_model(self, file_path: str | Path) -> "FDataGenerator":
        """Load a score-model state dict into the current model instance."""
        checkpoint = torch.load(
            Path(file_path),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format: expected a dictionary.")

        required_keys = {
            "score_model_state_dict",
            "grid_points",
            "M",
            "normalize",
            "standardize",
            "bias",
            "scale",
            "diffusion_process_state_dict",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required keys: "
                + ", ".join(sorted(missing_keys)),
            )

        if bool(checkpoint["normalize"]) != self.normalize:
            raise ValueError(
                "Checkpoint normalize flag does not match generator "
                "configuration.",
            )
        if bool(checkpoint["standardize"]) != self.standardize:
            raise ValueError(
                "Checkpoint standardize flag does not match generator "
                "configuration.",
            )
        process_state_dict = checkpoint["diffusion_process_state_dict"]
        if not isinstance(process_state_dict, dict):
            raise ValueError("Invalid diffusion_process_state_dict format: expected a dictionary.")
        try:
            self.diff_process.deserialize_fit_data(process_state_dict)
        except Exception as exc:
            raise ValueError(
                "Failed to load diffusion process state from checkpoint. "
                "Ensure the checkpoint was saved with a compatible diffusion process.",
            ) from exc
        del checkpoint["diffusion_process_state_dict"]
        
        if hasattr(self, "score_model_"):
            model = self.score_model_
        elif self.score_model is not None:
            if not isinstance(self.score_model, torch.nn.Module):
                raise TypeError("score_model must be a torch.nn.Module or None.")
            model = self.score_model.to(self.device)
        else:
            raise ValueError(
                "Cannot load score_model state_dict without a score_model "
                "instance. Pass score_model in __init__ or fit before loading.",
            )
        
        
        try:
            model.load_state_dict(
                checkpoint["score_model_state_dict"],
                strict=True,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint state_dict is incompatible with the current "
                "score_model architecture. Use the same architecture used "
                "when saving.",
            ) from exc

        self.score_model_ = model
        self.score_model_.eval()

        self.grid_points_ = np.asarray(checkpoint["grid_points"])
        self.grid_points = self.grid_points_
        self._bias_ = float(checkpoint["bias"])
        self._scale_ = float(checkpoint["scale"])
        self.score_model = self.score_model_
        return self

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

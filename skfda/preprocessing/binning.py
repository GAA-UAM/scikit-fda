"""Data Binning Module with DataBinner class."""

import itertools
from typing import Optional, Tuple, Union

import numpy as np

from skfda._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from skfda.representation import FDataGrid, FDataIrregular


class DataBinner(
    # class DomainBinner(
    # class Binner(
    BaseEstimator,
    TransformerMixin[FDataGrid, FDataGrid, object],
):
    r"""
    FData Binner.

    Class to group the grid points of a FDataGrid into bins. The values
    of the new grid points are computed based on the method specified in
    the bin_aggregation parameter.

    It follows the scikit-learn methodology of TransformerMixin, so it
    works with the fit and transform methods.

    Note: if a value falls in the limit of two bins, it will be included in
    the bin on the right.

    Parameters:
        bins: Number of bins if integer for 1-dimensional case or n-tuple
            of integers for n-dimensional case, and numpy array of bin edges
            if 1-dimensional case or tuple of numpy arrays of bin edges for
            n-dimensional case.
        range: Tuple with the minimum and maximum values of the domain range
            of the output FDataGrid. Ignored if given bin edges. If None, the
            domain range of the FDataGrid is used.
        output_grid: Method to select the grid points of the output FDataGrid.
            The validity of this parameter is not ensured untl the input
            FDataGrid is fitted. The available methods are: 'left', 'middle',
            'right' or a tuple of numpy arrays with the grid points for each
            dimension, which must fit within the output bins.
        bin_aggregation: Method to compute the value of the bin. The available
            methods are: 'mean', 'median'.

    Attributes:
        dim: Dimension of the FDataGrid the binner can process.
        bin_edges: Array with the specific bin edges. Can value to None until
            the FDataGrid is fitted.
        n_bins: Number of bins. Can value to None until the FDataGrid is
            fitted.
        min_domain: List with the minimum value of the domain range of the
            output FDataGrid for each dimension. Defined after fitting.
        max_domain: List with the maximum value of the domain range of the
            output FDataGrid for each dimension. Defined after fitting.
        bin_representative: Mode to compute the value in domain of each bin.
            Can be None if output_grid has been specified as array.
        output_grid: Value of the points in the output grid of the FDataGrid.
            Can be None if output_grid has been specified as string.
        bin_aggregation: Method to compute the value of the bin.
        is_irregular: Structure of the fitted data. If True, the data is
            irregular. If False, the data is regular. Defined after fitting.
        bins: User-defined n_bins or bin_edges, used for scikit-learn
            compatibility.

    --------------------------------------------------------------------------
    To complete

    Examples:
        Given a FDataGrid with 2 samples representing a function
        :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> import numpy as np
        >>> from skfda.representation import FDataGrid
        >>> from skfda.preprocessing import DataBinner
        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, grid_points,
        >>>                coordinate_names=["C1", "C2"])
        >>> binner = DataBinner(bins=1)
        >>> binned_fd = binner.fit_transform(fd)

        Given a FDataGrid with 2 samples representing a function
        :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> import numpy as np
        >>> from skfda.representation import FDataGrid
        >>> from skfda.preprocessing import DataBinner
        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [[2, 4], [3,6]]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> binner = DataBinner(bins=(1,2), bin_aggregation="median")
        >>> binned_fd = binner.fit_transform(fd)
    """

    def __init__(
        self,
        *,
        bins: Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]],
        range: Optional[
            Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]
        ] = None,
        output_grid: Union[str, np.ndarray, Tuple[np.ndarray, ...]] = "middle",
        bin_aggregation: str = "mean",
    ):
        # Used for scikit-learn compatibility
        self.bins = bins
        bins_result = self._process_bins(bins)
        self.dim = bins_result[0]
        self.bin_edges = bins_result[1]
        self.n_bins = bins_result[2]

        self._validate_range(range, self.dim)

        grid_result = self._process_output_grid(output_grid)
        self.bin_representative = grid_result[0]
        self.output_grid = grid_result[1]

        if bin_aggregation not in {"mean", "median"}:
            raise ValueError(
                "Bin aggregation must be one of 'mean' or 'median'.",
            )

        self.range = range
        self.bin_aggregation = bin_aggregation

    def _process_bins(self, bins):
        """
        Validate and process the bins parameter.

        Validate and process the bins parameter based on the dimension of the
        data.

        Args:
            bins: Bins parameter to be validated.

        Returns:
            Tuple with the dimension of the data, the bin edges and the number
            of bins.

        Raises:
            ValueError: If the bins parameter is invalid.
        """
        if isinstance(bins, int):
            if bins < 1:
                raise ValueError(
                    "Number of bins must be greater than 0 in every "
                    "dimension of the domain.",
                )
            return 1, None, [bins]

        if isinstance(bins, tuple):
            if all(isinstance(b, int) for b in bins):
                if not all(b > 0 for b in bins):
                    raise ValueError(
                        "Number of bins must be greater than 0 in every "
                        "dimension of the domain.",
                    )
                return len(bins), None, bins

            if all(isinstance(b, np.ndarray) for b in bins):
                if not all(self._check_bin_edges(b) for b in bins):
                    raise ValueError(
                        "If bins represent bin edges, each array must be a "
                        "1-dimensional array with at least two elements, "
                        "strictly increasing and without any NaN values.",
                    )
                return len(bins), bins, None

        if isinstance(bins, np.ndarray):
            if not self._check_bin_edges(bins):
                raise ValueError(
                    "If bins represent bin edges, the array must be a "
                    "1-dimensional array with at least two elements, "
                    "strictly increasing and without any NaN values.",
                )
            return 1, bins, None

        raise ValueError(
            "Bins must be an int or a numpy array for one-dimensional domain "
            "data, or a tuple of ints or numpy arrays for n-dimensional data.",
        )

    def _check_bin_edges(self, bin_edges: np.ndarray) -> bool:
        """
        Check if bin edges are valid.

        Check if bin edges are a 1-dimensional array with at least two
        elements, strictly increasing and without any NaN values.

        Args:
            bin_edges: Array with the specific bin edges (for one dimension).

        Returns:
            True if bin_edges are valid, False otherwise.
        """
        bin_edges = np.asarray(bin_edges, dtype=float)
        if bin_edges.ndim != 1 or len(bin_edges) < 2:
            return False

        return not (
            np.any(np.isnan(bin_edges)) or not np.all(np.diff(bin_edges) > 0)
        )

    def _validate_range(self, range_param, dim):
        """
        Validate the range parameter.

        Validate the range parameter based on the dimension of the domain of
        the data.

        Args:
            range_param: Range parameter to be validated.
            dim: Dimension of the data.

        Raises:
            ValueError: If the range parameter is invalid.
        """
        if range_param is None:
            return

        if not isinstance(range_param, tuple):
            raise ValueError("Range must be a tuple.")

        if dim == 1:
            if (
                len(range_param) == 2
                and all(isinstance(x, (float, int)) for x in range_param)
                and range_param[0] < range_param[1]
            ):
                return

            raise ValueError(
                "For 1-dimensional domain, range must be a tuple of two "
                "numbers with the first being smaller than the second.",
            )

        if len(range_param) == dim and all(
            isinstance(x, tuple)
            and len(x) == 2
            and all(isinstance(y, (float, int)) for y in x)
            and x[0] < x[1]
            for x in range_param
        ):
            return

        raise ValueError(
            f"For {dim}-dimensional domain, range must be a tuple with {dim} "
            "tuples, each containing two numbers where the first is smaller "
            "than the second.",
        )

    def _process_output_grid(self, output_grid):
        """
        Validate and process the output grid parameter.

        Validate and process the output grid parameter based on the dimension
        of the data.

        Args:
            output_grid: Output grid parameter to be validated.

        Returns:
            Tuple with the bin representative mode and the output grid.

        Raises:
            ValueError: If the output grid parameter is invalid.
        """
        if isinstance(output_grid, str) and output_grid in {
            "left",
            "middle",
            "right",
        }:
            return output_grid, None

        if isinstance(output_grid, np.ndarray):
            if self.dim != 1:
                raise ValueError(
                    f"Output grid must be 'left', 'middle', 'right' or a "
                    f"{self.dim} tuple of numpy arrays for {self.dim}"
                    f"-dimensional domain.",
                )
            if not np.all(np.diff(output_grid) > 0):
                raise ValueError(
                    "Output grid values must be strictly increasing.",
                )
            expected_length = (
                self.n_bins[0]
                if self.n_bins is not None
                else len(self.bin_edges) - 1
            )
            if len(output_grid) != expected_length:
                raise ValueError(
                    f"Output grid length ({len(output_grid)}) does not match "
                    f"expected length ({expected_length}). Ensure it matches "
                    "the expected number of bins.",
                )
            return None, output_grid

        if isinstance(output_grid, tuple) and all(
            isinstance(arr, np.ndarray) for arr in output_grid
        ):
            if len(output_grid) != self.dim:
                raise ValueError(
                    f"Output grid must be 'left', 'middle', 'right' or a "
                    f"{self.dim} tuple of numpy arrays for {self.dim}"
                    f"-dimensional domain.",
                )
            if not all(np.all(np.diff(arr) > 0) for arr in output_grid):
                raise ValueError(
                    "Each output grid must be strictly increasing.",
                )
            for i in range(self.dim):
                expected_length = (
                    self.n_bins[i]
                    if self.n_bins is not None
                    else len(self.bin_edges[i]) - 1
                )
                if len(output_grid[i]) != expected_length:
                    raise ValueError(
                        f"Output grid at dimension {i} has length "
                        f"{len(output_grid[i])}, but expected "
                        f"{expected_length} based on the number of bins.",
                    )
            return None, output_grid

        raise ValueError(
            f"Output grid must be 'left', 'middle', 'right' or a {self.dim} "
            f"tuple of numpy arrays for {self.dim}-dimensional domains.",
        )

    def fit(
        self,
        X: Union[FDataGrid, FDataIrregular],
        y: object = None,
    ) -> "DataBinner":
        """
        Prepare the binning parameters based on the domain range.

        Args:
            X: FDataGrid or FDataIrregular to be binned.
            y: Ignored.

        Returns:
            self
        """
        self.is_irregular = isinstance(X, FDataIrregular)

        if X.dim_domain != self.dim:
            raise ValueError(
                f"Input FData must have {self.dim} domain dimensions.",
            )

        self._compute_domain_range(X)

        if self.n_bins is not None:
            if self.dim == 1:
                self.bin_edges = np.linspace(
                    self.min_domain[0],
                    self.max_domain[0],
                    self.n_bins[0] + 1,
                )
            else:
                self.bin_edges = []
                for i in range(self.dim):
                    self.bin_edges.append(
                        np.linspace(
                            self.min_domain[i],
                            self.max_domain[i],
                            self.n_bins[i] + 1,
                        ),
                    )
        else:
            self.n_bins = []
            if self.dim == 1:
                self.n_bins.append(len(self.bin_edges) - 1)
            else:
                for j in range(self.dim):
                    self.n_bins.append(len(self.bin_edges[j]) - 1)

        self._compute_output_grid()

        return self

    def _compute_domain_range(self, X: FDataGrid):
        """
        Compute min and max domain values based on bin edges or range.

        Args:
            X: FDataGrid to be binned.
        """
        self.min_domain = []
        self.max_domain = []

        if self.bin_edges is not None:
            for edges in self.bin_edges if self.dim > 1 else [self.bin_edges]:
                self.min_domain.append(float(edges[0]))
                self.max_domain.append(float(edges[-1]))

        elif self.range is None:
            for domain in X.domain_range:
                self.min_domain.append(domain[0])
                self.max_domain.append(domain[1])

        else:
            for range_value in self.range if self.dim > 1 else [self.range]:
                self.min_domain.append(range_value[0])
                self.max_domain.append(range_value[1])

    def _compute_output_grid(self):
        """Prepare the output grid based on the domain range."""
        if self.output_grid is None:
            self.output_grid = []

            for i in range(self.dim):
                bin_edges_i = (
                    self.bin_edges[i] if self.dim > 1 else self.bin_edges
                )

                if self.bin_representative == "left":
                    grid = bin_edges_i[:-1]
                elif self.bin_representative == "middle":
                    grid = (bin_edges_i[:-1] + bin_edges_i[1:]) / 2
                elif self.bin_representative == "right":
                    grid = bin_edges_i[1:]

                self.output_grid.append(grid)

            if self.dim == 1:
                self.output_grid = self.output_grid[0]

        if isinstance(self.output_grid, np.ndarray):
            for index, point in enumerate(self.output_grid):
                if ~(
                    self.bin_edges[index] <= point <= self.bin_edges[index + 1]
                ):
                    raise ValueError(
                        f"Output grid point {self.output_grid[index]} is "
                        f"outside its bin range [{self.bin_edges[index]}, "
                        f"{self.bin_edges[index + 1]}].",
                    )
        elif isinstance(self.output_grid, tuple):
            for dim_index, (grid_values, edges) in enumerate(
                zip(self.output_grid, self.bin_edges),
            ):
                is_within_bounds = (edges[:-1] <= grid_values) & (
                    grid_values <= edges[1:]
                )
                if not np.all(is_within_bounds):
                    raise ValueError(
                        f"Some output grid points in dimension {dim_index} "
                        "are outside their bin ranges. Ensure all values lie "
                        f"within [{edges[0]}, {edges[-1]}] and their intended "
                        "bin.",
                    )

    def transform(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> FDataGrid:
        """
        Group the FDataGrid points into bins.

        Args:
            X: FDataGrid to be binned.
            y: Ignored.

        Returns:
            Binned FDataGrid.
        """
        if self.is_irregular:
            if isinstance(X, FDataGrid):
                raise ValueError(
                    "Binner fitted with irregular data must receive an "
                    "FDataIrregular instance.",
                )
            return self._transform_irregular(X)

        if isinstance(X, FDataIrregular):
            raise ValueError(
                "Binner fitted with regular data must receive an FDataGrid "
                "instance.",
            )

        grid_points = X.grid_points
        data_matrix = X.data_matrix

        binned_values = []

        if self.dim == 1:
            binned_values = self._compute_univariate_domain_binning(
                grid_points[0],
                data_matrix,
                self.bin_edges,
            )
            binned_values = np.array(binned_values)
            self.output_grid = [np.array(self.output_grid)]

        else:
            binned_values = self._compute_multivariate_domain_binning(
                grid_points,
                data_matrix,
                self.bin_edges,
            )

        return X.copy(
            data_matrix=binned_values,
            grid_points=self.output_grid,
            domain_range=tuple(zip(self.min_domain, self.max_domain)),
            dataset_name=X.dataset_name,
            argument_names=X.argument_names,
            coordinate_names=X.coordinate_names,
            extrapolation=X.extrapolation,
            interpolation=X.interpolation,
        )

    def _transform_irregular(
        self,
        X: FDataIrregular,
    ) -> FDataGrid:
        """
        Group the FDataIrregular points into bins.

        Args:
            X: FDataIrregular to be binned.

        Returns:
            Binned FDataGrid.
        """
        n_samples = len(X.start_indices)
        dim_domain = X.dim_domain
        dim_codomain = X.dim_codomain

        if self.dim == 1:
            self.bin_edges = (self.bin_edges,)
            self.output_grid = (self.output_grid,)

        grid_shape = tuple(len(g) for g in self.output_grid)
        binned_values = np.full((n_samples, *grid_shape, dim_codomain), np.nan)

        for sample_idx in range(n_samples):
            start = X.start_indices[sample_idx]
            end = (
                X.start_indices[sample_idx + 1]
                if sample_idx + 1 < len(X.start_indices)
                else len(X.points)
            )

            arg_array = X.points[start:end]
            val_array = X.values[start:end]

            if val_array.ndim == 1:
                val_array = val_array[:, np.newaxis]

            bin_indices = np.empty((arg_array.shape[0], dim_domain), dtype=int)
            for d in range(dim_domain):
                bin_indices[:, d] = (
                    np.digitize(
                        arg_array[:, d], self.bin_edges[d], right=False
                    )
                    - 1
                )

                out_of_range = (arg_array[:, d] < self.bin_edges[d][0]) | (
                    arg_array[:, d] > self.bin_edges[d][-1]
                )
                bin_indices[:, d][out_of_range] = -1

                last_bin_index = len(self.bin_edges[d]) - 2
                bin_indices[:, d][
                    arg_array[:, d] == self.bin_edges[d][-1],
                ] = last_bin_index

            for bin_idx in np.ndindex(*grid_shape):
                mask = np.logical_and.reduce(
                    [
                        bin_indices[:, d] == bin_idx[d]
                        for d in range(dim_domain)
                    ],
                )
                points_in_bin = val_array[mask]

                if points_in_bin.shape[0] > 0:
                    if self.bin_aggregation == "mean":
                        binned_values[sample_idx][bin_idx] = np.nanmean(
                            points_in_bin, axis=0,
                        )
                    else:
                        binned_values[sample_idx][bin_idx] = np.nanmedian(
                            points_in_bin, axis=0,
                        )

        grid_points = (
            self.output_grid if dim_domain > 1 else [self.output_grid[0]]
        )

        return FDataGrid(
            data_matrix=binned_values,
            grid_points=grid_points,
            domain_range=X.domain_range,
            dataset_name=X.dataset_name,
            argument_names=X.argument_names,
            coordinate_names=X.coordinate_names,
            extrapolation=X.extrapolation,
            interpolation=X.interpolation,
        )

    def _compute_univariate_domain_binning(
        self,
        grid_points: np.ndarray,
        data_matrix: np.ndarray,
        bin_edges: np.ndarray,
    ):
        """
        Compute bin values for univariate data.

        Args:
            grid_points: Grid points of the FDataGrid.
            data_matrix: Data matrix of the FDataGrid.
            bin_edges: Bin edges to be used for binning.

        Returns:
            Tuple with the counts of points in each bin and the computed bin
            values.
        """
        n_samples, _, n_codomain = data_matrix.shape
        binned_values = np.full(
            (n_samples, len(bin_edges) - 1, n_codomain),
            np.nan,
        )

        for i in range(len(bin_edges) - 1):
            points_in_bin = (grid_points >= bin_edges[i]) & (
                grid_points < bin_edges[i + 1]
            )
            if i == len(bin_edges) - 2:
                # Include right endpoint in the last bin
                points_in_bin |= grid_points == bin_edges[i + 1]

            if np.any(points_in_bin):
                bin_values = np.full((n_samples, n_codomain), np.nan)

                for codomain_dim in range(n_codomain):
                    bin_values[:, codomain_dim] = (
                        self._compute_univariate_bin_values(
                            data_matrix[:, :, codomain_dim],
                            points_in_bin,
                        )
                    )

                binned_values[:, i, :] = bin_values

        return binned_values

    def _compute_univariate_bin_values(
        self,
        data_matrix: np.ndarray,
        points_in_bin: np.ndarray,
    ) -> np.ndarray:
        """
        Compute individual bin value.

        Compute individual bin value based on the specified bin_aggregation
        for the univariate case.

        Args:
            data_matrix: The data matrix from the FDataGrid.
            points_in_bin: Boolean array indicating points in the current bin.

        Returns:
            Array of computed bin values for the current bin.
        """
        n_samples = data_matrix.shape[0]
        bin_values = np.full(n_samples, np.nan)

        for row_idx in range(n_samples):
            row_data = data_matrix[row_idx, points_in_bin]

            if not np.all(np.isnan(row_data)):
                if self.bin_aggregation == "mean":
                    bin_values[row_idx] = np.nanmean(row_data)
                else:
                    bin_values[row_idx] = np.nanmedian(row_data)

        return bin_values

    def _compute_multivariate_domain_binning(
        self,
        grid_points: np.ndarray,
        data_matrix: np.ndarray,
        bin_edges: np.ndarray,
    ):
        """
        Compute bin values for multivariate data.

        Args:
            grid_points: Grid points of the FDataGrid.
            data_matrix: Data matrix of the FDataGrid.
            bin_edges: Bin edges to be used for binning.

        Returns:
            Tuple with the counts of points in each bin and the computed bin
            values.
        """
        n_samples, *grid_shape, n_codomain = data_matrix.shape
        points_in_bin = []

        for i in range(self.dim):
            bin_edges_i = bin_edges[i]
            points_in_bin.append([])
            for j in range(len(bin_edges_i) - 1):
                mask = (grid_points[i] >= bin_edges_i[j]) & (
                    grid_points[i] < bin_edges_i[j + 1]
                )
                if j == len(bin_edges_i) - 2:
                    mask |= grid_points[i] == bin_edges_i[j + 1]
                points_in_bin[i].append(mask)

        points_in_bin_combinations = list(itertools.product(*points_in_bin))

        output_values = np.full((n_samples, *self.n_bins, n_codomain), np.nan)

        for k, combination in enumerate(points_in_bin_combinations):
            combination_data = data_matrix

            for dim, comb_mask in enumerate(combination, start=1):
                index_tuple = (slice(None),) * dim + (comb_mask,)
                combination_data = combination_data[index_tuple]

            if np.any(~np.isnan(combination_data)):
                combination_output_values = (
                    self._compute_multivariate_bin_values(combination_data)
                )

                multi_dim_index = np.unravel_index(k, self.n_bins)
                for obs_idx, value in enumerate(combination_output_values):
                    index_tuple = (obs_idx,) + multi_dim_index + (slice(None),)
                    output_values[index_tuple] = value

        return output_values

    def _compute_multivariate_bin_values(
        self,
        data_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute individual bin value.

        Compute individual bin value based on the specified bin_aggregation
        for the multivariate case.

        Args:
            data_matrix: The data matrix with the elements in the bin.

        Returns:
            Array of computed bin values for the current bin.
        """
        n_samples, *bin_shape, n_codomain = data_matrix.shape
        output_values = np.full((n_samples, n_codomain), np.nan)

        for codomain_dim in range(n_codomain):
            obs_data = []

            for idx in np.ndindex(*bin_shape):
                obs_data.append(
                    data_matrix[(slice(None),) + idx + (codomain_dim,)],
                )

            obs_data = np.array(obs_data).T

            for obs in range(n_samples):
                if not np.all(np.isnan(obs_data[obs])):
                    if self.bin_aggregation == "mean":
                        output_values[obs, codomain_dim] = np.nanmean(
                            obs_data[obs],
                        )
                    else:
                        output_values[obs, codomain_dim] = np.nanmedian(
                            obs_data[obs],
                        )

        return output_values

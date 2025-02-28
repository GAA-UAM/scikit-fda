"""Data Binning Module with CommonGridBinner class."""

import itertools
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from skfda._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from skfda.representation import FDataGrid, FDataIrregular

BinsTypeDim1 = Union[int, NDArray[np.float64]]
BinsType = Union[BinsTypeDim1, Tuple[BinsTypeDim1, ...]]
NBinsNDim = Union[Tuple[int, ...], List[int]]
ArrayNDImType = Union[
    NDArray[np.float64],
    Tuple[NDArray[np.float64], ...],
    List[NDArray[np.float64]],
]
RangeType = Optional[
    Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]
]
ProcessBins = Tuple[
    int,
    Optional[ArrayNDImType],
    Optional[Tuple[int, ...]],
]
OutputGridType = Union[
    str,
    NDArray[np.float64],
    Tuple[NDArray[np.float64], ...],
]
ProcessOutputGrid = Tuple[
    Optional[str],
    Optional[ArrayNDImType],
]
TupleOutputGrid = Tuple[None, Tuple[NDArray[np.float64], ...]]


class CommonGridBinner(  # noqa: WPS214, WPS230
    # class DomainBinner(
    # class Binner(
    # class FunctionalDataBinningTransformer(
    # class CommonGridTransformer(
    # class DataBinner(
    # class Discretizer(
    # class FunctionalDataDiscretizer(
    # class CommonGridDiscretizer(
    BaseEstimator,
    TransformerMixin[FDataGrid, FDataGrid, object],
):
    r"""
    Functional Data Binner based on the domain.

    Class to group the grid points of a FDataGrid into bins. The values
    of the new grid points are computed based on the method specified in
    the bin_aggregation parameter.

    It follows the scikit-learn methodology of TransformerMixin, so it
    works with the fit and transform methods.

    Note: if a value falls in the limit of two bins, it will be included in
    the bin on the right.

    Parameters:
    -----------
        bins: Number of bins if integer for 1-dimensional case or n-tuple
            of integers for n-dimensional case, and numpy array of bin edges
            if 1-dimensional case or tuple of numpy arrays of bin edges for
            n-dimensional case.
        domain_range: Tuple with the minimum and maximum values of the domain
            range of the output FDataGrid. Ignored if given bin edges. If None,
            the domain range of the FDataGrid is used.
        output_grid: Method to select the grid points of the output FDataGrid.
            The validity of this parameter is not ensured untl the input
            FDataGrid is fitted. The available methods are: 'left', 'middle',
            'right' or a tuple of numpy arrays with the grid points for each
            dimension, which must fit within the output bins.
        bin_aggregation: Method to compute the value of the bin. The available
            methods are: 'mean', 'median'.

    Attributes:
    -----------
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
        domain_range: User-defined domain_range, used for scikit-learn
            compatibility.

    --------------------------------------------------------------------------

    Examples:
    ---------
        Given an FDataIrregular with 2 samples representing a function
        :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> from skfda.preprocessing.binning import CommonGridBinner
        >>> from skfda.representation import FDataIrregular
        >>> indices=[0,2]
        >>> points=[[1.0],[4.0],[1.0],[3.0],[5.0]]
        >>> values=[
        ...     [1.0, 1.0],
        ...     [2.0, 2.0],
        ...     [3.0, 3.0],
        ...     [4.0, 4.0],
        ...     [5.0, 5.0],
        ... ]
        >>> fd = FDataIrregular(
        ...     start_indices=indices,
        ...     points=points,
        ...     values=values,
        ... )
        >>> binner = CommonGridBinner(bins=2)
        >>> binner.fit_transform(fd)

        Given a FDataGrid with 2 samples representing a function
        :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> import numpy as np
        >>> from skfda.preprocessing.binning import CommonGridBinner
        >>> from skfda.representation import FDataGrid
        >>> grid_points=[
        ...     [1., 2.],
        ...     [1., 2.],
        ... ]
        >>> values=np.array(
        ...     [
        ...         [
        ...             [[1.0], [2.]],
        ...             [[1.4], [1.8]],
        ...         ],
        ...         [
        ...             [[2.0], [2.1]],
        ...             [[0.5], [1.]],
        ...         ],
        ...     ]
        ... )
        >>> fd = FDataGrid(
        ...     grid_points=grid_points,
        ...     data_matrix=values
        ... )
        >>> binner = CommonGridBinner(bins=(2, 1))
        >>> binner.fit_transform(fd)
    """

    def __init__(
        self,
        *,
        bins: BinsType,
        domain_range: RangeType = None,
        output_grid: OutputGridType = "middle",
        bin_aggregation: str = "mean",
    ):
        # Used for scikit-learn compatibility
        self.bins = bins
        bins_result = self._process_bins(bins)
        self.dim = bins_result[0]
        self.bin_edges: Optional[ArrayNDImType] = bins_result[1]
        self.n_bins: Optional[NBinsNDim] = bins_result[2]

        self._validate_range(domain_range, self.dim)

        grid_result = self._process_output_grid(output_grid)
        self.bin_representative = grid_result[0]
        self.output_grid = grid_result[1]

        if bin_aggregation not in {"mean", "median"}:
            raise ValueError(
                "Bin aggregation must be one of 'mean' or 'median'.",
            )

        self.domain_range: RangeType = domain_range
        self.bin_aggregation = bin_aggregation

    def _process_bins(self, bins: BinsType) -> ProcessBins:
        """
        Validate and process the bins parameter.

        Args:
            bins: Bins parameter to be validated.

        Returns:
            Tuple with the dimension of the data, the bin edges, and the number
            of bins.

        Raises:
            ValueError: If the bins parameter is invalid.
        """
        if isinstance(bins, int):
            return self._process_int_bins(bins)

        if isinstance(bins, tuple):
            return self._process_tuple_bins(bins)

        if isinstance(bins, np.ndarray):
            return self._process_ndarray_bins(bins)

        raise ValueError(
            "Bins must be an int or a numpy array for one-dimensional domain "
            "data, or a tuple of ints or numpy arrays for n-dimensional data.",
        )

    def _process_int_bins(self, bins: int) -> ProcessBins:
        """Process bins when it's an integer."""
        if bins < 1:
            raise ValueError(
                "Number of bins must be greater than 0 in every dimension of "
                "the domain.",
            )
        return 1, None, (bins,)

    def _process_tuple_bins(
        self,
        bins: tuple[BinsTypeDim1, ...],
    ) -> ProcessBins:
        """Process bins when it's a tuple."""
        if all(isinstance(b, int) for b in bins):
            if not all(b > 0 for b in bins):
                raise ValueError(
                    "Number of bins must be greater than 0 in every dimension "
                    "of the domain.",
                )
            return (
                len(bins),
                None,
                tuple(b for b in bins if isinstance(b, int)) if bins else None,
            )

        if all(isinstance(b, np.ndarray) for b in bins):
            if not all(
                self._check_bin_edges(b)
                for b in bins
                if isinstance(b, np.ndarray)
            ):
                raise ValueError(
                    "If bins represent bin edges, each array must be a "
                    "1-dimensional array with at least two elements, "
                    "strictly increasing and without any NaN values.",
                )
            return (
                len(bins),
                (
                    tuple(b for b in bins if isinstance(b, np.ndarray))
                    if bins
                    else None
                ),
                None,
            )

        raise ValueError(
            "If bins is a tuple, it must contain either integers or numpy "
            "arrays.",
        )

    def _process_ndarray_bins(self, bins: NDArray[np.float64]) -> ProcessBins:
        """Process bins when it's a numpy ndarray."""
        if not self._check_bin_edges(bins):
            raise ValueError(
                "If bins represent bin edges, the array must be a "
                "1-dimensional array with at least two elements, "
                "strictly increasing and without any NaN values.",
            )
        return 1, bins, None

    def _check_bin_edges(self, bin_edges: NDArray[np.float64]) -> bool:
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

    def _validate_range(self, range_param: RangeType, dim: int) -> None:
        """
        Validate the domain_range parameter.

        Validate the domain_range parameter based on the dimension of the
        domain of the data.

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
                and isinstance(range_param[0], (float, int))
                and isinstance(range_param[1], (float, int))
                and float(range_param[0]) < float(range_param[1])
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

    def _process_output_grid(
        self,
        output_grid: OutputGridType,
    ) -> ProcessOutputGrid:
        """
        Validate and process the output grid parameter.

        Args:
            output_grid: Output grid parameter to be validated.

        Returns:
            Tuple with the bin representative mode and the output grid.

        Raises:
            ValueError: If the output grid parameter is invalid.
        """
        if isinstance(output_grid, str):
            return self._process_string_output_grid(output_grid)

        if isinstance(output_grid, np.ndarray):
            return self._process_ndarray_output_grid(output_grid)

        if isinstance(output_grid, tuple) and all(
            isinstance(arr, np.ndarray) for arr in output_grid
        ):
            return self._process_tuple_output_grid(output_grid)

        raise ValueError(
            f"Output grid must be 'left', 'middle', 'right' or a {self.dim} "
            f"tuple of numpy arrays for {self.dim}-dimensional domains.",
        )

    def _process_string_output_grid(
        self,
        output_grid: str,
    ) -> Tuple[str, None]:
        """Process string output grid."""
        if output_grid in {"left", "middle", "right"}:
            return output_grid, None
        raise ValueError(
            "Invalid output grid string. Must be 'left', 'middle', or "
            "'right'.",
        )

    def _process_ndarray_output_grid(
        self,
        output_grid: NDArray[np.float64],
    ) -> Tuple[None, NDArray[np.float64]]:
        """Process numpy ndarray output grid."""
        if self.dim != 1:
            raise ValueError(
                f"Output grid must be 'left', 'middle', 'right' or a "
                f"{self.dim} tuple of numpy arrays for {self.dim}-dimensional "
                f"domain.",
            )
        if not np.all(np.diff(output_grid) > 0):
            raise ValueError("Output grid values must be strictly increasing.")

        if self.bin_edges is not None:
            if self.n_bins is None:
                expected_length = len(self.bin_edges) - 1
            else:
                expected_length = self.n_bins[0]
        else:
            expected_length = 0

        if len(output_grid) != expected_length:
            raise ValueError(
                f"Output grid length ({len(output_grid)}) does not match "
                f"expected length ({expected_length}). Ensure it matches "
                "the expected number of bins.",
            )
        return None, output_grid

    def _process_tuple_output_grid(
        self,
        output_grid: Tuple[NDArray[np.float64], ...],
    ) -> TupleOutputGrid:
        """Process tuple output grid."""
        if len(output_grid) != self.dim:
            raise ValueError(
                f"Output grid must be 'left', 'middle', 'right' or a "
                f"{self.dim} tuple of numpy arrays for {self.dim}-dimensional "
                f"domain.",
            )
        for i, arr in enumerate(output_grid):
            if not np.all(np.diff(arr) > 0):
                raise ValueError(
                    "Each output grid must be strictly increasing.",
                )

            if self.bin_edges is not None and self.n_bins is None:
                expected_length = len(self.bin_edges[i]) - 1
            elif self.n_bins is not None:
                expected_length = self.n_bins[i]
            else:
                expected_length = 0

            if len(arr) != expected_length:
                raise ValueError(
                    f"Output grid at dimension {i} has length "
                    f"{len(arr)}, but expected {expected_length} based on the "
                    f"number of bins.",
                )
        return None, output_grid

    def fit(
        self,
        X: Union[FDataGrid, FDataIrregular],
        y: object = None,
    ) -> "CommonGridBinner":
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

        if self.n_bins is None:
            self._initialize_n_bins()

        else:
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

        self._compute_output_grid()

        return self

    def _compute_domain_range(self, X: FDataGrid) -> None:
        """Compute min and max domain values based on bin edges or range.

        Args:
            X: FDataGrid to be binned.
        """
        self.min_domain: List[float] = []
        self.max_domain: List[float] = []

        if self.bin_edges is None:
            self._process_domain_range(X)
        else:
            self._process_bin_edges()

    def _process_domain_range(self, X: FDataGrid) -> None:
        """Process domain range when bin edges are None."""
        if self.domain_range is None:
            for domain in X.domain_range:
                self.min_domain.append(domain[0])
                self.max_domain.append(domain[1])
        else:
            self._extract_min_max_from_range()

    def _extract_min_max_from_range(self) -> None:
        """Extract min and max values from the domain range."""
        if self.domain_range is not None:
            for range_value in (
                self.domain_range if self.dim > 1 else [self.domain_range]
            ):
                if (
                    isinstance(range_value, tuple)
                    and isinstance(range_value[0], (int, float))
                    and isinstance(range_value[1], (int, float))
                ):
                    self.min_domain.append(range_value[0])
                    self.max_domain.append(range_value[1])

    def _process_bin_edges(self) -> None:
        """Process bin edges to determine min and max domain values."""
        if isinstance(self.bin_edges, np.ndarray):
            self.min_domain.append(float(self.bin_edges[0]))
            self.max_domain.append(float(self.bin_edges[-1]))
        elif isinstance(self.bin_edges, (list, tuple)):
            for edges in self.bin_edges:
                if isinstance(edges, np.ndarray):
                    self.min_domain.append(float(edges[0]))
                    self.max_domain.append(float(edges[-1]))

    def _initialize_n_bins(self) -> None:
        """Initialize self.n_bins to avoid deep nesting in fit()."""
        self.n_bins = []

        if self.dim == 1:
            if isinstance(self.bin_edges, np.ndarray):
                self.n_bins.append(len(self.bin_edges) - 1)
        else:
            self._process_multidimensional_bins()

    def _process_multidimensional_bins(self) -> None:
        """Process multidimensional bins to avoid deep nesting."""
        for j in range(self.dim):
            if (
                isinstance(self.bin_edges, tuple)
                and isinstance(self.bin_edges[j], np.ndarray)
                and isinstance(self.n_bins, list)
            ):
                self.n_bins.append(len(self.bin_edges[j]) - 1)

    def _compute_output_grid(self) -> None:
        """Prepare the output grid based on the domain range."""
        if self.output_grid is None:
            self.output_grid = []

            for i in range(self.dim):
                bin_edges_i = (
                    self.bin_edges[i]
                    if self.dim > 1
                    and isinstance(self.bin_edges, (tuple, list))
                    else self.bin_edges
                )
                if isinstance(bin_edges_i, np.ndarray):
                    grid = self._compute_grid(bin_edges_i)
                    self.output_grid.append(grid)

            if self.dim == 1:
                self.output_grid = self.output_grid[0]

        self._validate_output_grid()

    def _compute_grid(
        self,
        bin_edges_i: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the grid based on bin edges and representative type."""
        if self.bin_representative == "left":
            return bin_edges_i[:-1]
        elif self.bin_representative == "middle":
            return (bin_edges_i[:-1] + bin_edges_i[1:]) / 2
        return bin_edges_i[1:]

    def _validate_output_grid(self) -> None:
        """Validate the output grid to ensure points are within bin ranges."""
        if isinstance(self.output_grid, np.ndarray) and isinstance(
            self.bin_edges,
            np.ndarray,
        ):
            for index, point in enumerate(
                self.output_grid[:-1],
            ):  # Exclude the last point
                if (
                    point < self.bin_edges[index]
                    or point > self.bin_edges[index + 1]
                ):
                    raise ValueError(
                        f"Output grid point {point} is "
                        f"outside its bin range [{self.bin_edges[index]}, "
                        f"{self.bin_edges[index + 1]}].",
                    )
        elif isinstance(self.output_grid, tuple) and isinstance(
            self.bin_edges,
            (tuple, list),
        ):
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

        if self.dim == 1 and isinstance(self.bin_edges, np.ndarray):
            binned_values = self._compute_univariate_domain_binning(
                grid_points[0],
                data_matrix,
                self.bin_edges,
            )
            binned_values = np.array(binned_values)
            self.output_grid = [np.array(self.output_grid)]

        elif self.bin_edges is not None:
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

        if (
            self.dim == 1
            and isinstance(self.bin_edges, np.ndarray)
            and isinstance(self.output_grid, np.ndarray)
        ):
            self.bin_edges = (self.bin_edges,)
            self.output_grid = (self.output_grid,)

        if self.output_grid is not None:
            grid_shape = tuple(len(g) for g in self.output_grid)
            binned_values = np.full(  # noqa: WPS317
                (n_samples, *grid_shape, dim_codomain),
                np.nan,
            )

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

            bin_indices = self._get_bin_indices(dim_domain, arg_array)

            for bin_idx in np.ndindex(*grid_shape):
                mask = np.all(bin_indices == bin_idx, axis=1)
                points_in_bin = val_array[mask]

                if points_in_bin.size > 0:
                    binned_values[sample_idx][bin_idx] = (
                        np.nanmean(points_in_bin, axis=0)
                        if self.bin_aggregation == "mean"
                        else np.nanmedian(points_in_bin, axis=0)
                    )
        if self.output_grid is not None:
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

    def _get_bin_indices(
        self,
        dim_domain: int,
        arg_array: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute bin indices for the given arguments."""
        n_points = arg_array.shape[0]
        bin_indices = np.empty((n_points, dim_domain), dtype=int)

        for d in range(dim_domain):
            if self.bin_edges is not None:
                bin_edges_d = self.bin_edges[d]
                arg_array_d = arg_array[:, d]

                bin_indices[:, d] = (
                    np.digitize(arg_array_d, bin_edges_d, right=False) - 1
                )

                out_of_range = (arg_array_d < bin_edges_d[0]) | (
                    arg_array_d > bin_edges_d[-1]
                )
                bin_indices[:, d][out_of_range] = -1

                last_bin_index = len(bin_edges_d) - 2
                bin_indices[:, d][
                    arg_array_d == bin_edges_d[-1]
                ] = last_bin_index

        return bin_indices

    def _compute_univariate_domain_binning(
        self,
        grid_points: NDArray[np.float64],
        data_matrix: NDArray[np.float64],
        bin_edges: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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
        binned_values = np.full(  # noqa: WPS317
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
        data_matrix: NDArray[np.float64],
        points_in_bin: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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
        grid_points: tuple[NDArray[np.float64], ...],
        data_matrix: NDArray[np.float64],
        bin_edges: ArrayNDImType,
    ) -> NDArray[np.float64]:
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
        points_in_bin = self._get_points_in_bin(grid_points, bin_edges)
        points_in_bin_combinations = list(itertools.product(*points_in_bin))

        if self.n_bins is not None:
            output_values = np.full(  # noqa: WPS317
                (n_samples, *tuple(self.n_bins), n_codomain),
                np.nan,
            )

        for k, combination in enumerate(points_in_bin_combinations):
            combination_data = self._filter_combination_data(
                data_matrix,
                combination,
            )

            if np.any(~np.isnan(combination_data)):
                combination_output_values = (
                    self._compute_multivariate_bin_values(combination_data)
                )
                self._assign_output_values(
                    output_values,
                    combination_output_values,
                    k,
                )

        return output_values

    def _get_points_in_bin(
        self,
        grid_points: tuple[NDArray[np.float64], ...],
        bin_edges: ArrayNDImType,
    ) -> list[list[Any]]:
        """Get boolean masks for points in bins."""
        points_in_bin = []
        for i in range(self.dim):
            bin_edges_i = bin_edges[i]
            masks = []
            for j in range(len(bin_edges_i) - 1):
                mask = (grid_points[i] >= bin_edges_i[j]) & (
                    grid_points[i] < bin_edges_i[j + 1]
                )
                if j == len(bin_edges_i) - 2:
                    mask |= grid_points[i] == bin_edges_i[j + 1]
                masks.append(mask)
            points_in_bin.append(masks)
        return points_in_bin

    def _filter_combination_data(
        self,
        data_matrix: NDArray[np.float64],
        combination: tuple[Any, ...],
    ) -> NDArray[np.float64]:
        """Filter the data matrix based on the bin combination masks."""
        combination_data = data_matrix
        for dim, comb_mask in enumerate(combination, start=1):
            index_tuple = (slice(None),) * dim + (comb_mask,)
            combination_data = combination_data[index_tuple]
        return combination_data

    def _assign_output_values(
        self,
        output_values: NDArray[np.float64],
        combination_output_values: NDArray[np.float64],
        k: int,
    ) -> None:
        """Assign computed values to the output array."""
        if self.n_bins is not None:
            multi_dim_index = np.unravel_index(k, self.n_bins)
        for obs_idx, value in enumerate(combination_output_values):
            index_tuple = (obs_idx,) + multi_dim_index + (slice(None),)
            output_values[index_tuple] = value

    def _compute_multivariate_bin_values(
        self,
        data_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
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
            obs_data = self._get_obs_data(data_matrix, bin_shape, codomain_dim)

            for obs in range(n_samples):
                if not np.all(np.isnan(obs_data[obs])):
                    output_values[obs, codomain_dim] = self._aggregate_data(
                        obs_data[obs],
                    )

        return output_values

    def _get_obs_data(
        self,
        data_matrix: NDArray[np.float64],
        bin_shape: list[int],
        codomain_dim: int,
    ) -> NDArray[np.float64]:
        """Extract observed data for the given codomain dimension."""
        obs_data = []
        for idx in np.ndindex(*bin_shape):
            obs_data.append(
                data_matrix[(slice(None),) + idx + (codomain_dim,)],
            )
        return np.array(obs_data).T

    def _aggregate_data(self, obs_data: NDArray[np.float64]) -> float:
        """
        Aggregate the data.

        Aggregate the data based on the specified bin aggregation method.
        """
        if self.bin_aggregation == "mean":
            return float(np.nanmean(obs_data))

        return float(np.nanmedian(obs_data))

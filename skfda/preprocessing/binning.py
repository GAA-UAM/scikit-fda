"""Data Binning Module with DataBinner class."""

import warnings
from typing import Optional

import numpy as np

from skfda._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from skfda.representation import FDataGrid


class DataBinner(
    BaseEstimator,
    TransformerMixin[FDataGrid, FDataGrid, object],
):
    """
    Data Binner.

    Class to group the grid points of a FDataGrid into bins. The values
    of the new grid points, which are the midpoints of the bins, are
    computed based on the method specified in the mode.

    It follows the scikit-learn methodology of TransformerMixin, so it
    works with the fit and transform methods.

    Parameters:
        n_bins: Number of bins. This parameter is incompatible to
            bin_edges.
        bin_edges: Array with the specific bin edges. Must be a sorted
            array. This parameter is incompatible to n_bins.
        mode: Method to compute the value of the bin. The available
            methods are: 'mean', 'median', 'weighted_mean'.
        non_empty: If True, exclude empty bins from the output grid.

    Attributes:
        n_bins: Number of bins. Always defined after fitting.
        bin_edges: Array with the specific bin edges. Always defined
            after fitting.
        mode: Method to compute the value of the bin.
        non_empty: If True, exclude empty bins from the output grid.
        min_domain: Minimum value of the domain range of the FDataGrid.
            Does not have to match that of the input FDataGrid.
        max_domain: Maximum value of the domain range of the FDataGrid.
            Does not have to match that of the input FDataGrid.

    Examples:
        Given a FDataGrid with 10 grid points and 2 functions, bin the data
        into 3 bins.

        >>> import numpy as np
        >>> import skfda
        >>> grid_points = np.linspace(0, 10, 10)
        >>> data_matrix = np.array(
        >>>     [
        >>>         [2, 2, 3, 3, 4, 4, 5, 5, np.nan, np.nan],
        >>>         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        >>>     ]
        >>> )
        >>> fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
        >>> bin_edges = np.array([-1, 1, 4, 10])
        >>> binner = DataBinner(bin_edges=bin_edges, non_empty=False)
        >>> binned_fd = binner.fit_transform(fd)
    """

    def __init__(
        self,
        *,
        n_bins: Optional[int] = None,
        bin_edges: Optional[np.ndarray] = None,
        mode: str = "mean",
        non_empty: bool = False,
    ):
        if (n_bins is not None and bin_edges is not None) or (
            n_bins is None and bin_edges is None
        ):
            raise ValueError(
                "Specify either n_bins or bin_edges, but not both.",
            )

        if bin_edges is not None:
            bin_edges = np.asarray(bin_edges, dtype=float)
            if bin_edges.ndim != 1 or len(bin_edges) < 2:
                raise ValueError(
                    "bin_edges must be a 1-dimensional array with at least "
                    "two elements.",
                )
            if np.any(np.isnan(bin_edges)) or not np.all(
                np.diff(bin_edges) > 0
            ):
                raise ValueError(
                    "Values in bin_edges have to be strictly increasing and "
                    "without any NaN values.",
                )

        if mode not in {"mean", "median", "weighted_mean"}:
            raise ValueError(
                "Mode must be one of 'mean', 'median', or 'weighted_mean'.",
            )

        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.mode = mode
        self.non_empty = non_empty

    def fit(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> "DataBinner":
        """
        Prepare the binning parameters based on the domain range.

        Args:
            X: FDataGrid to be binned.
            y: Ignored.

        Returns:
            self
        """
        domain_range = X.domain_range[0]
        self.min_domain = domain_range[0]
        self.max_domain = domain_range[1]

        if self.n_bins:
            self.bin_edges = np.linspace(
                self.min_domain, self.max_domain, self.n_bins + 1
            )
        else:
            self.n_bins = len(self.bin_edges) - 1

        if (
            self.bin_edges[0] < self.min_domain
            or self.bin_edges[-1] > self.max_domain
        ):
            warnings.warn(
                "Some bin edges are outside the domain range of the "
                "FDataGrid. If a bin is completely outside the domain, it "
                "will not contain any data.",
                UserWarning,
                stacklevel=2,
            )
            if self.bin_edges[0] < self.min_domain:
                self.min_domain = self.bin_edges[0]

            if self.bin_edges[-1] > self.max_domain:
                self.max_domain = self.bin_edges[-1]

        return self

    @staticmethod
    def _weighted_mean(
        data: np.ndarray,
        distances: np.ndarray,
        max_distance: float,
    ) -> float:
        """
        Compute the weighted mean of the data.

        Compute the weighted mean of the data based on distances from the
        center, ignoring NaN values.

        Args:
            data: Input data array.
            distances: Array of distances from the bin center.
            max_distance: Maximum distance from the bin center to an edge.

        Returns:
            Weighted mean of the data.
        """
        data = data.ravel()
        valid = ~np.isnan(data)

        if not np.any(valid):
            return np.nan

        # Scale distances to weights in the range [0.8, 1.0]
        weights = 1.0 - 0.2 * (distances / max_distance)

        return np.sum(data[valid] * weights[valid]) / np.sum(weights[valid])

    def _compute_bin_values(
        self,
        data_matrix: np.ndarray,
        points_in_bin: np.ndarray,
        midpoints: np.ndarray,
        i: int,
        grid_points: np.ndarray,
    ) -> np.ndarray:
        """
        Compute bin values based on the specified mode.

        Args:
            data_matrix: The data matrix from the FDataGrid.
            points_in_bin: Boolean array indicating points in the current bin.
            midpoints: Array of bin midpoints.
            i: Index of the current bin.
            grid_points: Original grid points of the FDataGrid.

        Returns:
            Array of computed bin values for the current bin.
        """
        mean_values = np.full(data_matrix.shape[0], np.nan)

        for row_idx in range(data_matrix.shape[0]):
            row_data = data_matrix[row_idx, points_in_bin]

            if not np.all(np.isnan(row_data)):
                if self.mode == "mean":
                    mean_values[row_idx] = np.nanmean(row_data)
                elif self.mode == "median":
                    mean_values[row_idx] = np.nanmedian(row_data)
                elif self.mode == "weighted_mean":
                    distances = np.abs(
                        grid_points[points_in_bin] - midpoints[i]
                    )
                    max_distance = np.abs(self.bin_edges[i] - midpoints[i])
                    mean_values[row_idx] = self._weighted_mean(
                        row_data,
                        distances,
                        max_distance,
                    )
        return mean_values

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
        grid_points = X.grid_points[0]
        data_matrix = X.data_matrix

        midpoints = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        binned_values = []
        counts = []

        for i in range(len(self.bin_edges) - 1):
            points_in_bin = (grid_points >= self.bin_edges[i]) & (
                grid_points < self.bin_edges[i + 1]
            )
            if i == len(self.bin_edges) - 2:
                # Include right endpoint in the last bin
                points_in_bin |= grid_points == self.bin_edges[i + 1]

            if ~np.all(np.isnan(data_matrix[:, points_in_bin])):
                mean_values = self._compute_bin_values(
                    data_matrix, points_in_bin, midpoints, i, grid_points,
                )
                counts.append(np.sum(points_in_bin))
                binned_values.append(mean_values)
            else:
                counts.append(0)
                binned_values.append(np.full(data_matrix.shape[0], np.nan))

        binned_values = np.array(binned_values).T

        if self.non_empty:
            mask = np.array(counts) > 0
            midpoints = midpoints[mask]
            binned_values = binned_values[:, mask]

        return X.copy(
            data_matrix=binned_values,
            grid_points=midpoints,
            domain_range=(self.min_domain, self.max_domain),
        )

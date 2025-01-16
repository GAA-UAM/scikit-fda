from typing import Optional

import numpy as np

from skfda._utils._sklearn_adapter import BaseEstimator, TransformerMixin
from skfda.representation import FDataGrid


class DataBinner(
    BaseEstimator,
    TransformerMixin[FDataGrid, FDataGrid, object],
):
    def __init__(
        self,
        *,
        n_bins: Optional[int] = None,
        bin_width: Optional[int] = None,
        mode: str = "mean",
        non_empty: bool = False,
    ):
        r"""
        Args:
            n_bins: Number of bins. This parameter is incompatible to
                bin_width.
            bin_width: Relative width of each bin. Let :math:`D`, `d` be the
                maximum and minimum points in the domain range, respectively,
                and :math:`\text{bw}` the initial bin width specified by the
                user. Let :math:`\text{w}` be the adjusted bin width. It will
                be computed as:
                    .. math::
                        w = \frac{D - d}{\lceil \frac{D - d}{bw} \rceil}
                This parameter is incompatible to n_bins.
            mode: Method to compute the value of the bin. The available
                methods are: 'mean', 'median', 'weighted_mean'.
            non_empty: If True, exclude empty bins from the output grid.
        """
        if n_bins is not None and bin_width is not None:
            raise ValueError(
                "Only one of n_bins and bin_width can be specified."
            )

        if mode not in {"mean", "median", "weighted_mean"}:
            raise ValueError(
                "Mode must be one of 'mean', 'median', or 'weighted_mean'."
            )

        self.n_bins = n_bins
        self.bin_width = bin_width
        self.mode = mode
        self.non_empty = non_empty

    def fit(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> "DataBinner":
        """
        Prepares the binning parameters based on the domain range.

        Args:
            X: FDataGrid to be binned.
            y: Ignored.

        Returns:
            self
        """
        domain_range = X.domain_range[0]
        self.min_domain, self.max_domain = domain_range

        if self.bin_width:
            self.n_bins = int(
                np.ceil((self.max_domain - self.min_domain) / self.bin_width)
            )
        elif not self.n_bins:
            # Function to automatically calculate n bins
            pass

        return self

    @staticmethod
    def _weighted_mean(
        data: np.ndarray, distances: np.ndarray, max_distance: float
    ) -> float:
        """
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

        bin_edges = np.linspace(
            self.min_domain, self.max_domain, self.n_bins + 1
        )
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        binned_values = []
        counts = []

        for i in range(len(bin_edges) - 1):
            points_in_bin = (grid_points >= bin_edges[i]) & (
                grid_points < bin_edges[i + 1]
            )
            if i == len(bin_edges) - 2:
                # Include right endpoint in the last bin
                points_in_bin |= grid_points == bin_edges[i + 1]

            if not np.all(np.isnan(data_matrix[:, points_in_bin])):
                mean_values = np.full(data_matrix.shape[0], np.nan)

                for row_idx in range(data_matrix.shape[0]):
                    row_data = data_matrix[row_idx, points_in_bin]

                    # Only calculate the mean if there are elements
                    if not np.all(np.isnan(row_data)):
                        if self.mode == "mean":
                            mean_values[row_idx] = np.nanmean(row_data)
                        elif self.mode == "median":
                            mean_values[row_idx] = np.nanmedian(row_data)
                        elif self.mode == "weighted_mean":
                            distances = np.abs(
                                grid_points[points_in_bin] - midpoints[i]
                            )
                            max_distance = np.abs(bin_edges[i] - midpoints[i])
                            mean_values[row_idx] = self._weighted_mean(
                                row_data, distances, max_distance
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
        )

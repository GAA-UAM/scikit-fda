"""Benchmarks for KMeans."""
from typing import Any

import numpy as np

from skfda.ml.clustering import KMeans
from skfda.representation.basis import FDataBasis, FourierBasis
from skfda.representation.grid import FDataGrid


class TimeKMeans:
    """Performance of :class:`skfda.ml.clustering.KMeans` for FDataGrid."""
    fdatagrid: FDataGrid
    fd_basis: FDataBasis
    kmeans_fdgrid: KMeans
    kmeans_fdbasis: KMeans

    def setup(self) -> None:
        """Create the data for the test."""
        t: np.ndarray = np.linspace(0, 1, 100)
        data_matrix: np.ndarray = (
            np.sin(2 * np.pi * t)[None, :] + np.random.randn(50, 100) * 0.1
        )
        self.fdatagrid = FDataGrid(data_matrix=data_matrix, grid_points=t)

        basis: FourierBasis = FourierBasis(domain_range=(0, 1), n_basis=7)
        self.fd_basis = self.fdatagrid.to_basis(basis)

        self.kmeans_fdgrid = KMeans(n_clusters=3)
        self.kmeans_fdbasis = KMeans(n_clusters=3)

    def time_fit_fdgrid(self) -> None:
        """Time to fit KMeans to a FDataGrid."""
        self.kmeans_fdgrid.fit(self.fdatagrid)


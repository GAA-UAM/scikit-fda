"""Benchmarks for dealing with missing data."""

import numpy as np
from asv_runner.benchmarks.mark import parameterize

from skfda import FDataGrid
from skfda.preprocessing.missing import MissingValuesInterpolation

seed = 3155136


@parameterize({
    "n_samples": [10, 100, 1000],
    "n_points": [10, 100, 1000],
    "na_probability": [0.1, 0.5, 0.9],
})
class MissingValuesInterpolationCurves:
    """Performance of :class:`MissingValuesInterpolation` for curves (1D)."""

    def setup(
        self,
        n_samples: int,
        n_points: int,
        na_probability: float,
    ) -> None:
        """Create the data for the test."""
        self.rng = np.random.default_rng()
        size = (n_samples, n_points)
        na_positions = self.rng.choice(
            [True, False],
            p=[na_probability, 1 - na_probability],
            size=size,
        )
        values = self.rng.standard_normal(size=size)
        values[na_positions] = np.nan
        self.data = FDataGrid(values)
        self.transformer = MissingValuesInterpolation[FDataGrid]()

    def time_missing_interpolation(
        self,
        n_samples: int,
        n_points: int,
        na_probability: float,
    ) -> None:
        """Time the interpolation of missing values."""
        self.transformer.fit_transform(self.data)

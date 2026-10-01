"""Benchmarks for dealing with Fisher-Rao elastic registration."""

import numpy as np
from asv_runner.benchmarks.mark import parameterize

from skfda.datasets import make_multimodal_samples
from skfda.preprocessing.registration import FisherRaoElasticRegistration

seed = 715839


@parameterize({
    "n_samples": [10, 100, 1000],
    "n_points": [10, 100, 125],
})
class FisherRaoRegistration:
    """Performance of Fisher-Rao registration."""

    timeout=120

    def setup(
        self,
        n_samples: int,
        n_points: int,
    ) -> None:
        """Create the data for the test."""
        self.rng = np.random.default_rng(seed)

        self.data = make_multimodal_samples(
            n_samples=n_samples,
            points_per_dim=n_points,
            n_modes=2,
            random_state=self.rng,
        )
        self.transformer = FisherRaoElasticRegistration()

    def time_fisher_rao_registration(
        self,
        n_samples: int,
        n_points: int,
    ) -> None:
        """Time the interpolation of missing values."""
        self.transformer.fit_transform(self.data)

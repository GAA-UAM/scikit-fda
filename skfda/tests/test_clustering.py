"""Tests for clustering methods."""
import unittest

import numpy as np

import skfda
from skfda.datasets import make_gaussian_process
from skfda.misc.covariances import Brownian
from skfda.ml.clustering import FuzzyCMeans, KMeans
from skfda.representation.grid import FDataGrid


class TestKMeans(unittest.TestCase):
    """Test the KMeans clustering method."""

    def test_distinct_clusters(self) -> None:
        """
        Test with two clearly distinct clusters.

        We use two Brownian processes with different means (-1 and 1), so
        clustering should be perfect.
        """
        n_features = 100
        n_samples = 1000
        random_state = np.random.RandomState(0)

        cluster_0 = make_gaussian_process(
            n_samples=n_samples,
            n_features=n_features,
            mean=-1,
            cov=Brownian(variance=0.01),
            random_state=random_state,
        )
        cluster_1 = make_gaussian_process(
            n_samples=n_samples,
            n_features=n_features,
            mean=1,
            cov=Brownian(variance=0.01),
            random_state=random_state,
        )

        X = skfda.concatenate((cluster_0, cluster_1))

        kmeans = KMeans[FDataGrid](random_state=random_state)
        prediction = kmeans.fit_predict(X)

        np.testing.assert_allclose(
            kmeans.cluster_centers_.data_matrix[..., 0],
            [
                np.ones(n_features),
                -np.ones(n_features),
            ],
            rtol=1e-2,
        )

        np.testing.assert_equal(
            prediction,
            np.concatenate([
                np.ones(n_samples),
                np.zeros(n_samples),
            ]),
        )

    def test_predict_does_not_modify_centers(self) -> None:
        """Test that predicting new data does not change the centers."""
        X = FDataGrid([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.1, 5.0],
        ])

        kmeans = KMeans[FDataGrid](random_state=0)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_.data_matrix.copy()

        kmeans.predict(X + 1)

        np.testing.assert_array_equal(
            kmeans.cluster_centers_.data_matrix,
            centers,
        )

    def test_fit_does_not_modify_init(self) -> None:
        """Test that fitting does not change the initial centers."""
        X = FDataGrid([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.1, 5.0],
        ])
        init = FDataGrid([
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0],
        ])
        init_matrix = init.data_matrix.copy()

        kmeans = KMeans[FDataGrid](init=init)
        kmeans.fit(X)

        np.testing.assert_array_equal(init.data_matrix, init_matrix)


class TestFuzzyCMeans(unittest.TestCase):
    """Test the FuzzyCMeans clustering method."""

    def test_distinct_clusters(self) -> None:
        """
        Test with two clearly distinct clusters.

        We use two Brownian processes with different means (-1 and 1), so
        clustering should be perfect.
        """
        n_features = 100
        n_samples = 1000
        random_state = np.random.RandomState(0)

        cluster_0 = make_gaussian_process(
            n_samples=n_samples,
            n_features=n_features,
            mean=-1,
            cov=Brownian(variance=0.01),
            random_state=random_state,
        )
        cluster_1 = make_gaussian_process(
            n_samples=n_samples,
            n_features=n_features,
            mean=1,
            cov=Brownian(variance=0.01),
            random_state=random_state,
        )

        X = skfda.concatenate((cluster_0, cluster_1))

        fcmeans = FuzzyCMeans[FDataGrid](random_state=random_state)
        prediction = fcmeans.fit_predict(X)

        np.testing.assert_allclose(
            fcmeans.cluster_centers_.data_matrix[..., 0],
            [
                np.ones(n_features),
                -np.ones(n_features),
            ],
            rtol=1e-2,
        )

        np.testing.assert_equal(
            prediction,
            np.concatenate([
                np.ones(n_samples),
                np.zeros(n_samples),
            ]),
        )

        predict_proba = fcmeans.predict_proba(X)
        true_proba = np.zeros((n_samples * 2, 2))
        true_proba[n_samples:, 0] = 1
        true_proba[:n_samples, 1] = 1

        # We need to use atol because we compare against 0.
        np.testing.assert_allclose(
            predict_proba,
            true_proba,
            atol=1e-1,
        )

    def test_predict_proba_does_not_modify_centers(self) -> None:
        """Test that repeated predictions do not change the model."""
        X = FDataGrid([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.1, 5.0],
        ])

        fcmeans = FuzzyCMeans[FDataGrid](random_state=0)
        fcmeans.fit(X)
        centers = fcmeans.cluster_centers_.data_matrix.copy()

        first_proba = fcmeans.predict_proba(X + 1)
        second_proba = fcmeans.predict_proba(X + 1)

        np.testing.assert_array_equal(
            fcmeans.cluster_centers_.data_matrix,
            centers,
        )
        np.testing.assert_allclose(first_proba, second_proba)


if __name__ == '__main__':
    unittest.main()

"""Test neighbors classifiers and regressors."""
from __future__ import annotations

import unittest
from typing import Any, Sequence

import numpy as np
from sklearn.neighbors._base import KNeighborsMixin, RadiusNeighborsMixin

from skfda.datasets import make_multimodal_samples, make_sinusoidal_process
from skfda.exploratory.outliers import LocalOutlierFactor  # Pending theory
from skfda.misc.metrics import PairwiseMetric, l2_distance
from skfda.ml.classification import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
)
from skfda.ml.clustering import NearestNeighbors
from skfda.ml.regression import KNeighborsRegressor, RadiusNeighborsRegressor
from skfda.representation import FDataBasis, FDataGrid
from skfda.representation.basis import FourierBasis


class TestNeighbors(unittest.TestCase):
    """Tests for neighbors methods."""

    def setUp(self) -> None:
        """Create test data."""
        random_state = np.random.RandomState(0)
        modes_location = np.concatenate((
            random_state.normal(-0.3, 0.04, size=15),
            random_state.normal(0.3, 0.04, size=15),
        ))

        idx = np.arange(30)
        random_state.shuffle(idx)

        modes_location = modes_location[idx]
        self.modes_location = modes_location
        self.y = np.array(15 * [0] + 15 * [1])[idx]

        self.X = make_multimodal_samples(
            n_samples=30,
            modes_location=modes_location,
            noise=0.05,
            random_state=random_state,
        )
        self.X2 = make_multimodal_samples(
            n_samples=30,
            modes_location=modes_location,
            noise=0.05,
            random_state=1,
        )

        self.probs = np.array(15 * [[1.0, 0.0]] + 15 * [[0.0, 1.0]])[idx]

        # Dataset with outliers
        fd_clean = make_sinusoidal_process(
            n_samples=25,
            error_std=0,
            phase_std=0.1,
            random_state=0,
        )
        fd_outliers = make_sinusoidal_process(
            n_samples=2,
            error_std=0,
            phase_mean=0.5,
            random_state=5,
        )
        self.fd_lof = fd_outliers.concatenate(fd_clean)

    def test_predict_classifier(self) -> None:
        """Tests predict for neighbors classifier."""
        classifiers: Sequence[
            KNeighborsClassifier[FDataGrid]
            | RadiusNeighborsClassifier[FDataGrid]
        ] = (
            KNeighborsClassifier(),
            RadiusNeighborsClassifier(radius=0.1),
        )

        for neigh in classifiers:
            neigh.fit(self.X, self.y)
            pred = neigh.predict(self.X)
            np.testing.assert_array_equal(
                pred,
                self.y,
                err_msg=f'fail in {type(neigh)}',
            )

    def test_predict_proba_classifier(self) -> None:
        """Tests predict proba for k neighbors classifier."""
        neigh = KNeighborsClassifier(metric=l2_distance)

        neigh.fit(self.X, self.y)
        probs = neigh.predict_proba(self.X)

        np.testing.assert_array_almost_equal(probs, self.probs)

    def test_predict_regressor(self) -> None:
        """Test scalar regression, predicts mode location."""
        # Dummy test, with weight = distance, only the sample with distance 0
        # will be returned, obtaining the exact location
        knnr = KNeighborsRegressor[FDataGrid, np.typing.NDArray[np.float_]](
            weights='distance',
        )
        rnnr = RadiusNeighborsRegressor[
            FDataGrid,
            np.typing.NDArray[np.float_],
        ](
            weights='distance',
            radius=0.1,
        )

        knnr.fit(self.X, self.modes_location)
        rnnr.fit(self.X, self.modes_location)

        np.testing.assert_array_almost_equal(
            knnr.predict(self.X),
            self.modes_location,
        )
        np.testing.assert_array_almost_equal(
            rnnr.predict(self.X),
            self.modes_location,
        )

    def test_kneighbors(self) -> None:
        """Test k neighbor searches for all k-neighbors estimators."""
        nn = NearestNeighbors()
        nn.fit(self.X)

        lof = LocalOutlierFactor(n_neighbors=5)
        lof.fit(self.X)

        knn = KNeighborsClassifier()
        knn.fit(self.X, self.y)

        knnr = KNeighborsRegressor[FDataGrid, np.typing.NDArray[np.float_]]()
        knnr.fit(self.X, self.modes_location)

        neigh: KNeighborsMixin[FDataGrid, Any]
        for neigh in (nn, knn, knnr, lof):

            dist, links = neigh.kneighbors(self.X[:4])

            np.testing.assert_array_equal(
                links,
                [[0, 7, 21, 23, 15],
                 [1, 12, 19, 18, 17],
                 [2, 17, 22, 27, 26],
                 [3, 4, 9, 5, 25],
                 ],
            )

            graph = neigh.kneighbors_graph(self.X[:4])

            dist_kneigh = l2_distance(self.X[0], self.X[7])

            np.testing.assert_array_almost_equal(dist[0, 1], dist_kneigh)

            for i in range(30):
                self.assertEqual(graph[0, i] == 1, i in links[0])
                self.assertEqual(graph[0, i] == 0, i not in links[0])

    def test_radius_neighbors(self) -> None:
        """Test query with radius."""
        nn = NearestNeighbors(radius=0.1)
        nn.fit(self.X)

        knn = RadiusNeighborsClassifier(radius=0.1)
        knn.fit(self.X, self.y)

        knnr = RadiusNeighborsRegressor[
            FDataGrid,
            np.typing.NDArray[np.float_],
        ](radius=0.1)
        knnr.fit(self.X, self.modes_location)

        neigh: RadiusNeighborsMixin[FDataGrid, Any]
        for neigh in (nn, knn, knnr):

            dist, links = neigh.radius_neighbors(self.X[:4])

            np.testing.assert_array_equal(links[0], np.array([0, 7]))
            np.testing.assert_array_equal(links[1], np.array([1]))
            np.testing.assert_array_equal(links[2], np.array([2, 17, 22, 27]))
            np.testing.assert_array_equal(links[3], np.array([3, 4, 9]))

            dist_kneigh = l2_distance(self.X[0], self.X[7])

            np.testing.assert_array_almost_equal(dist[0][1], dist_kneigh)

            graph = neigh.radius_neighbors_graph(self.X[:4])

            for i in range(30):
                self.assertEqual(graph[0, i] == 1, i in links[0])
                self.assertEqual(graph[0, i] == 0, i not in links[0])

    def test_knn_functional_response(self) -> None:
        """Test prediction of functional response."""
        knnr = KNeighborsRegressor[FDataGrid, FDataGrid](n_neighbors=1)

        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.data_matrix,
            self.X.data_matrix,
        )

    def test_knn_functional_response_precomputed(self) -> None:
        """Test that precomputed distances work for functional response."""
        knnr = KNeighborsRegressor[
            np.typing.NDArray[np.float_],
            FDataGrid,
        ](
            n_neighbors=4,
            weights='distance',
            metric='precomputed',
        )
        d = PairwiseMetric(l2_distance)
        distances = d(self.X[:4], self.X[:4])

        knnr.fit(distances, self.X[:4])

        res = knnr.predict(distances)
        np.testing.assert_array_almost_equal(
            res.data_matrix, self.X[:4].data_matrix,
        )

    def test_radius_functional_response(self) -> None:
        """Test that radius regression work with functional response."""
        knnr = RadiusNeighborsRegressor[
            FDataGrid,
            FDataGrid,
        ](
            metric=l2_distance,
            weights='distance',
        )

        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.data_matrix, self.X.data_matrix,
        )

    def test_functional_response_custom_weights(self) -> None:
        """Test that custom weights work with functional response."""
        knnr = KNeighborsRegressor[
            FDataGrid,
            FDataGrid,
        ](weights=self._weights, n_neighbors=5)
        response = self.X.to_basis(
            FourierBasis(domain_range=(-1, 1), n_basis=10),
        )
        knnr.fit(self.X, response)

        res = knnr.predict(self.X)
        np.testing.assert_allclose(
            res.coefficients, response.coefficients,
        )

    def test_functional_response_distance_weights(self) -> None:
        """Test that distance weights work with functional response."""
        knnr = KNeighborsRegressor[
            FDataGrid,
            FDataGrid,
        ](
            weights='distance',
            n_neighbors=10,
        )
        knnr.fit(self.X[:10], self.X[:10])
        res = knnr.predict(self.X[11])

        d = PairwiseMetric(l2_distance)
        distances = d(self.X[:10], self.X[11]).flatten()

        weights = 1 / distances
        weights /= weights.sum()

        response = (self.X[:10] * weights).sum()
        np.testing.assert_array_almost_equal(
            res.data_matrix, response.data_matrix,
        )

    def test_functional_response_basis(self) -> None:
        """Test FDataBasis response."""
        knnr = KNeighborsRegressor[
            FDataGrid,
            FDataBasis,
        ](weights='distance', n_neighbors=5)
        response = self.X.to_basis(
            FourierBasis(domain_range=(-1, 1), n_basis=10),
        )
        knnr.fit(self.X, response)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.coefficients, response.coefficients,
        )

    def test_radius_outlier_functional_response(self) -> None:
        """Test response with no neighbors."""
        # Test response
        knnr = RadiusNeighborsRegressor[
            FDataGrid,
            FDataGrid,
        ](
            radius=0.001,
        )
        knnr.fit(self.X[:6], self.X[:6])

        res = knnr.predict(self.X[:7])
        np.testing.assert_array_almost_equal(
            res[6].data_matrix, np.nan,
        )

    def test_functional_regressor_exceptions(self) -> None:
        """Test exception with unequal sizes."""
        knnr = RadiusNeighborsRegressor[
            FDataGrid,
            FDataBasis,
        ]()

        with np.testing.assert_raises(ValueError):
            knnr.fit(self.X[:3], self.X[:4])

    def test_search_neighbors_precomputed(self) -> None:
        """Test search neighbors with precomputed distances."""
        d = PairwiseMetric(l2_distance)
        distances = d(self.X[:4], self.X[:4])

        nn = NearestNeighbors(metric='precomputed', n_neighbors=2)
        nn.fit(distances, self.y[:4])

        _, neighbors = nn.kneighbors(distances)

        np.testing.assert_array_almost_equal(
            neighbors,
            np.array([[0, 3], [1, 2], [2, 1], [3, 0]]),
        )

    def test_score_scalar_response(self) -> None:
        """Test regression with scalar response."""
        neigh = KNeighborsRegressor[
            FDataGrid,
            np.typing.NDArray[np.float_],
        ]()

        neigh.fit(self.X, self.modes_location)
        r = neigh.score(self.X, self.modes_location)
        np.testing.assert_almost_equal(r, 0.9975889963743335)

    def test_score_functional_response(self) -> None:
        """Test functional score."""
        neigh = KNeighborsRegressor[
            FDataGrid,
            FDataGrid,
        ]()

        y = 5 * self.X + 1
        neigh.fit(self.X, y)
        r = neigh.score(self.X, y)
        np.testing.assert_almost_equal(r, 0.65599399478951)

        # Weighted case and basis form
        y = y.to_basis(FourierBasis(domain_range=y.domain_range[0], n_basis=5))
        neigh.fit(self.X, y)

        r = neigh.score(
            self.X[:7],
            y[:7],
            sample_weight=np.array(4 * [1.0 / 5] + 3 * [1.0 / 15]),
        )
        np.testing.assert_almost_equal(r, 0.9802105817331564)

    def test_score_functional_response_exceptions(self) -> None:
        """Test weights with invalid length."""
        neigh = RadiusNeighborsRegressor[
            FDataGrid,
            FDataGrid,
        ]()
        neigh.fit(self.X, self.X)

        with np.testing.assert_raises(ValueError):
            neigh.score(self.X, self.X, sample_weight=np.array([1, 2, 3]))

    def test_lof_fit_predict(self) -> None:
        """Test same results with different forms to call fit_predict."""
        # Outliers
        expected = np.ones(len(self.fd_lof))
        expected[:2] = -1

        # With default l2 distance
        lof = LocalOutlierFactor()
        res = lof.fit_predict(self.fd_lof)
        np.testing.assert_array_equal(expected, res)

        # With explicit l2 distance
        lof2 = LocalOutlierFactor(metric=l2_distance)
        res2 = lof2.fit_predict(self.fd_lof)
        np.testing.assert_array_equal(expected, res2)

        d = PairwiseMetric(l2_distance)
        distances = d(self.fd_lof, self.fd_lof)

        # With precompute distances
        lof3 = LocalOutlierFactor(metric='precomputed')
        res3 = lof3.fit_predict(distances)
        np.testing.assert_array_equal(expected, res3)

        # Check values of negative outlier factor
        negative_lof = [  # noqa: WPS317
            -7.1068, -1.5412, -0.9961,
            -0.9854, -0.9896, -1.0993,
            -1.065, -0.9871, -0.9821,
            -0.9955, -1.0385, -1.0072,
            -0.9832, -1.0134, -0.9939,
            -1.0074, -0.992, -0.992,
            -0.9883, -1.0012, -1.1149,
            -1.002, -0.9994, -0.9869,
            -0.9726, -0.9989, -0.9904,
        ]

        np.testing.assert_array_almost_equal(
            lof.negative_outlier_factor_.round(4), negative_lof,
        )

        # Check same negative outlier factor
        np.testing.assert_array_almost_equal(
            lof.negative_outlier_factor_,
            lof2.negative_outlier_factor_,
        )

        np.testing.assert_array_almost_equal(
            lof.negative_outlier_factor_,
            lof3.negative_outlier_factor_,
        )

    def test_lof_decision_function(self) -> None:
        """Test decision function and score samples of LOF."""
        lof = LocalOutlierFactor(novelty=True)
        lof.fit(self.fd_lof[5:])

        score = lof.score_samples(self.fd_lof[:5])

        np.testing.assert_array_almost_equal(
            score.round(4),
            [-5.9726, -1.3445, -0.9853, -0.9817, -0.985],
            err_msg='Error in LocalOutlierFactor.score_samples',
        )

        # Test decision_function = score_function - offset
        np.testing.assert_array_almost_equal(
            lof.decision_function(self.fd_lof[:5]),
            score - lof.offset_,
            err_msg='Error in LocalOutlierFactor.decision_function',
        )

    def test_lof_exceptions(self) -> None:
        """Test error due to novelty attribute."""
        lof = LocalOutlierFactor(novelty=True)

        # Error in fit_predict function
        with np.testing.assert_raises(AttributeError):
            lof.fit_predict(self.fd_lof[5:])

        lof.set_params(novelty=False)
        lof.fit(self.fd_lof[5:])

        # Error in predict function
        with np.testing.assert_raises(AttributeError):
            lof.predict(self.fd_lof[5:])

    def _weights(
        self,
        weights: np.typing.NDArray[np.float_],
    ) -> np.typing.NDArray[np.float_]:
        return np.array([w == np.min(weights) for w in weights], dtype=float)


if __name__ == '__main__':
    unittest.main()

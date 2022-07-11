"""Test neighbors classifiers and regressors."""

import unittest

import numpy as np

from skfda.datasets import make_multimodal_samples, make_sinusoidal_process
from skfda.exploratory.outliers import LocalOutlierFactor  # Pending theory
from skfda.exploratory.stats import mean
from skfda.misc.metrics import PairwiseMetric, l2_distance
from skfda.ml.classification import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from skfda.ml.clustering import NearestNeighbors
from skfda.ml.regression import KNeighborsRegressor, RadiusNeighborsRegressor
from skfda.representation.basis import Fourier


class TestNeighbors(unittest.TestCase):

    def setUp(self):
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

    def test_predict_classifier(self):
        """Tests predict for neighbors classifier."""
        for neigh in (
            KNeighborsClassifier(),
            RadiusNeighborsClassifier(radius=0.1),
            NearestCentroid(),
            NearestCentroid(metric=l2_distance, centroid=mean),
        ):

            neigh.fit(self.X, self.y)
            pred = neigh.predict(self.X)
            np.testing.assert_array_equal(
                pred,
                self.y,
                err_msg=f'fail in {type(neigh)}',
            )

    def test_predict_proba_classifier(self):
        """Tests predict proba for k neighbors classifier."""
        neigh = KNeighborsClassifier(metric=l2_distance)

        neigh.fit(self.X, self.y)
        probs = neigh.predict_proba(self.X)

        np.testing.assert_array_almost_equal(probs, self.probs)

    def test_predict_regressor(self):
        """Test scalar regression, predicts mode location."""
        # Dummy test, with weight = distance, only the sample with distance 0
        # will be returned, obtaining the exact location
        knnr = KNeighborsRegressor(weights='distance')
        rnnr = RadiusNeighborsRegressor(weights='distance', radius=0.1)

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

    def test_kneighbors(self):
        """Test k neighbor searches for all k-neighbors estimators."""
        nn = NearestNeighbors()
        nn.fit(self.X)

        lof = LocalOutlierFactor(n_neighbors=5)
        lof.fit(self.X)

        knn = KNeighborsClassifier()
        knn.fit(self.X, self.y)

        knnr = KNeighborsRegressor()
        knnr.fit(self.X, self.modes_location)

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
                self.assertEqual(graph[0, i] == 1.0, i in links[0])
                self.assertEqual(graph[0, i] == 0.0, i not in links[0])

    def test_radius_neighbors(self):
        """Test query with radius."""
        nn = NearestNeighbors(radius=0.1)
        nn.fit(self.X)

        knn = RadiusNeighborsClassifier(radius=0.1)
        knn.fit(self.X, self.y)

        knnr = RadiusNeighborsRegressor(radius=0.1)
        knnr.fit(self.X, self.modes_location)

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
                self.assertEqual(graph[0, i] == 1.0, i in links[0])
                self.assertEqual(graph[0, i] == 0.0, i not in links[0])

    def test_knn_functional_response(self):
        knnr = KNeighborsRegressor(n_neighbors=1)

        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.data_matrix,
            self.X.data_matrix,
        )

    def test_knn_functional_response_sklearn(self):
        # Check sklearn metric
        knnr = KNeighborsRegressor(
            n_neighbors=1,
            metric='euclidean',
            multivariate_metric=True,
        )
        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.data_matrix,
            self.X.data_matrix,
        )

    def test_knn_functional_response_precomputed(self):
        knnr = KNeighborsRegressor(
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

    def test_radius_functional_response(self):
        knnr = RadiusNeighborsRegressor(
            metric=l2_distance,
            weights='distance',
        )

        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.data_matrix, self.X.data_matrix,
        )

    def test_functional_response_custom_weights(self):

        knnr = KNeighborsRegressor(weights=self._weights, n_neighbors=5)
        response = self.X.to_basis(Fourier(domain_range=(-1, 1), n_basis=10))
        knnr.fit(self.X, response)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.coefficients, response.coefficients,
        )

    def test_functional_regression_distance_weights(self):

        knnr = KNeighborsRegressor(
            weights='distance', n_neighbors=10,
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

    def test_functional_response_basis(self):
        knnr = KNeighborsRegressor(weights='distance', n_neighbors=5)
        response = self.X.to_basis(Fourier(domain_range=(-1, 1), n_basis=10))
        knnr.fit(self.X, response)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(
            res.coefficients, response.coefficients,
        )

    def test_radius_outlier_functional_response(self):
        knnr = RadiusNeighborsRegressor(radius=0.001)
        knnr.fit(self.X[3:6], self.X[3:6])

        # No value given
        with np.testing.assert_raises(ValueError):
            knnr.predict(self.X[:10])

        # Test response
        knnr = RadiusNeighborsRegressor(
            radius=0.001, outlier_response=self.X[0],
        )
        knnr.fit(self.X[:6], self.X[:6])

        res = knnr.predict(self.X[:7])
        np.testing.assert_array_almost_equal(
            self.X[0].data_matrix, res[6].data_matrix,
        )

    def test_nearest_centroids_exceptions(self):

        # Test more than one class
        nn = NearestCentroid()
        with np.testing.assert_raises(ValueError):
            nn.fit(self.X[:3], 3 * [0])

        # Precomputed not supported
        nn = NearestCentroid(metric='precomputed')
        with np.testing.assert_raises(ValueError):
            nn.fit(self.X[:3], 3 * [0])

    def test_functional_regressor_exceptions(self):

        knnr = RadiusNeighborsRegressor()

        with np.testing.assert_raises(ValueError):
            knnr.fit(self.X[:3], self.X[:4])

    def test_search_neighbors_precomputed(self):
        d = PairwiseMetric(l2_distance)
        distances = d(self.X[:4], self.X[:4])

        nn = NearestNeighbors(metric='precomputed', n_neighbors=2)
        nn.fit(distances, self.y[:4])

        _, neighbors = nn.kneighbors(distances)

        np.testing.assert_array_almost_equal(
            neighbors,
            np.array([[0, 3], [1, 2], [2, 1], [3, 0]]),
        )

    def test_search_neighbors_sklearn(self):

        nn = NearestNeighbors(
            metric='euclidean',
            multivariate_metric=True,
            n_neighbors=2,
        )
        nn.fit(self.X[:4], self.y[:4])

        _, neighbors = nn.kneighbors(self.X[:4])

        np.testing.assert_array_almost_equal(
            neighbors,
            np.array([[0, 3], [1, 2], [2, 1], [3, 0]]),
        )

    def test_score_scalar_response(self):

        neigh = KNeighborsRegressor()

        neigh.fit(self.X, self.modes_location)
        r = neigh.score(self.X, self.modes_location)
        np.testing.assert_almost_equal(r, 0.9975889963743335)

    def test_score_functional_response(self):

        neigh = KNeighborsRegressor()

        y = 5 * self.X + 1
        neigh.fit(self.X, y)
        r = neigh.score(self.X, y)
        np.testing.assert_almost_equal(r, 0.962651178452408)

        # Weighted case and basis form
        y = y.to_basis(Fourier(domain_range=y.domain_range[0], n_basis=5))
        neigh.fit(self.X, y)

        r = neigh.score(
            self.X[:7],
            y[:7],
            sample_weight=4 * [1.0 / 5] + 3 * [1.0 / 15],
        )
        np.testing.assert_almost_equal(r, 0.9982527586114364)

    def test_score_functional_response_exceptions(self):
        neigh = RadiusNeighborsRegressor()
        neigh.fit(self.X, self.X)

        with np.testing.assert_raises(ValueError):
            neigh.score(self.X, self.X, sample_weight=[1, 2, 3])

    def test_multivariate_response_score(self):

        neigh = RadiusNeighborsRegressor()
        y = make_multimodal_samples(n_samples=5, dim_domain=2, random_state=0)
        neigh.fit(self.X[:5], y)

        # It is not supported the multivariate score by the moment
        with np.testing.assert_raises(ValueError):
            neigh.score(self.X[:5], y)

    def test_lof_fit_predict(self):
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

        # With multivariate sklearn
        lof4 = LocalOutlierFactor(metric='euclidean', multivariate_metric=True)
        res4 = lof4.fit_predict(self.fd_lof)
        np.testing.assert_array_equal(expected, res4)

        # Other way of call fit_predict, undocumented in sklearn
        lof5 = LocalOutlierFactor(novelty=True)
        res5 = lof5.fit(self.fd_lof).predict()
        np.testing.assert_array_equal(expected, res5)

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

    def test_lof_decision_function(self):
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

    def test_lof_exceptions(self):
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

    def _weights(self, weights_):
        return np.array([w == 0 for w in weights_], dtype=float)


if __name__ == '__main__':
    unittest.main()

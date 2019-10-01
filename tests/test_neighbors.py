"""Test neighbors classifiers and regressors"""

import unittest

import numpy as np
from skfda.datasets import make_multimodal_samples
from skfda.exploratory.stats import mean as l2_mean
from skfda.misc.metrics import lp_distance, pairwise_distance
from skfda.ml.classification import (KNeighborsClassifier,
                                     RadiusNeighborsClassifier,
                                     NearestCentroids)
from skfda.ml.clustering import NearestNeighbors
from skfda.ml.regression import KNeighborsRegressor, RadiusNeighborsRegressor
from skfda.representation.basis import Fourier


class TestNeighbors(unittest.TestCase):

    def setUp(self):
        """Creates test data"""
        random_state = np.random.RandomState(0)
        modes_location = np.concatenate(
            (random_state.normal(-.3, .04, size=15),
             random_state.normal(.3, .04, size=15)))

        idx = np.arange(30)
        random_state.shuffle(idx)

        modes_location = modes_location[idx]
        self.modes_location = modes_location
        self.y = np.array(15 * [0] + 15 * [1])[idx]

        self.X = make_multimodal_samples(n_samples=30,
                                         modes_location=modes_location,
                                         noise=.05,
                                         random_state=random_state)
        self.X2 = make_multimodal_samples(n_samples=30,
                                          modes_location=modes_location,
                                          noise=.05,
                                          random_state=1)

        self.probs = np.array(15 * [[1., 0.]] + 15 * [[0., 1.]])[idx]

    def test_predict_classifier(self):
        """Tests predict for neighbors classifier"""

        for neigh in (KNeighborsClassifier(),
                      RadiusNeighborsClassifier(radius=.1),
                      NearestCentroids(),
                      NearestCentroids(metric=lp_distance, mean=l2_mean)):

            neigh.fit(self.X, self.y)
            pred = neigh.predict(self.X)
            np.testing.assert_array_equal(pred, self.y,
                                          err_msg=f"fail in {type(neigh)}")

    def test_predict_proba_classifier(self):
        """Tests predict proba for k neighbors classifier"""

        neigh = KNeighborsClassifier(metric=lp_distance)

        neigh.fit(self.X, self.y)
        probs = neigh.predict_proba(self.X)

        np.testing.assert_array_almost_equal(probs, self.probs)

    def test_predict_regressor(self):
        """Test scalar regression, predics mode location"""

        # Dummy test, with weight = distance, only the sample with distance 0
        # will be returned, obtaining the exact location
        knnr = KNeighborsRegressor(weights='distance')
        rnnr = RadiusNeighborsRegressor(weights='distance', radius=.1)

        knnr.fit(self.X, self.modes_location)
        rnnr.fit(self.X, self.modes_location)

        np.testing.assert_array_almost_equal(knnr.predict(self.X),
                                             self.modes_location)
        np.testing.assert_array_almost_equal(rnnr.predict(self.X),
                                             self.modes_location)

    def test_kneighbors(self):
        """Test k neighbor searches for all k-neighbors estimators"""

        nn = NearestNeighbors()
        nn.fit(self.X)

        knn = KNeighborsClassifier()
        knn.fit(self.X, self.y)

        knnr = KNeighborsRegressor()
        knnr.fit(self.X, self.modes_location)

        for neigh in [nn, knn, knnr]:

            dist, links = neigh.kneighbors(self.X[:4])

            np.testing.assert_array_equal(links, [[0, 7, 21, 23, 15],
                                                  [1, 12, 19, 18, 17],
                                                  [2, 17, 22, 27, 26],
                                                  [3, 4, 9, 5, 25]])

            dist_kneigh = lp_distance(self.X[0], self.X[7])

            np.testing.assert_array_almost_equal(dist[0, 1], dist_kneigh)

            graph = neigh.kneighbors_graph(self.X[:4])

            for i in range(30):
                self.assertEqual(graph[0, i] == 1.0, i in links[0])
                self.assertEqual(graph[0, i] == 0.0, i not in links[0])

    def test_radius_neighbors(self):
        """Test query with radius"""
        nn = NearestNeighbors(radius=.1)
        nn.fit(self.X)

        knn = RadiusNeighborsClassifier(radius=.1)
        knn.fit(self.X, self.y)

        knnr = RadiusNeighborsRegressor(radius=.1)
        knnr.fit(self.X, self.modes_location)

        for neigh in [nn, knn, knnr]:

            dist, links = neigh.radius_neighbors(self.X[:4])

            np.testing.assert_array_equal(links[0], np.array([0, 7]))
            np.testing.assert_array_equal(links[1], np.array([1]))
            np.testing.assert_array_equal(links[2], np.array([2, 17, 22, 27]))
            np.testing.assert_array_equal(links[3], np.array([3, 4, 9]))

            dist_kneigh = lp_distance(self.X[0], self.X[7])

            np.testing.assert_array_almost_equal(dist[0][1], dist_kneigh)

            graph = neigh.radius_neighbors_graph(self.X[:4])

            for i in range(30):
                self.assertEqual(graph[0, i] == 1.0, i in links[0])
                self.assertEqual(graph[0, i] == 0.0, i not in links[0])

    def test_knn_functional_response(self):
        knnr = KNeighborsRegressor(n_neighbors=1)

        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(res.data_matrix,
                                             self.X.data_matrix)

    def test_knn_functional_response_sklearn(self):
        # Check sklearn metric
        knnr = KNeighborsRegressor(n_neighbors=1, metric='euclidean',
                                   multivariate_metric=True)
        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(res.data_matrix,
                                             self.X.data_matrix)

    def test_knn_functional_response_precomputed(self):
        knnr = KNeighborsRegressor(n_neighbors=4, weights='distance',
                                   metric='precomputed')
        d = pairwise_distance(lp_distance)
        distances = d(self.X[:4], self.X[:4])

        knnr.fit(distances, self.X[:4])

        res = knnr.predict(distances)
        np.testing.assert_array_almost_equal(res.data_matrix,
                                             self.X[:4].data_matrix)

    def test_radius_functional_response(self):
        knnr = RadiusNeighborsRegressor(metric=lp_distance,
                                        weights='distance',
                                        regressor=l2_mean)

        knnr.fit(self.X, self.X)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(res.data_matrix,
                                             self.X.data_matrix)

    def test_functional_response_custom_weights(self):

        def weights(weights):

            return np.array([w == 0 for w in weights], dtype=float)

        knnr = KNeighborsRegressor(weights=weights, n_neighbors=5)
        response = self.X.to_basis(Fourier(domain_range=(-1, 1), n_basis=10))
        knnr.fit(self.X, response)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(res.coefficients,
                                             response.coefficients)

    def test_functional_regression_distance_weights(self):

        knnr = KNeighborsRegressor(
            weights='distance', n_neighbors=10)
        knnr.fit(self.X[:10], self.X[:10])
        res = knnr.predict(self.X[11])

        d = pairwise_distance(lp_distance)
        distances = d(self.X[:10], self.X[11]).flatten()

        weights = 1 / distances
        weights /= weights.sum()

        response = self.X[:10].mean(weights=weights)
        np.testing.assert_array_almost_equal(res.data_matrix,
                                             response.data_matrix)

    def test_functional_response_basis(self):
        knnr = KNeighborsRegressor(weights='distance', n_neighbors=5)
        response = self.X.to_basis(Fourier(domain_range=(-1, 1), n_basis=10))
        knnr.fit(self.X, response)

        res = knnr.predict(self.X)
        np.testing.assert_array_almost_equal(res.coefficients,
                                             response.coefficients)

    def test_radius_outlier_functional_response(self):
        knnr = RadiusNeighborsRegressor(radius=0.001)
        knnr.fit(self.X[3:6], self.X[3:6])

        # No value given
        with np.testing.assert_raises(ValueError):
            knnr.predict(self.X[:10])

        # Test response
        knnr = RadiusNeighborsRegressor(radius=0.001,
                                        outlier_response=self.X[0])
        knnr.fit(self.X[:6], self.X[:6])

        res = knnr.predict(self.X[:7])
        np.testing.assert_array_almost_equal(self.X[0].data_matrix,
                                             res[6].data_matrix)

    def test_nearest_centroids_exceptions(self):

        # Test more than one class
        nn = NearestCentroids()
        with np.testing.assert_raises(ValueError):
            nn.fit(self.X[0:3], 3 * [0])

        # Precomputed not supported
        nn = NearestCentroids(metric='precomputed')
        with np.testing.assert_raises(ValueError):
            nn.fit(self.X[0:3], 3 * [0])

    def test_functional_regressor_exceptions(self):

        knnr = RadiusNeighborsRegressor()

        with np.testing.assert_raises(ValueError):
            knnr.fit(self.X[:3], self.X[:4])

    def test_search_neighbors_precomputed(self):
        d = pairwise_distance(lp_distance)
        distances = d(self.X[:4], self.X[:4])

        nn = NearestNeighbors(metric='precomputed', n_neighbors=2)
        nn.fit(distances, self.y[:4])

        _, neighbors = nn.kneighbors(distances)

        result = np.array([[0, 3], [1, 2], [2, 1], [3, 0]])
        np.testing.assert_array_almost_equal(neighbors, result)

    def test_search_neighbors_sklearn(self):

        nn = NearestNeighbors(metric='euclidean', multivariate_metric=True,
                              n_neighbors=2)
        nn.fit(self.X[:4], self.y[:4])

        _, neighbors = nn.kneighbors(self.X[:4])

        result = np.array([[0, 3], [1, 2], [2, 1], [3, 0]])
        np.testing.assert_array_almost_equal(neighbors, result)

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

        r = neigh.score(self.X[:7], y[:7],
                        sample_weight=4 * [1. / 5] + 3 * [1. / 15])
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


if __name__ == '__main__':
    print()
    unittest.main()

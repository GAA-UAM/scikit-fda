"""Test neighbors classifiers and regressors"""

import unittest

import numpy as np
from skfda.datasets import make_multimodal_samples

from skfda.ml.classification import (KNeighborsClassifier,
                                     RadiusNeighborsClassifier,
                                     NearestCentroids,
                                     NearestNeighbors)

from skfda.ml.regression import (KNeighborsScalarRegressor,
                                 RadiusNeighborsScalarRegressor)
from skfda.misc.metrics import lp_distance, lp_distance

class TestNeighbors(unittest.TestCase):

    def setUp(self):
        """Creates test data"""
        random_state = np.random.RandomState(0)
        modes_location = np.concatenate((random_state.normal(-.3, .04, size=15),
                                         random_state.normal(.3, .04, size=15)))

        idx = np.arange(30)
        random_state.shuffle(idx)


        modes_location = modes_location[idx]
        self.modes_location = modes_location
        self.y = np.array(15*[0] + 15*[1])[idx]

        self.X = make_multimodal_samples(n_samples=30,
                                         modes_location=modes_location,
                                         noise=.05,
                                         random_state=random_state)

        self.probs = np.array(15*[[1., 0.]] + 15*[[0., 1.]])[idx]

    def test_predict_classifier(self):
        """Tests predict for neighbors classifier"""

        for neigh in (KNeighborsClassifier(),
                      RadiusNeighborsClassifier(radius=.1),
                      NearestCentroids()):

            neigh.fit(self.X, self.y)
            pred = neigh.predict(self.X)
            np.testing.assert_array_equal(pred, self.y,
                                          err_msg=f"fail in {type(neigh)}")

    def test_predict_proba_classifier(self):
        """Tests predict proba for k neighbors classifier"""

        neigh = KNeighborsClassifier()

        neigh.fit(self.X, self.y)
        probs = neigh.predict_proba(self.X)

        np.testing.assert_array_almost_equal(probs, self.probs)

    def test_predict_regressor(self):
        """Test scalar regression, predics mode location"""

        #Dummy test, with weight = distance, only the sample with distance 0
        # will be returned, obtaining the exact location
        knnr = KNeighborsScalarRegressor(weights='distance')
        rnnr = RadiusNeighborsScalarRegressor(weights='distance', radius=.1)


        knnr.fit(self.X, self.modes_location)
        rnnr.fit(self.X, self.modes_location)

        np.testing.assert_array_almost_equal(knnr.predict(self.X),
                                             self.modes_location)
        np.testing.assert_array_almost_equal(rnnr.predict(self.X),
                                             self.modes_location)


    def test_kneighbors(self):

        nn = NearestNeighbors()
        nn.fit(self.X)

        knn = KNeighborsClassifier()
        knn.fit(self.X, self.y)

        knnr = KNeighborsScalarRegressor()
        knnr.fit(self.X, self.modes_location)

        for neigh in [nn, knn, knnr]:

            dist, links = neigh.kneighbors(self.X[:4])

            np.testing.assert_array_equal(links, [[ 0,  7, 21, 23, 15],
                                                  [ 1, 12, 19, 18, 17],
                                                  [ 2, 17, 22, 27, 26],
                                                  [ 3,  4,  9,  5, 25]])

            dist_kneigh = lp_distance(self.X[0], self.X[7])

            np.testing.assert_array_almost_equal(dist[0,1], dist_kneigh)

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

        knnr = RadiusNeighborsScalarRegressor(radius=.1)
        knnr.fit(self.X, self.modes_location)

        for neigh in [nn, knn, knnr]:

            dist, links = neigh.radius_neighbors(self.X[:4])

            np.testing.assert_array_equal(links[0], np.array([0, 7]))
            np.testing.assert_array_equal(links[1], np.array([1]))
            np.testing.assert_array_equal(links[2], np.array([ 2, 17, 22, 27]))
            np.testing.assert_array_equal(links[3], np.array([3, 4, 9]))

            dist_kneigh = lp_distance(self.X[0], self.X[7])

            np.testing.assert_array_almost_equal(dist[0][1], dist_kneigh)

            graph = neigh.radius_neighbors_graph(self.X[:4])

            for i in range(30):
                self.assertEqual(graph[0, i] == 1.0, i in links[0])
                self.assertEqual(graph[0, i] == 0.0, i not in links[0])

if __name__ == '__main__':
    print()
    unittest.main()

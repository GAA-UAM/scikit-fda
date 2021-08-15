import unittest

import numpy as np
from scipy.integrate import cumtrapz

from skfda.datasets import make_gaussian, make_gaussian_process
from skfda.misc.covariances import Gaussian
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import TikhonovRegularization
from skfda.ml.regression import HistoricalLinearRegression, LinearRegression
from skfda.representation.basis import BSpline, FDataBasis, Fourier, Monomial
from skfda.representation.grid import FDataGrid


class TestScalarLinearRegression(unittest.TestCase):

    def test_regression_single_explanatory(self):

        x_basis = Monomial(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = Fourier(n_basis=5)
        beta_fd = FDataBasis(beta_basis, [1, 1, 1, 1, 1])
        y = [0.9999999999999993,
             0.162381381441085,
             0.08527083481359901,
             0.08519946930844623,
             0.09532291032042489,
             0.10550022969639987,
             0.11382675064746171]

        scalar = LinearRegression(coef_basis=[beta_basis])
        scalar.fit(x_fd, y)
        np.testing.assert_allclose(scalar.coef_[0].coefficients,
                                   beta_fd.coefficients)
        np.testing.assert_allclose(scalar.intercept_,
                                   0.0, atol=1e-6)

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

        scalar = LinearRegression(coef_basis=[beta_basis],
                                  fit_intercept=False)
        scalar.fit(x_fd, y)
        np.testing.assert_allclose(scalar.coef_[0].coefficients,
                                   beta_fd.coefficients)
        np.testing.assert_equal(scalar.intercept_,
                                0.0)

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

    def test_regression_multiple_explanatory(self):
        y = [1, 2, 3, 4, 5, 6, 7]

        X = FDataBasis(Monomial(n_basis=7), np.identity(7))

        beta1 = BSpline(domain_range=(0, 1), n_basis=5)

        scalar = LinearRegression(coef_basis=[beta1])

        scalar.fit(X, y)

        np.testing.assert_allclose(scalar.intercept_.round(4),
                                   np.array([32.65]), rtol=1e-3)

        np.testing.assert_allclose(
            scalar.coef_[0].coefficients.round(4),
            np.array([[-28.6443,
                       80.3996,
                       -188.587,
                       236.5832,
                       -481.3449]]), rtol=1e-3)

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(y_pred, y, atol=0.01)

    def test_regression_mixed(self):

        multivariate = np.array([[0, 0], [2, 7], [1, 7], [3, 9],
                                 [4, 16], [2, 14], [3, 5]])

        X = [multivariate,
             FDataBasis(Monomial(n_basis=3), [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                              [1, 0, 1], [1, 0, 0], [0, 1, 0],
                                              [0, 0, 1]])]

        # y = 2 + sum([3, 1] * array) + int(3 * function)
        intercept = 2
        coefs_multivariate = np.array([3, 1])
        coefs_functions = FDataBasis(
            Monomial(n_basis=3), [[3, 0, 0]])
        y_integral = np.array([3, 3 / 2, 1, 4, 3, 3 / 2, 1])
        y_sum = multivariate @ coefs_multivariate
        y = 2 + y_sum + y_integral

        scalar = LinearRegression()
        scalar.fit(X, y)

        np.testing.assert_allclose(scalar.intercept_,
                                   intercept, atol=0.01)

        np.testing.assert_allclose(
            scalar.coef_[0],
            coefs_multivariate, atol=0.01)

        np.testing.assert_allclose(
            scalar.coef_[1].coefficients,
            coefs_functions.coefficients, atol=0.01)

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(y_pred, y, atol=0.01)

    def test_regression_mixed_regularization(self):

        multivariate = np.array([[0, 0], [2, 7], [1, 7], [3, 9],
                                 [4, 16], [2, 14], [3, 5]])

        X = [multivariate,
             FDataBasis(Monomial(n_basis=3), [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                              [1, 0, 1], [1, 0, 0], [0, 1, 0],
                                              [0, 0, 1]])]

        # y = 2 + sum([3, 1] * array) + int(3 * function)
        intercept = 2
        coefs_multivariate = np.array([3, 1])
        y_integral = np.array([3, 3 / 2, 1, 4, 3, 3 / 2, 1])
        y_sum = multivariate @ coefs_multivariate
        y = 2 + y_sum + y_integral

        scalar = LinearRegression(
            regularization=[TikhonovRegularization(lambda x: x),
                            TikhonovRegularization(
                                LinearDifferentialOperator(2))])
        scalar.fit(X, y)

        np.testing.assert_allclose(scalar.intercept_,
                                   intercept, atol=0.01)

        np.testing.assert_allclose(
            scalar.coef_[0],
            [2.536739, 1.072186], atol=0.01)

        np.testing.assert_allclose(
            scalar.coef_[1].coefficients,
            [[2.125676, 2.450782, 5.808745e-4]], atol=0.01)

        y_pred = scalar.predict(X)
        np.testing.assert_allclose(
            y_pred,
            [5.349035, 16.456464, 13.361185, 23.930295,
                32.650965, 23.961766, 16.29029],
            atol=0.01)

    def test_regression_regularization(self):

        x_basis = Monomial(n_basis=7)
        x_fd = FDataBasis(x_basis, np.identity(7))

        beta_basis = Fourier(n_basis=5)
        beta_fd = FDataBasis(beta_basis, [1.0403, 0, 0, 0, 0])
        y = [1.0000684777229512,
             0.1623672257830915,
             0.08521053851548224,
             0.08514200869281137,
             0.09529138749665378,
             0.10549625973303875,
             0.11384314859153018]

        y_pred_compare = [0.890341,
                          0.370162,
                          0.196773,
                          0.110079,
                          0.058063,
                          0.023385,
                          -0.001384]

        scalar = LinearRegression(
            coef_basis=[beta_basis],
            regularization=TikhonovRegularization(
                LinearDifferentialOperator(2)))
        scalar.fit(x_fd, y)
        np.testing.assert_allclose(scalar.coef_[0].coefficients,
                                   beta_fd.coefficients, atol=1e-3)
        np.testing.assert_allclose(scalar.intercept_,
                                   -0.15, atol=1e-4)

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y_pred_compare, atol=1e-4)

        x_basis = Monomial(n_basis=3)
        x_fd = FDataBasis(x_basis, [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [2, 0, 1]])

        beta_fd = FDataBasis(x_basis, [3, 2, 1])
        y = [1 + 13 / 3, 1 + 29 / 12, 1 + 17 / 10, 1 + 311 / 30]

        # Non regularized
        scalar = LinearRegression()
        scalar.fit(x_fd, y)
        np.testing.assert_allclose(scalar.coef_[0].coefficients,
                                   beta_fd.coefficients)
        np.testing.assert_allclose(scalar.intercept_,
                                   1)

        y_pred = scalar.predict(x_fd)
        np.testing.assert_allclose(y_pred, y)

        # Regularized
        beta_fd_reg = FDataBasis(x_basis, [2.812, 3.043, 0])
        y_reg = [5.333, 3.419, 2.697, 11.366]

        scalar_reg = LinearRegression(
            regularization=TikhonovRegularization(
                LinearDifferentialOperator(2)))
        scalar_reg.fit(x_fd, y)
        np.testing.assert_allclose(scalar_reg.coef_[0].coefficients,
                                   beta_fd_reg.coefficients, atol=0.001)
        np.testing.assert_allclose(scalar_reg.intercept_,
                                   0.998, atol=0.001)

        y_pred = scalar_reg.predict(x_fd)
        np.testing.assert_allclose(y_pred, y_reg, atol=0.001)

    def test_error_X_not_FData(self):
        """Tests that at least one of the explanatory variables
        is an FData object. """

        x_fd = np.identity(7)
        y = np.zeros(7)

        scalar = LinearRegression(coef_basis=[Fourier(n_basis=5)])

        with np.testing.assert_warns(UserWarning):
            scalar.fit([x_fd], y)

    def test_error_y_is_FData(self):
        """Tests that none of the explained variables is an FData object
        """
        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = list(FDataBasis(Monomial(n_basis=7), np.identity(7)))

        scalar = LinearRegression(coef_basis=[Fourier(n_basis=5)])

        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

    def test_error_X_beta_len_distinct(self):
        """ Test that the number of beta bases and explanatory variables
        are not different """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        beta = Fourier(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd, x_fd], y)

        scalar = LinearRegression(coef_basis=[beta, beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

    def test_error_y_X_samples_different(self):
        """ Test that the number of response samples and explanatory samples
        are not different """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(8)]
        beta = Fourier(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

        x_fd = FDataBasis(Monomial(n_basis=8), np.identity(8))
        y = [1 for _ in range(7)]
        beta = Fourier(n_basis=5)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y)

    def test_error_beta_not_basis(self):
        """ Test that all beta are Basis objects. """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        beta = FDataBasis(Monomial(n_basis=7), np.identity(7))

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(TypeError):
            scalar.fit([x_fd], y)

    def test_error_weights_lenght(self):
        """ Test that the number of weights is equal to the
        number of samples """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        weights = [1 for _ in range(8)]
        beta = Monomial(n_basis=7)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y, weights)

    def test_error_weights_negative(self):
        """ Test that none of the weights are negative. """

        x_fd = FDataBasis(Monomial(n_basis=7), np.identity(7))
        y = [1 for _ in range(7)]
        weights = [-1 for _ in range(7)]
        beta = Monomial(n_basis=7)

        scalar = LinearRegression(coef_basis=[beta])
        with np.testing.assert_raises(ValueError):
            scalar.fit([x_fd], y, weights)


class TestHistoricalLinearRegression(unittest.TestCase):
    """Tests for historical linear regression."""

    def setUp(self) -> None:
        """Generate data according to the model."""
        self.random = np.random.RandomState(1)

        self.n_samples = 50
        self.n_features = 20
        self.intercept = make_gaussian_process(
            n_samples=1,
            n_features=self.n_features,
            cov=Gaussian(length_scale=0.4),
            random_state=self.random,
        )

        np.testing.assert_almost_equal(
            self.intercept.data_matrix[..., 0],
            np.array([[
                -0.44419728, -0.56909477, -0.68783434, -0.80186766, -0.91540068,
                -1.03397827, -1.16239266, -1.30246822, -1.45134619, -1.60079727,
                -1.73785278, -1.84672707, -1.9116814, -1.92023053, -1.86597929,
                -1.75042757, -1.58329321, -1.38122881, -1.16517441, -0.95690171,
            ]])
        )

        self.X = make_gaussian_process(
            n_samples=self.n_samples,
            n_features=self.n_features,
            cov=Gaussian(length_scale=0.4),
            random_state=self.random,
        )
        self.coefficients = make_gaussian(
            n_samples=1,
            grid_points=[np.linspace(0, 1, self.n_features)] * 2,
            cov=Gaussian(length_scale=1),
            random_state=self.random,
        )

        test = self.random.random(size=10)
        np.testing.assert_almost_equal(
            test,
            np.array([0.39564827, 0.77171848, 0.25120318, 0.65576077,
                      0.96039715, 0.00355476,
                      0.72336104, 0.10899503, 0.54378749, 0.62691602])
        )

        np.testing.assert_almost_equal(
            self.coefficients.data_matrix[0, ..., 0],
            np.array([
                [4.93663563e-01, 4.78010146e-01, 4.63788522e-01,
                    4.50928237e-01, 4.39314629e-01, 4.28798261e-01,
                    4.19192312e-01, 4.10280148e-01, 4.01813089e-01,
                    3.93515764e-01, 3.85089077e-01, 3.76209900e-01,
                    3.66535004e-01, 3.55705856e-01, 3.43348716e-01,
                    3.29083247e-01, 3.12524641e-01, 2.93292495e-01,
                    2.71019709e-01, 2.45359953e-01],
                [5.31535010e-01, 5.17720140e-01, 5.04907227e-01,
                    4.93003500e-01, 4.81882668e-01, 4.71390399e-01,
                    4.61342759e-01, 4.51534164e-01, 4.41733815e-01,
                    4.31691405e-01, 4.21136776e-01, 4.09782265e-01,
                    3.97324811e-01, 3.83448921e-01, 3.67826456e-01,
                    3.50123751e-01, 3.30004632e-01, 3.07136237e-01,
                    2.81197769e-01, 2.51887492e-01],
                [5.61268157e-01, 5.49413977e-01, 5.38156769e-01,
                    5.27383042e-01, 5.16954232e-01, 5.06709032e-01,
                    4.96466435e-01, 4.86028192e-01, 4.75180833e-01,
                    4.63695845e-01, 4.51331706e-01, 4.37835185e-01,
                    4.22940282e-01, 4.06372937e-01, 3.87849145e-01,
                    3.67081529e-01, 3.43779743e-01, 3.17658698e-01,
                    2.88441954e-01, 2.55873009e-01],
                [5.82214282e-01, 5.72443820e-01, 5.62899224e-01,
                    5.53445283e-01, 5.43927740e-01, 5.34178785e-01,
                    5.24016376e-01, 5.13250432e-01, 5.01678997e-01,
                    4.89095545e-01, 4.75283435e-01, 4.60020810e-01,
                    4.43078173e-01, 4.24219504e-01, 4.03203927e-01,
                    3.79785535e-01, 3.53720408e-01, 3.24767458e-01,
                    2.92695286e-01, 2.57289404e-01],
                [5.93890073e-01, 5.86327635e-01, 5.78658665e-01,
                    5.70723768e-01, 5.62353524e-01, 5.53368146e-01,
                    5.43585005e-01, 5.32816681e-01, 5.20874443e-01,
                    5.07566358e-01, 4.92700607e-01, 4.76083195e-01,
                    4.57516183e-01, 4.36799910e-01, 4.13732435e-01,
                    3.88109758e-01, 3.59729121e-01, 3.28392005e-01,
                    2.93909255e-01, 2.56107758e-01],
                [5.95979512e-01, 5.90749572e-01, 5.85121442e-01,
                    5.78911820e-01, 5.71931019e-01, 5.63990198e-01,
                    5.54899635e-01, 5.44473706e-01, 5.32530642e-01,
                    5.18894359e-01, 5.03391058e-01, 4.85850840e-01,
                    4.66105368e-01, 4.43987537e-01, 4.19330529e-01,
                    3.91967867e-01, 3.61737266e-01, 3.28478056e-01,
                    2.92041731e-01, 2.52293795e-01],
                [5.88331159e-01, 5.85553166e-01, 5.82129597e-01,
                    5.77850567e-01, 5.72507449e-01, 5.65895238e-01,
                    5.57817817e-01, 5.48087161e-01, 5.36525668e-01,
                    5.22967184e-01, 5.07253137e-01, 4.89235017e-01,
                    4.68768294e-01, 4.45715658e-01, 4.19940279e-01,
                    3.91310913e-01, 3.59699908e-01, 3.24984279e-01,
                    2.87052173e-01, 2.45804580e-01],
                [5.70945355e-01, 5.70732608e-01, 5.69672008e-01,
                    5.67526575e-01, 5.64064529e-01, 5.59064647e-01,
                    5.52320762e-01, 5.43637511e-01, 5.32839843e-01,
                    5.19766631e-01, 5.04271253e-01, 4.86220453e-01,
                    4.65490596e-01, 4.41967813e-01, 4.15543684e-01,
                    3.86116938e-01, 3.53590594e-01, 3.17876843e-01,
                    2.78895404e-01, 2.36583668e-01],
                [5.43965927e-01, 5.46421980e-01, 5.47873568e-01,
                    5.48054218e-01, 5.46709103e-01, 5.43597425e-01,
                    5.38497473e-01, 5.31207983e-01, 5.21548846e-01,
                    5.09360115e-01, 4.94503082e-01, 4.76855471e-01,
                    4.56310745e-01, 4.32772489e-01, 4.06157107e-01,
                    3.76387702e-01, 3.43396592e-01, 3.07124177e-01,
                    2.67522260e-01, 2.24557489e-01],
                [5.07663154e-01, 5.12878825e-01, 5.16978529e-01,
                    5.19665633e-01, 5.20659115e-01, 5.19696420e-01,
                    5.16538243e-01, 5.10971363e-01, 5.02808400e-01,
                    4.91887178e-01, 4.78071524e-01, 4.61245444e-01,
                    4.41312805e-01, 4.18194086e-01, 3.91822409e-01,
                    3.62143509e-01, 3.29112722e-01, 2.92696484e-01,
                    2.52874094e-01, 2.09640776e-01],
                [4.62417185e-01, 4.70469373e-01, 4.77336768e-01,
                    4.82693233e-01, 4.86227716e-01, 4.87654065e-01,
                    4.86713876e-01, 4.83176946e-01, 4.76844995e-01,
                    4.67549900e-01, 4.55151684e-01, 4.39538412e-01,
                    4.20619758e-01, 3.98325863e-01, 3.72603904e-01,
                    3.43417253e-01, 3.10739459e-01, 2.74560952e-01,
                    2.34883811e-01, 1.91729340e-01],
                [4.08701273e-01, 4.19648971e-01, 4.29385830e-01,
                    4.37551868e-01, 4.43806630e-01, 4.47838242e-01,
                    4.49364099e-01, 4.48135656e-01, 4.43940027e-01,
                    4.36597506e-01, 4.25962501e-01, 4.11919873e-01,
                    3.94380971e-01, 3.73282426e-01, 3.48580043e-01,
                    3.20249019e-01, 2.88280822e-01, 2.52682487e-01,
                    2.13477046e-01, 1.70708405e-01],
                [3.47064514e-01, 3.60948529e-01, 3.73633162e-01,
                    3.84724533e-01, 3.93852096e-01, 4.00674175e-01,
                    4.04883237e-01, 4.06209018e-01, 4.04418871e-01,
                    3.99319584e-01, 3.90754458e-01, 3.78600793e-01,
                    3.62768890e-01, 3.43194737e-01, 3.19839536e-01,
                    2.92687753e-01, 2.61741537e-01, 2.27022588e-01,
                    1.88571803e-01, 1.46452555e-01],
                [2.78115147e-01, 2.94954322e-01, 3.10640282e-01,
                    3.24745523e-01, 3.36866989e-01, 3.46632733e-01,
                    3.53706943e-01, 3.57793827e-01, 3.58640294e-01,
                    3.56033094e-01, 3.49802162e-01, 3.39813697e-01,
                    3.25970385e-01, 3.08204815e-01, 2.86481038e-01,
                    2.60784909e-01, 2.31127858e-01, 1.97542609e-01,
                    1.60085349e-01, 1.18834923e-01],
                [2.02505870e-01, 2.22295616e-01, 2.41010390e-01,
                    2.58188053e-01, 2.73391887e-01, 2.86217460e-01,
                    2.96299516e-01, 3.03314572e-01, 3.06984085e-01,
                    3.07075180e-01, 3.03397257e-01, 2.95802774e-01,
                    2.84183751e-01, 2.68466045e-01, 2.48607812e-01,
                    2.24598277e-01, 1.96451275e-01, 1.64208453e-01,
                    1.27936293e-01, 8.77289872e-02],
                [1.20919279e-01, 1.43631393e-01, 1.65374150e-01,
                    1.85650125e-01, 2.03989120e-01, 2.19952959e-01,
                    2.33145175e-01, 2.43211654e-01, 2.49848567e-01,
                    2.52796354e-01, 2.51843737e-01, 2.46825811e-01,
                    2.37618011e-01, 2.24136868e-01, 2.06333636e-01,
                    1.84192432e-01, 1.57730629e-01, 1.26992827e-01,
                    9.20536943e-02, 5.30199705e-02],
                [3.40581354e-02, 5.96384941e-02, 8.43785197e-02,
                    1.07746837e-01, 1.29237742e-01, 1.48379055e-01,
                    1.64741413e-01, 1.77940510e-01, 1.87641263e-01,
                    1.93558170e-01, 1.95456087e-01, 1.93147244e-01,
                    1.86491192e-01, 1.75388591e-01, 1.59780096e-01,
                    1.39645886e-01, 1.14997927e-01, 8.58838546e-02,
                    5.23841084e-02, 1.46146099e-02],
                [-5.73636876e-02, -2.89952105e-02, -1.31777107e-03,
                    2.51021740e-02, 4.97245150e-02, 7.20431501e-02,
                    9.15945992e-02, 1.07961645e-01, 1.20778708e-01,
                    1.29730922e-01, 1.34556570e-01, 1.35044553e-01,
                    1.31031788e-01, 1.22403703e-01, 1.09086975e-01,
                    9.10515217e-02, 6.83044338e-02, 4.08921161e-02,
                    8.89870469e-03, -2.75528458e-02],
                [-1.52626004e-01, -1.21575121e-01, -9.10514034e-02,
                    -6.16531260e-02, -3.39560582e-02, -8.50001123e-03,
                    1.42172550e-02, 3.37460153e-02, 4.96871596e-02,
                    6.16958027e-02, 6.94813559e-02, 7.28063083e-02,
                    7.14858605e-02, 6.53839428e-02, 5.44120992e-02,
                    3.85273779e-02, 1.77287890e-02, -7.93995107e-03,
                    -3.83950945e-02, -7.35085564e-02],
                [-2.51007663e-01, -2.17405703e-01, -1.84154945e-01,
                    -1.51885874e-01, -1.21206316e-01, -9.26905561e-02,
                    -6.68699310e-02, -4.42281214e-02, -2.51976265e-02,
                    -1.01544938e-02, 5.80976532e-04, 6.74184827e-03,
                    8.11965928e-03, 4.55584966e-03, -4.05706483e-03,
                    -1.77762916e-02, -3.66129300e-02, -6.05304857e-02,
                    -8.94449365e-02, -1.23227367e-01],
            ]),
        )

        self.create_model()

    def create_model(self) -> None:
        """Create a functional response according to historical model."""
        integral_body = (
            self.X.data_matrix[..., 0, np.newaxis]
            * self.coefficients.data_matrix[..., 0]
        )
        integral_matrix = cumtrapz(
            integral_body,
            x=self.X.grid_points[0],
            initial=0,
            axis=1,
        )
        integral = np.diagonal(integral_matrix, axis1=1, axis2=2)
        self.y = FDataGrid(self.intercept.data_matrix[..., 0] + integral)

    def test_historical(self) -> None:
        """Test historical regression with data following the model."""
        regression = HistoricalLinearRegression(n_intervals=6)
        regression.fit(self.X, self.y)
        np.testing.assert_allclose(
            regression.predict(self.X).data_matrix,
            self.y.data_matrix,
            atol=1e-1,
            rtol=0,
        )

        np.testing.assert_allclose(
            regression.intercept_.data_matrix,
            self.intercept.data_matrix,
            rtol=1e-2,
        )

        np.testing.assert_allclose(
            regression.coef_.data_matrix[0, ..., 0],
            np.triu(self.coefficients.data_matrix[0, ..., 0]),
            atol=0.2,
            rtol=0,
        )


if __name__ == '__main__':
    print()
    unittest.main()

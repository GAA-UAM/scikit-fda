"""Tests for FPCA."""
import unittest

import numpy as np
from sklearn.decomposition import PCA

from skfda import FDataBasis, FDataGrid
from skfda.datasets import fetch_weather
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import Basis, BSplineBasis, FourierBasis


class FPCATestCase(unittest.TestCase):
    """Tests for principal component analysis."""

    def test_basis_fpca_fit_exceptions(self) -> None:
        """Check that invalid arguments in fit raise exception for basis."""
        fpca = FPCA()
        with self.assertRaises(AttributeError):
            fpca.fit(None)  # type: ignore[arg-type]

        basis = FourierBasis(n_basis=1)
        # Check that if n_components is bigger than the number of samples then
        # an exception should be thrown
        fd = FDataBasis(basis, [[0.9]])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

        # Check that n_components must be smaller than the number of elements
        # of target basis
        fd = FDataBasis(basis, [[0.9], [0.7], [0.5]])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

    def test_discretized_fpca_fit_exceptions(self) -> None:
        """Check that invalid arguments in fit raise exception for grid."""
        fpca = FPCA()
        with self.assertRaises(AttributeError):
            fpca.fit(None)  # type: ignore[arg-type]

        # Check that if n_components is bigger than the number of samples then
        # an exception should be thrown
        fd = FDataGrid([[0.5], [0.1]], grid_points=[0])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

        # Check that n_components must be smaller than the number of attributes
        # in the FDataGrid object
        fd = FDataGrid([[0.9], [0.7], [0.5]], grid_points=[0])
        with self.assertRaises(AttributeError):
            fpca.fit(fd)

    def test_basis_fpca_fit_result(self) -> None:
        """Compare the components in basis against the fda package."""
        n_basis = 9
        n_components = 3

        fd_data = fetch_weather()['data'].coordinates[0]

        # Initialize basis data
        basis = FourierBasis(n_basis=n_basis, domain_range=(0, 365))
        fd_basis = fd_data.to_basis(basis)

        fpca = FPCA(
            n_components=n_components,
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
                regularization_parameter=1e5,
            ),
        )
        fpca.fit(fd_basis)

        # Results obtained using Ramsay's R package
        results = np.array([
            [  # noqa: WPS317
                0.92407552, 0.13544888, 0.35399023,
                0.00805966, -0.02148108, -0.01709549,
                -0.00208469, -0.00297439, -0.00308224,
            ],
            [  # noqa: WPS317
                -0.33314436, -0.05116842, 0.89443418,
                0.14673902, 0.21559073, 0.02046924,
                0.02203431, -0.00787185, 0.00247492,
            ],
            [  # noqa: WPS317
                -0.14241092, 0.92131899, 0.00514715,
                0.23391411, -0.19497613, 0.09800817,
                0.01754439, -0.00205874, 0.01438185,
            ],
        ])

        # Compare results obtained using this library. There are slight
        # variations due to the fact that we are in two different packages
        # If the sign of the components is not the same the component is
        # reflected.
        results *= (
            np.sign(fpca.components_.coefficients[:, 0])
            * np.sign(results[:, 0])
        )[:, np.newaxis]

        np.testing.assert_allclose(
            fpca.components_.coefficients,
            results,
            atol=1e-7,
        )

    def test_basis_fpca_transform_result(self) -> None:
        """Compare the scores in basis against the fda package."""
        n_basis = 9
        n_components = 3

        fd_data = fetch_weather()['data'].coordinates[0]

        # Initialize basis data
        basis = FourierBasis(n_basis=n_basis, domain_range=(0, 365))
        fd_basis = fd_data.to_basis(basis)

        fpca = FPCA(
            n_components=n_components,
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
                regularization_parameter=1e5,
            ),
        )
        fpca.fit(fd_basis)
        scores = fpca.transform(fd_basis)

        # Results obtained using Ramsay's R package
        results = np.array([
            [-7.68307641e1, 5.69034443e1, -1.22440149e1],
            [-9.02873996e1, 1.46262257e1, -1.78574536e1],
            [-8.21155683e1, 3.19159491e1, -2.56212328e1],
            [-1.14163637e2, 3.66425562e1, -1.00810836e1],
            [-6.97263223e1, 1.22817168e1, -2.39417618e1],
            [-6.41886364e1, -1.07261045e1, -1.10587407e1],
            [1.35824412e2, 2.03484658e1, -9.04815324e0],
            [-1.46816399e1, -2.66867491e1, -1.20233465e1],
            [1.02507511e0, -2.29840736e1, -9.06081296e0],
            [-3.62936903e1, -2.09520442e1, -1.14799951e1],
            [-4.20649313e1, -1.13618094e1, -6.24909009e0],
            [-7.38115985e1, -3.18423866e1, -1.50298626e1],
            [-6.69822456e1, -3.35518632e1, -1.25167352e1],
            [-1.03534763e2, -1.29513941e1, -1.49103879e1],
            [-1.04542036e2, -1.36794907e1, -1.41555965e1],
            [-7.35863347e0, -1.41171956e1, -2.97562788e0],
            [7.28804530e0, -5.34421830e1, -3.39823418e0],
            [5.59974094e1, -4.02154080e1, 3.78800103e-1],
            [1.80778702e2, 1.87798201e1, -1.99043247e1],
            [-3.69700617e0, -4.19441020e1, 6.45820740e0],
            [3.76527216e1, -4.23056953e1, 1.04221757e1],
            [1.23850646e2, -4.24648130e1, -2.22336786e-1],
            [-7.23588457e0, -1.20579536e1, 2.07502089e1],
            [-4.96871011e1, 8.88483448e0, 2.02882768e1],
            [-1.36726355e2, -1.86472599e1, 1.89076217e1],
            [-1.83878661e2, 4.12118550e1, 1.78960356e1],
            [-1.81568820e2, 5.20817910e1, 2.01078870e1],
            [-5.08775852e1, 1.34600555e1, 3.18602712e1],
            [-1.37633866e2, 7.50809631e1, 2.42320782e1],
            [4.98276375e1, 1.33401270e0, 3.50611066e1],
            [1.51149934e2, -5.47417776e1, 3.97592325e1],
            [1.58366096e2, -3.80762686e1, -5.62415023e0],
            [2.17139548e2, 6.34055987e1, -1.98853635e1],
            [2.33615480e2, -7.90787574e-2, 2.69069525e0],
            [3.45371437e2, 9.58703622e1, 8.47570770e0],
        ])

        # Compare results
        np.testing.assert_allclose(scores, results, atol=1e-7)

    def test_basis_fpca_noregularization_fit_result(self) -> None:
        """
        Compare the components in basis against the fda package.

        Replication code:

            ... library(fda)
            ...
            ... data("CanadianWeather")
            ...  temp = CanadianWeather$dailyAv[,,1]
            ...
            ...  basis = create.fourier.basis(c(0,365), 9)
            ...  fdata_temp = Data2fd(1:365, temp, basis)
            ...  fpca = pca.fd(fdata_temp, nharm = 3)
            ...
            ...  paste(
            ...      round(fpca$harmonics$coefs[,1], 8),
            ...      collapse=", "
            ...  ) # first component, others are analogous
            ...
            ...  fpca$varprop # explained variance ratio

        """
        n_basis = 9
        n_components = 3

        fd_data = fetch_weather()['data'].coordinates[0]

        # Initialize basis data
        basis = FourierBasis(n_basis=n_basis, domain_range=(0, 365))
        fd_basis = fd_data.to_basis(basis)

        fpca = FPCA(n_components=n_components)
        fpca.fit(fd_basis)

        # Results obtained using Ramsay's R package
        results = np.array([
            [  # noqa: WPS317
                0.92315509, 0.1395638, 0.35575705,
                0.00877893, -0.02460726, -0.02932107,
                -0.0028108, -0.00999328, -0.00966805,
            ],
            [  # noqa: WPS317
                -0.33152114, -0.04318338, 0.89258995,
                0.17111744, 0.24248046, 0.03645764,
                0.03700911, -0.02547251, 0.00929922,
            ],
            [  # noqa: WPS317
                -0.13791076, 0.91248735, -0.00643356,
                0.26200806, -0.21919224, 0.16909055,
                0.02715258, -0.00513581, 0.04751166,
            ],
        ])

        explained_variance_ratio = [0.88958975, 0.08483036, 0.01844100]

        # Compare results obtained using this library. There are slight
        # variations due to the fact that we are in two different packages
        # If the sign of the components is not the same the component is
        # reflected.
        results *= (
            np.sign(fpca.components_.coefficients[:, 0])
            * np.sign(results[:, 0])
        )[:, np.newaxis]

        np.testing.assert_allclose(
            fpca.components_.coefficients,
            results,
            atol=0.008,
        )

        np.testing.assert_allclose(
            fpca.explained_variance_ratio_,
            explained_variance_ratio,
        )

    def test_grid_fpca_fit_sklearn(self) -> None:
        """Compare the components in grid against the multivariate case."""
        n_components = 3

        fd_data = fetch_weather()['data'].coordinates[0]

        fpca = FPCA(n_components=n_components, _weights=[1] * 365)
        fpca.fit(fd_data)

        pca = PCA(n_components=n_components)
        pca.fit(fd_data.data_matrix[..., 0])

        np.testing.assert_allclose(
            fpca.components_.data_matrix[..., 0],
            pca.components_,
        )

        np.testing.assert_allclose(
            fpca.explained_variance_,
            pca.explained_variance_,
        )

    def test_grid_fpca_fit_result(self) -> None:
        """
        Compare the components in grid against the fda.usc package.

        Replication code:

            ... library(fda)
            ... library(fda.usc)
            ...
            ... data("CanadianWeather")
            ... temp = CanadianWeather$dailyAv[,,1]
            ...
            ... fdata_temp = fdata(t(temp))
            ... fpca = fdata2pc(fdata_temp, ncomp = 1)
            ...
            ... paste(
            ...     round(fpca$rotation$data[1,], 8),
            ...     collapse=", "
            ... ) # components
            ... fpca$d[1] # singular value
            ... paste(
            ...     round(fpca$x[,1], 8),
            ...     collapse=", "
            ... ) # transform

        """
        n_components = 1

        fd_data = fetch_weather()['data'].coordinates[0]

        fpca = FPCA(n_components=n_components, _weights=[1] * 365)
        fpca.fit(fd_data)

        # Results obtained using fda.usc for the first component
        results = np.array([  # noqa: WPS317
            -0.06958281, -0.07015412, -0.07095115, -0.07185632, -0.07128256,
            -0.07124209, -0.07364828, -0.07297663, -0.07235438, -0.07307498,
            -0.07293423, -0.07449293, -0.07647909, -0.07796823, -0.07582476,
            -0.07263243, -0.07241871, -0.07181360, -0.07015477, -0.07132331,
            -0.07115270, -0.07435933, -0.07602666, -0.07697830, -0.07707199,
            -0.07503802, -0.07703020, -0.07705581, -0.07633515, -0.07624817,
            -0.07631568, -0.07619913, -0.07568000, -0.07595155, -0.07506939,
            -0.07181941, -0.06907624, -0.06735476, -0.06853985, -0.06902363,
            -0.07098882, -0.07479412, -0.07425241, -0.07555835, -0.07659030,
            -0.07651853, -0.07682536, -0.07458996, -0.07631711, -0.07726509,
            -0.07641246, -0.07440660, -0.07501397, -0.07302722, -0.07045571,
            -0.06912529, -0.06792186, -0.06830739, -0.06898433, -0.07000192,
            -0.07014513, -0.06994886, -0.07115909, -0.07399900, -0.07292669,
            -0.07139879, -0.07226865, -0.07187915, -0.07122995, -0.06975022,
            -0.06800613, -0.06900793, -0.07186378, -0.07114479, -0.07015252,
            -0.06944782, -0.06829100, -0.06905348, -0.06925773, -0.06834624,
            -0.06837319, -0.06824067, -0.06644614, -0.06637313, -0.06626312,
            -0.06470209, -0.06450580, -0.06477729, -0.06411049, -0.06158499,
            -0.06305197, -0.06398006, -0.06277579, -0.06282124, -0.06317684,
            -0.06141250, -0.05961922, -0.05875443, -0.05845781, -0.05828608,
            -0.05666474, -0.05495706, -0.05446301, -0.05468254, -0.05478609,
            -0.05440798, -0.05312339, -0.05102368, -0.05160285, -0.05077954,
            -0.04979648, -0.04890853, -0.04745462, -0.04496763, -0.04487130,
            -0.04599596, -0.04688998, -0.04488872, -0.04404507, -0.04420729,
            -0.04368153, -0.04254381, -0.04117640, -0.04022811, -0.03999746,
            -0.03963634, -0.03832502, -0.03839560, -0.04015374, -0.03875440,
            -0.03777315, -0.03830728, -0.03768616, -0.03714081, -0.03781918,
            -0.03739374, -0.03659894, -0.03563342, -0.03658407, -0.03686991,
            -0.03543746, -0.03518799, -0.03361226, -0.03215340, -0.03050438,
            -0.02958411, -0.02855023, -0.02913402, -0.02992464, -0.02899548,
            -0.02891629, -0.02809554, -0.02702642, -0.02672194, -0.02678648,
            -0.02698471, -0.02628085, -0.02674285, -0.02658515, -0.02604447,
            -0.02457110, -0.02413174, -0.02342496, -0.02289800, -0.02216152,
            -0.02272283, -0.02199741, -0.02305362, -0.02371371, -0.02320865,
            -0.02234777, -0.02250180, -0.02104359, -0.02203346, -0.02052545,
            -0.01987457, -0.01947911, -0.01986949, -0.02012196, -0.01958515,
            -0.01906753, -0.01857869, -0.01874101, -0.01827973, -0.01775200,
            -0.01702056, -0.01759611, -0.01888485, -0.01988159, -0.01951675,
            -0.01872967, -0.01866667, -0.01835760, -0.01909758, -0.01859900,
            -0.01910036, -0.01930315, -0.01958856, -0.02129936, -0.02166140,
            -0.02043970, -0.02002368, -0.02058828, -0.02149915, -0.02167326,
            -0.02238569, -0.02211907, -0.02168336, -0.02124387, -0.02131655,
            -0.02130508, -0.02181227, -0.02230632, -0.02223732, -0.02282160,
            -0.02355137, -0.02275145, -0.02286893, -0.02437776, -0.02523897,
            -0.02483540, -0.02319174, -0.02335831, -0.02405789, -0.02483273,
            -0.02428119, -0.02395295, -0.02437185, -0.02476434, -0.02347973,
            -0.02385957, -0.02451257, -0.02414586, -0.02439035, -0.02357782,
            -0.02417295, -0.02504764, -0.02682569, -0.02807111, -0.02886335,
            -0.02943406, -0.02956806, -0.02893096, -0.02903812, -0.02999862,
            -0.02942100, -0.03016203, -0.03118823, -0.03076205, -0.03005985,
            -0.03079187, -0.03215188, -0.03271075, -0.03146124, -0.03040965,
            -0.03008436, -0.03085897, -0.03015341, -0.03014661, -0.03110255,
            -0.03271278, -0.03217399, -0.03317210, -0.03459221, -0.03572073,
            -0.03560707, -0.03531492, -0.03687657, -0.03800143, -0.03738080,
            -0.03729927, -0.03748666, -0.03754171, -0.03790408, -0.03963726,
            -0.03992153, -0.03812243, -0.03738440, -0.03853940, -0.03849716,
            -0.03826345, -0.03743958, -0.03808610, -0.03857622, -0.04099357,
            -0.04102509, -0.04170207, -0.04283573, -0.04320618, -0.04269438,
            -0.04467527, -0.04470603, -0.04496092, -0.04796417, -0.04796633,
            -0.04786300, -0.04883668, -0.05059390, -0.05112441, -0.04960962,
            -0.05000041, -0.04962112, -0.05087008, -0.05216710, -0.05369792,
            -0.05478139, -0.05559221, -0.05669698, -0.05654505, -0.05731113,
            -0.05783543, -0.05766056, -0.05754354, -0.05724272, -0.05831026,
            -0.05847512, -0.05804533, -0.05875046, -0.06021703, -0.06147975,
            -0.06213918, -0.06458050, -0.06500849, -0.06361716, -0.06315227,
            -0.06306436, -0.06425743, -0.06626847, -0.06615213, -0.06881004,
            -0.06942296, -0.06889225, -0.06868663, -0.06786670, -0.06720133,
            -0.06771172, -0.06885042, -0.06896979, -0.06961627, -0.07211988,
            -0.07252956, -0.07265559, -0.07264195, -0.07306334, -0.07282035,
            -0.07196505, -0.07210595, -0.07203942, -0.07105821, -0.06920599,
            -0.06892264, -0.06699939, -0.06537829, -0.06543323, -0.06913186,
            -0.07210039, -0.07219987, -0.07124228, -0.07065497, -0.06996833,
            -0.06744570, -0.06800847, -0.06784175, -0.06592871, -0.06723401,
        ])

        singular_value = 728.9945

        # Compare results obtained using this library. There are slight
        # variations due to the fact that we are in two different packages
        # If the sign of the components is not the same the component is
        # reflected.
        results *= (
            np.sign(fpca.components_.data_matrix.ravel()[0])
            * np.sign(results[0])
        )

        np.testing.assert_allclose(
            fpca.components_.data_matrix.ravel(),
            results,
            rtol=1e-6,
        )

        np.testing.assert_allclose(
            fpca.singular_values_,
            singular_value,
        )

    def test_grid_fpca_transform_result(self) -> None:
        """
        Compare the scores in grid against the fda.usc package.

        See test_grid_fpca_fit_result for the replication code.

        """
        n_components = 1

        fd_data = fetch_weather()['data'].coordinates[0]

        fpca = FPCA(n_components=n_components, _weights=[1] * 365)
        fpca.fit(fd_data)

        # The fda.usc uses the trapezoid rule to compute the integral
        # with the following weights
        weights = np.ones(len(fd_data.grid_points[0]))
        weights[0] = 0.5
        weights[-1] = 0.5
        fpca._weights = weights  # noqa: WPS437 (protected access)
        scores = fpca.transform(fd_data)

        # results obtained
        results = np.array([  # noqa: WPS317
            -76.43217603, -90.02095494, -81.80476223, -113.69868192,
            -69.54664059, -64.15532621, 134.93536815, -15.00125409,
            0.60569550, -36.37615052, -42.18300642, -73.71660038,
            -66.88119544, -103.15419038, -104.12065321, -7.49806764,
            7.14456774, 55.76321474, 180.16351452, -3.76283358,
            37.49075282, 123.73187622, -7.05384351, -49.36562021,
            -136.37428322, -183.00666524, -180.64875116, -50.94411798,
            -136.95768454, 49.83695668, 150.67710532, 158.20189044,
            216.43002289, 233.53770292, 344.18479151,
        ])

        np.testing.assert_allclose(scores.ravel(), results)

    def test_grid_fpca_regularization_fit_result(self) -> None:
        """Compare the components in grid against the fda.usc package."""
        n_components = 1

        fd_data = fetch_weather()['data'].coordinates[0]

        fpca = FPCA(
            n_components=n_components,
            _weights=[1] * 365,
            regularization=L2Regularization(
                LinearDifferentialOperator(2),
            ),
        )
        fpca.fit(fd_data)

        # Results obtained using fda.usc for the first component
        results = np.array([  # noqa: WPS317
            -0.06961236, -0.07027042, -0.07090496, -0.07138247, -0.07162215,
            -0.07202264, -0.07264893, -0.07279174, -0.07274672, -0.07300075,
            -0.07365471, -0.07489002, -0.07617455, -0.07658708, -0.07551923,
            -0.07375128, -0.0723776, -0.07138373, -0.07080555, -0.07111745,
            -0.0721514, -0.07395427, -0.07558341, -0.07650959, -0.0766541,
            -0.07641352, -0.07660864, -0.07669081, -0.0765396, -0.07640671,
            -0.07634668, -0.07626304, -0.07603638, -0.07549114, -0.07410347,
            -0.07181791, -0.06955356, -0.06824034, -0.06834077, -0.06944125,
            -0.07133598, -0.07341109, -0.07471501, -0.07568844, -0.07631904,
            -0.07647264, -0.07629453, -0.07598431, -0.07628157, -0.07654062,
            -0.07616026, -0.07527189, -0.07426683, -0.07267961, -0.07079998,
            -0.06927394, -0.068412, -0.06838534, -0.06888439, -0.0695309,
            -0.07005508, -0.07066637, -0.07167196, -0.07266978, -0.07275299,
            -0.07235183, -0.07207819, -0.07159814, -0.07077697, -0.06977026,
            -0.0691952, -0.06965756, -0.07058327, -0.07075751, -0.07025415,
            -0.06954233, -0.06899785, -0.06891026, -0.06887079, -0.06862183,
            -0.06830082, -0.06777765, -0.06700202, -0.06639394, -0.06582435,
            -0.06514987, -0.06467236, -0.06425272, -0.06359187, -0.062922,
            -0.06300068, -0.06325494, -0.06316979, -0.06296254, -0.06246343,
            -0.06136836, -0.0600936, -0.05910688, -0.05840872, -0.0576547,
            -0.05655684, -0.05546518, -0.05484433, -0.05465746, -0.05449286,
            -0.05397004, -0.05300742, -0.05196686, -0.05133129, -0.05064617,
            -0.04973418, -0.04855687, -0.04714356, -0.04588103, -0.04547284,
            -0.04571493, -0.04580704, -0.04523509, -0.04457293, -0.04405309,
            -0.04338468, -0.04243512, -0.04137278, -0.04047946, -0.03984531,
            -0.03931376, -0.0388847, -0.03888507, -0.03908662, -0.03877577,
            -0.03830952, -0.03802713, -0.03773521, -0.03752388, -0.03743759,
            -0.03714113, -0.03668387, -0.0363703, -0.03642288, -0.03633051,
            -0.03574618, -0.03486536, -0.03357797, -0.03209969, -0.0306837,
            -0.02963987, -0.029102, -0.0291513, -0.02932013, -0.02912619,
            -0.02869407, -0.02801974, -0.02732363, -0.02690451, -0.02676622,
            -0.0267323, -0.02664896, -0.02661708, -0.02637166, -0.02577496,
            -0.02490428, -0.02410813, -0.02340367, -0.02283356, -0.02246305,
            -0.0224229, -0.0225435, -0.02295603, -0.02324663, -0.02310005,
            -0.02266893, -0.02221522, -0.02168056, -0.02129419, -0.02064909,
            -0.02007801, -0.01979083, -0.01979541, -0.01978879, -0.01954269,
            -0.0191623, -0.01879572, -0.01849678, -0.01810297, -0.01769666,
            -0.01753802, -0.01794351, -0.01871307, -0.01930005, -0.01933,
            -0.01901017, -0.01873486, -0.01861838, -0.01870777, -0.01879,
            -0.01904219, -0.01945078, -0.0200607, -0.02076936, -0.02100213,
            -0.02071439, -0.02052113, -0.02076313, -0.02128468, -0.02175631,
            -0.02206387, -0.02201054, -0.02172142, -0.02143092, -0.02133647,
            -0.02144956, -0.02176286, -0.02212579, -0.02243861, -0.02278316,
            -0.02304113, -0.02313356, -0.02349275, -0.02417028, -0.0245954,
            -0.0244062, -0.02388557, -0.02374682, -0.02401071, -0.02431126,
            -0.02433125, -0.02427656, -0.02430442, -0.02424977, -0.02401619,
            -0.02402294, -0.02415424, -0.02413262, -0.02404076, -0.02397651,
            -0.0243893, -0.0253322, -0.02664395, -0.0278802, -0.02877936,
            -0.02927182, -0.02937318, -0.02926277, -0.02931632, -0.02957945,
            -0.02982133, -0.03023224, -0.03060406, -0.03066011, -0.03070932,
            -0.03116429, -0.03179009, -0.03198094, -0.03149462, -0.03082037,
            -0.03041594, -0.0303307, -0.03028465, -0.03052841, -0.0311837,
            -0.03199307, -0.03262025, -0.03345083, -0.03442665, -0.03521313,
            -0.0356433, -0.03606037, -0.03677406, -0.03735165, -0.03746578,
            -0.03744154, -0.03752143, -0.03780898, -0.03837639, -0.03903232,
            -0.03911629, -0.03857567, -0.03816592, -0.03819285, -0.03818405,
            -0.03801684, -0.03788493, -0.03823232, -0.03906142, -0.04023251,
            -0.04112434, -0.04188011, -0.04254759, -0.043, -0.04340181,
            -0.04412687, -0.04484482, -0.04577669, -0.04700832, -0.04781373,
            -0.04842662, -0.04923723, -0.05007637, -0.05037817, -0.05009794,
            -0.04994083, -0.05012712, -0.05094001, -0.05216065, -0.05350458,
            -0.05469781, -0.05566309, -0.05641011, -0.05688106, -0.05730818,
            -0.05759156, -0.05763771, -0.05760073, -0.05766117, -0.05794587,
            -0.05816696, -0.0584046, -0.05905105, -0.06014331, -0.06142231,
            -0.06270788, -0.06388225, -0.06426245, -0.06386721, -0.0634656,
            -0.06358049, -0.06442514, -0.06570047, -0.06694328, -0.0682621,
            -0.06897846, -0.06896583, -0.06854621, -0.06797142, -0.06763755,
            -0.06784024, -0.06844314, -0.06918567, -0.07021928, -0.07148473,
            -0.07232504, -0.07272276, -0.07287021, -0.07289836, -0.07271531,
            -0.07239956, -0.07214086, -0.07170078, -0.07081195, -0.06955202,
            -0.06825156, -0.06690167, -0.06617102, -0.06683291, -0.06887539,
            -0.07089424, -0.07174837, -0.07150888, -0.07070378, -0.06960066,
            -0.06842496, -0.06777666, -0.06728403, -0.06681262, -0.06679066,
        ])

        # Compare results obtained using this library. There are slight
        # variations due to the fact that we are in two different packages
        # If the sign of the components is not the same the component is
        # reflected.
        results *= (
            np.sign(fpca.components_.data_matrix.ravel()[0])
            * np.sign(results[0])
        )

        np.testing.assert_allclose(
            fpca.components_.data_matrix.ravel(),
            results,
            rtol=1e-2,
        )

    def draw_one_random_fun(
        self,
        basis: Basis,
        random_state: np.random.RandomState,
    ) -> FDataBasis:
        """Draw a true function in a given basis with random coef."""
        coef = random_state.uniform(-10, 10, size=basis.n_basis)
        return FDataBasis(
            basis=basis,
            coefficients=coef,
        )

    def _test_vs_dim_grid(
        self,
        random_state: np.random.RandomState,
        n_samples: int,
        n_grid: int,
        base_fun: FDataBasis,
    ) -> None:
        """Test function w.r.t n_samples, n_grid."""
        # Random offsetting base_fun and form dataset fd_random
        offset = random_state.uniform(-5, 5, size=n_samples)

        fd_random = FDataBasis(
            basis=base_fun.basis,
            coefficients=base_fun.coefficients * offset[:, np.newaxis],
        ).to_grid(np.linspace(0, 1, n_grid))

        # Take the allowed maximum number of components
        # In almost high dimension: n_components=n_samples-1 < n_samples
        # In low dimension: n_components=n_grid << n_samples
        fpca = FPCA(
            n_components=min(n_samples - 1, n_grid),
        )

        # Project the non-random dataset on FPCs
        pc_scores_fd_random_all_equal = fpca.fit_transform(
            fd_random,
        )

        # Project the pc scores back to the input functional space
        fd_random_hat = fpca.inverse_transform(
            pc_scores_fd_random_all_equal,
        )

        # Compare fitting data to the reconstructed ones
        np.testing.assert_allclose(
            fd_random.data_matrix,
            fd_random_hat.data_matrix,
        )

    def test_grid_fpca_inverse_transform(self) -> None:
        """Compare the reconstructions.data_matrix to fitting data."""
        random_state = np.random.RandomState(seed=42)

        # Low dimensional case (n_samples>n_grid)
        n_samples = 1000
        n_grid = 100
        bsp = BSplineBasis(
            domain_range=(0, 50),
            n_basis=100,
            order=3,
        )
        true_fun = self.draw_one_random_fun(bsp, random_state)
        self._test_vs_dim_grid(
            random_state=random_state,
            n_samples=n_samples,
            n_grid=n_grid,
            base_fun=true_fun,
        )

        # (almost) High dimensional case (n_samples<n_grid)
        n_samples = 100
        n_grid = 1000
        bsp = BSplineBasis(
            domain_range=(0, 50),
            n_basis=100,
            order=3,
        )
        true_fun = self.draw_one_random_fun(bsp, random_state)
        self._test_vs_dim_grid(
            random_state=random_state,
            n_samples=n_samples,
            n_grid=n_grid,
            base_fun=true_fun,
        )

    def _test_vs_dim_basis(
        self,
        random_state: np.random.RandomState,
        n_samples: int,
        base_fun: FDataBasis,
    ) -> None:
        """Test function w.t.t n_samples and basis."""
        # Random offsetting base_fun and form dataset fd_random
        offset = random_state.uniform(-5, 5, size=n_samples)

        fd_random = FDataBasis(
            basis=base_fun.basis,
            coefficients=base_fun.coefficients * offset[:, np.newaxis],
        )

        # Take the allowed maximum number of components
        # In almost high dimension: n_components=n_samples-1 < n_samples
        # In low dimension: n_components=n_basis<<n_samples
        fpca = FPCA(n_components=min(n_samples - 1, base_fun.n_basis))

        # Project non-random dataset on fitted FPCs
        pc_scores = fpca.fit_transform(fd_random)

        # Project back pc_scores to functional input space
        fd_random_hat = fpca.inverse_transform(pc_scores)

        # Compare fitting data to the reconstructed ones
        np.testing.assert_allclose(
            fd_random.coefficients,
            fd_random_hat.coefficients,
        )

    def test_basis_fpca_inverse_transform(self) -> None:
        """Compare the coef reconstructions to fitting data."""
        random_state = np.random.RandomState(seed=42)

        # Low dimensional case: n_basis<<n_samples
        n_samples = 1000
        n_basis = 20
        bsp = BSplineBasis(
            domain_range=(0, 50),
            n_basis=n_basis,
            order=3,
        )
        true_fun = self.draw_one_random_fun(bsp, random_state)
        self._test_vs_dim_basis(
            random_state=random_state,
            n_samples=n_samples,
            base_fun=true_fun,
        )

        # Case n_samples<n_basis
        n_samples = 10
        n_basis = 20
        bsp = BSplineBasis(
            domain_range=(0, 50),
            n_basis=n_basis,
            order=3,
        )
        true_fun = self.draw_one_random_fun(bsp, random_state)
        self._test_vs_dim_basis(
            random_state=random_state,
            n_samples=n_samples,
            base_fun=true_fun,
        )


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
from skfda import datasets, FDataBasis
from skfda.representation.basis import Fourier
from skfda.ml.regression.functional import FunctionalRegression


class TestRegression(unittest.TestCase):

    def test_Functional(self):
        canadian = datasets.fetch_cran('CanadianWeather', 'fda')
        day5 = datasets.fetch_cran('dateAccessories', 'fda')['day.5'].tolist()
        tempav = canadian['CanadianWeather']['dailyAv'][:, :, 0].values
        region = canadian['CanadianWeather']['region']
        regions = np.unique(region).tolist()

        tempbasis = Fourier((0, 365), nbasis=65)
        tempfd = FDataBasis.from_data(np.transpose(tempav), day5,
                                      Fourier((0, 365), 65))

        coef = tempfd.coefficients
        coef36 = np.concatenate((coef, np.zeros((1, 65))), axis=0)

        temp36fd = FDataBasis(tempbasis, coef36)

        regionList = []
        regionList.append(np.ones((1, 36)).tolist())
        regionList[0][0][35] = 0
        for j in range(len(regions)):
            xj = (((regions[j] == np.array(region)) * 1).tolist())
            xj.append(1)
            regionList.append(xj)

        betabasis = Fourier((0, 365), 11)

        betalist = [betabasis.copy() for _ in range(len(regions) + 1)]

        functional = FunctionalRegression(betalist)

        functional.fit(temp36fd, regionList)

        r = []
        r.append(
            FDataBasis(Fourier(domain_range=(0, 365), nbasis=11, period=365),
                       np.array([-0.105765640113201, -70.1008906808154,
                                 -192.711924292549, 0.373395529218503,
                                 -2.52690914730158, 0.829515570963221,
                                 -2.49501279723799, 2.73529633035966,
                                 -1.11445193998509, -1.34873376025209,
                                 1.48810474871709]).reshape((1, 11))))
        r.append(
            FDataBasis(Fourier(domain_range=(0, 365), nbasis=11, period=365),
                       np.array([-225.317255259659, -39.7006394695827,
                                 -50.1563807946539, 4.66031887532537,
                                 24.0639511885448, 9.79596379506588,
                                 4.3561651212213, -0.984004413620673,
                                 3.37727269260748, -0.93903293330625,
                                 -0.311057012884717]).reshape((1, 11))))
        r.append(
            FDataBasis(Fourier(domain_range=(0, 365), nbasis=11, period=365),
                       np.array([87.5273504312752, -4.56520238242003,
                                 16.9986721136565, -8.59489293693191,
                                 -3.38834439921161, -8.72708831511192,
                                 -1.52430096756341, -0.961323383195038,
                                 -0.98833970247679, 3.02748195729272,
                                 -1.85295730632822]).reshape((1, 11))))
        r.append(
            FDataBasis(Fourier(domain_range=(0, 365), nbasis=11, period=365),
                       np.array([-10.7427170816731, 1.0530798165301,
                                 -44.756843929379, -6.98754511058317,
                                 -16.3779405806873, -1.13463588145337,
                                 -1.60130708615852, 3.37867645021071,
                                 -0.0903358352298147, 0.647696889261022,
                                 1.52895224395047]).reshape((1, 11))))
        r.append(
            FDataBasis(Fourier(domain_range=(0, 365), nbasis=11, period=365),
                       np.array([149.307805641942, 38.7840860386028,
                                 77.154585724783, 10.8127338785809,
                                 -4.88995600594543, -1.17593920166156,
                                 -0.258161039215971, -2.09496368598739,
                                 -2.01604238528718, -0.36253880424383,
                                 0.0423903933831828]).reshape((1, 11))))

        for i in range(5):
            np.testing.assert_array_almost_equal(
                r[i].coefficients,
                functional.beta[i].coefficients,
                decimal=10,
                err_msg="error on sample " + str(i))

    def test_functional(self):
        xbasis = Fourier(nbasis=5)
        xcoef = [[1, 2, 3, 4, 5], [4, 8, 7, 6, 2], [1, 2, 5, 9, 8],
                 [3, 6, 4, 7, 8], [1, 5, 9, 4, 6]]
        x = FDataBasis(xbasis, xcoef)


        xbasis = Fourier(nbasis=5)
        bcoef = [[1, 2, 3, 4, 5]]
        b = FDataBasis(xbasis, bcoef)

        functional = FunctionalRegression([b])
        print(functional.predict([x]))

if __name__ == '__main__':
    print()
    unittest.main()

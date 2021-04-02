"""Outliergram testing module.

Module containing the test coverage of outliergram module.
"""

import unittest

import numpy as np

from skfda.datasets import fetch_weather
from skfda.exploratory.visualization import Outliergram


class TestOutliergram(unittest.TestCase):
    """
    Outliergram testing class.

    Class containing the test coverage of outliergram module.
    """

    def test_outliergram(self) -> None:
        """
        Outliergram testing method.

        Method containing the test coverage of outliergram module.
        """
        fd = fetch_weather()["data"]
        fd_temperatures = fd.coordinates[0]
        outliergram = Outliergram(
            fd_temperatures,
        )
        # noqa: WPS317
        np.testing.assert_allclose(
            outliergram.mei,
            np.array(
                [  # noqa: WPS317
                    0.46272668, 0.27840835, 0.36268754, 0.27908676, 0.36112198,
                    0.30802348, 0.82969341, 0.45904762, 0.53907371, 0.38799739,
                    0.41283757, 0.20420091, 0.23564253, 0.14737117, 0.14379648,
                    0.54035225, 0.43459883, 0.6378604, 0.86964123, 0.4421396,
                    0.58906719, 0.75561644, 0.54982387, 0.46095238, 0.09969993,
                    0.13166341, 0.18776256, 0.4831833, 0.36816699, 0.72962818,
                    0.80313112, 0.79934768, 0.90643183, 0.90139596, 0.9685062,
                ],
            ),
            rtol=1e-5,
        )

        np.testing.assert_array_almost_equal(
            outliergram.mbd,
            np.array(
                [  # noqa: WPS317
                    0.40685162, 0.42460381, 0.43088139, 0.35833775, 0.47847435,
                    0.46825985, 0.29228349, 0.51299183, 0.5178558, 0.49868539,
                    0.52408733, 0.34457312, 0.36996431, 0.2973209, 0.29107555,
                    0.53304017, 0.44185565, 0.46346341, 0.23620736, 0.47652354,
                    0.4814397, 0.38233529, 0.51173171, 0.51164882, 0.21551437,
                    0.23084916, 0.25650589, 0.46760447, 0.30787767, 0.40929051,
                    0.31801082, 0.3234519, 0.17015617, 0.17977514, 0.05769541,
                ],
            ),
        )


if __name__ == '__main__':
    print()  # noqa: WPS421
    unittest.main()

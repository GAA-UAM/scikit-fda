import warnings
from typing import Optional

import numpy as np

from ...typing._base import DomainRangeLike
from ._fourier_basis import FourierBasis


class Fourier(FourierBasis):
    r"""Fourier basis.

    Defines a functional basis for representing functions on a fourier
    series expansion of period :math:`T`. The number of basis is always odd.
    If instantiated with an even number of basis, they will be incremented
    automatically by one.

    .. math::
        \phi_0(t) = \frac{1}{\sqrt{2}}

    .. math::
        \phi_{2n -1}(t) = \frac{sin\left(\frac{2 \pi n}{T} t\right)}
                                                    {\sqrt{\frac{T}{2}}}

    .. math::
        \phi_{2n}(t) = \frac{cos\left(\frac{2 \pi n}{T} t\right)}
                                                    {\sqrt{\frac{T}{2}}}


    This basis will be orthonormal if the period coincides with the length
    of the interval in which it is defined.

    .. deprecated:: 0.8
        Use :class:`~skfda.representation.basis.FourierBasis` instead.

    Parameters:
        domain_range: A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis: Number of functions in the basis.
        period: Period (:math:`T`).

    Examples:
        Constructs specifying number of basis, definition interval and period.

        >>> fb = FourierBasis((0, np.pi), n_basis=3, period=1)
        >>> fb([0, np.pi / 4, np.pi / 2, np.pi]).round(2)
        array([[[ 1.  ],
                [ 1.  ],
                [ 1.  ],
                [ 1.  ]],
               [[ 0.  ],
                [-1.38],
                [-0.61],
                [ 1.1 ]],
               [[ 1.41],
                [ 0.31],
                [-1.28],
                [ 0.89]]])

        And evaluate second derivative

        >>> deriv2 = fb.derivative(order=2)
        >>> deriv2([0, np.pi / 4, np.pi / 2, np.pi]).round(2)
        array([[[  0.  ],
                [  0.  ],
                [  0.  ],
                [  0.  ]],
               [[  0.  ],
                [ 54.46],
                [ 24.02],
                [-43.37]],
               [[-55.83],
                [-12.32],
                [ 50.4 ],
                [-35.16]]])

    """

    def __init__(
        self,
        domain_range: Optional[DomainRangeLike] = None,
        n_basis: int = 3,
        period: Optional[float] = None,
    ) -> None:
        warnings.warn(
            "The Fourier class is deprecated. Use FourierBasis instead.",
            DeprecationWarning,
        )
        super().__init__(
            domain_range=domain_range,
            n_basis=n_basis,
            period=period,
        )

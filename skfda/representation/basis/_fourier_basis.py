import warnings
from typing import Any, Optional, Sequence, Tuple, TypeVar

import numpy as np
from typing_extensions import Protocol

from ...typing._base import DomainRangeLike
from ...typing._numpy import NDArrayFloat
from ._basis import Basis

T = TypeVar("T", bound="FourierBasis")


class _SinCos(Protocol):
    def __call__(
        self,
        __array: NDArrayFloat,  # noqa: WPS112
        out: NDArrayFloat,
    ) -> NDArrayFloat:
        pass


class FourierBasis(Basis):
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
        """
        Construct a FourierBasis object.

        It forces the object to have an odd number of basis. If n_basis is
        even, it is incremented by one.

        Args:
            domain_range: Tuple defining the domain over which the
                function is defined.
            n_basis: Number of basis functions.
            period: Period of the trigonometric functions that
                define the basis.

        """
        from ...misc.validation import validate_domain_range

        if domain_range is not None:
            domain_range = validate_domain_range(domain_range)

            if len(domain_range) != 1:
                raise ValueError("Domain range should be unidimensional.")

            domain_range = domain_range[0]

        self._period = period
        # If number of basis is even, add 1
        n_basis += 1 - n_basis % 2
        super().__init__(domain_range=domain_range, n_basis=n_basis)

    @property
    def period(self) -> float:
        if self._period is None:
            return self.domain_range[0][1] - self.domain_range[0][0]

        return self._period

    def _evaluate(self, eval_points: NDArrayFloat) -> NDArrayFloat:

        # Input is scalar
        eval_points = eval_points[..., 0]

        functions: Sequence[_SinCos] = [np.sin, np.cos]
        omega = 2 * np.pi / self.period

        normalization_denominator = np.sqrt(self.period / 2)

        seq = 1 + np.arange((self.n_basis - 1) // 2)
        seq_pairs = np.array([seq, seq]).T
        phase_coefs = omega * seq_pairs

        # Multiply the phase coefficients elementwise
        res = np.einsum("ij,k->ijk", phase_coefs, eval_points)

        # Apply odd and even functions
        for i in (0, 1):
            functions[i](res[:, i, :], out=res[:, i, :])

        res = res.reshape(-1, len(eval_points))
        res /= normalization_denominator

        constant_basis = np.full(
            shape=(1, len(eval_points)),
            fill_value=1 / (np.sqrt(2) * normalization_denominator),
        )

        return np.concatenate((constant_basis, res))

    def _derivative_basis_and_coefs(
        self: T,
        coefs: NDArrayFloat,
        order: int = 1,
    ) -> Tuple[T, NDArrayFloat]:

        omega = 2 * np.pi / self.period
        deriv_factor = (np.arange(1, (self.n_basis + 1) / 2) * omega) ** order

        deriv_coefs = np.zeros(coefs.shape)

        cos_sign, sin_sign = (
            (-1) ** int((order + 1) / 2),
            (-1) ** int(order / 2),
        )

        if order % 2 == 0:
            deriv_coefs[:, 1::2] = sin_sign * coefs[:, 1::2] * deriv_factor
            deriv_coefs[:, 2::2] = cos_sign * coefs[:, 2::2] * deriv_factor
        else:
            deriv_coefs[:, 2::2] = sin_sign * coefs[:, 1::2] * deriv_factor
            deriv_coefs[:, 1::2] = cos_sign * coefs[:, 2::2] * deriv_factor

        # normalise
        return self.copy(), deriv_coefs

    def _gram_matrix(self) -> NDArrayFloat:

        # Orthogonal in this case
        if self.period == (self.domain_range[0][1] - self.domain_range[0][0]):
            return np.identity(self.n_basis)

        return super()._gram_matrix()

    def rescale(  # noqa: D102
        self: T,
        domain_range: Optional[DomainRangeLike] = None,
        *,
        rescale_period: bool = False,
    ) -> T:

        rescale_basis = super().rescale(domain_range)

        if rescale_period is True:

            domain_rescaled = rescale_basis.domain_range[0]
            domain = self.domain_range[0]

            rescale_basis._period = (  # noqa: WPS437
                self.period
                * (domain_rescaled[1] - domain_rescaled[0])
                / (domain[1] - domain[0])
            )

        return rescale_basis

    def _to_R(self) -> str:  # noqa: N802
        drange = self.domain_range[0]
        rangeval = f"c({drange[0]}, {drange[1]})"
        return (
            f"create.fourier.basis("
            f"rangeval = {rangeval}, "
            f"nbasis = {self.n_basis}, "
            f"period = {self.period})"
        )

    def __repr__(self) -> str:
        """Representation of a Fourier basis."""
        return (
            f"{self.__class__.__name__}("
            f"domain_range={self.domain_range}, "
            f"n_basis={self.n_basis}, "
            f"period={self.period})"
        )

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and self.period == other.period

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.period))


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

        >>> fb = Fourier((0, np.pi), n_basis=3, period=1)
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

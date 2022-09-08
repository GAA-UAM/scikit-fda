"""Defines the most commonly used kernels."""
import math

import numpy as np
from scipy import stats

from ..typing._numpy import NDArrayFloat


def normal(u: NDArrayFloat) -> NDArrayFloat:
    r"""Evaluate a normal kernel.

    .. math::
        K(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}

    """
    return stats.norm.pdf(u)  # type: ignore[no-any-return]


def cosine(u: NDArrayFloat) -> NDArrayFloat:
    r"""Cosine kernel.

    .. math::
        K(x) =
        \begin{cases}
        \frac{\pi}{4} cos\left( \frac{\pi x}{2} \right) & \mbox{if } |x| \le
        1 \\
        0 & \mbox{elsewhere}
        \end{cases}

    """
    if isinstance(u, np.ndarray):
        res = np.zeros(u.shape)
        res[abs(u) <= 1] = math.pi / 4 * (
            math.cos(math.pi * u[abs(u) <= 1] / 2)
        )
        return res
    if abs(u) <= 1:
        return math.pi / 4 * (math.cos(math.pi * u / 2))
    return 0


def epanechnikov(u: NDArrayFloat) -> NDArrayFloat:
    r"""Epanechnikov kernel.

    .. math::
        K(x) =
        \begin{cases}
        0.75(1-x^2) & \mbox{if } |x| \le 1 \\
        0 & \mbox{elsewhere}
        \end{cases}

    """
    if isinstance(u, np.ndarray):
        res = np.zeros(u.shape)
        res[abs(u) <= 1] = 0.75 * (1 - u[abs(u) <= 1] ** 2)
        return res
    if abs(u) <= 1:
        return 0.75 * (1 - u ** 2)
    return 0


def tri_weight(u: NDArrayFloat) -> NDArrayFloat:
    r"""Tri-weight kernel.

    .. math::
        K(x) =
        \begin{cases}
        \frac{35}{32} \left(1 - u^2 \right) ^3 & \mbox{if } |x| \le 1 \\
        0 & \mbox{elsewhere}
        \end{cases}

    """
    if isinstance(u, np.ndarray):
        res = np.zeros(u.shape)
        res[abs(u) <= 1] = 35 / 32 * (1 - u[abs(u) <= 1] ** 2) ** 3
        return res
    if abs(u) <= 1:
        return 35 / 32 * (1 - u ** 2) ** 3
    return 0


def quartic(u: NDArrayFloat) -> NDArrayFloat:
    r"""Quartic kernel.

    .. math::
        K(x) =
        \begin{cases}
        \frac{15}{16} \left( 1- u^2 \right) ^2 & \mbox{if } |x| \le 1 \\
        0 & \mbox{elsewhere}
        \end{cases}

    """
    if isinstance(u, np.ndarray):
        res = np.zeros(u.shape)
        res[abs(u) <= 1] = 15 / 16 * (1 - u[abs(u) <= 1] ** 2) ** 2
        return res
    if abs(u) <= 1:
        return 15 / 16 * (1 - u ** 2) ** 2
    return 0


def uniform(u: NDArrayFloat) -> NDArrayFloat:
    r"""Uniform kernel.

    .. math::
        K(x) =
        \begin{cases}
        0.5 & \mbox{if } |x| \le
        1 \\
        0 & \mbox{elsewhere}
        \end{cases}

    """
    if isinstance(u, np.ndarray):
        res = np.zeros(u.shape)
        res[abs(u) <= 1] = 0.5
        return res
    if abs(u) <= 1:
        return 0.5
    return 0

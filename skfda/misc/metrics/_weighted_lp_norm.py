"""Implementation of Weighted Lp norms."""

from collections.abc import Callable

import numpy as np

from ..._utils import nquad_vec
from ...representation import FData, FDataBasis, FDataGrid
from ...typing._base import (
    GridPointsLike,
)
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat


class WeightedLpNorm:
    r"""
    Weighted Lp norm for functional data objects.

    This class generalizes the standard Lp norm by incorporating optional
    weighting functions over the domain of the data. It supports both
    `FDataGrid` and `FDataBasis` representations, as well as raw NumPy arrays.

    For a univariate or vector valued function :math:`X: \mathcal{T}
    \rightarrow \mathbb{R}^D`, the weighted Lp norm is defined as:

    .. math::
        \| \mathbf{X} \|_{p,w} = \left( \int_{\mathcal{T}}
        w(t)\| \mathbf{X}(t) \|_{\mathbb{R}^D}^p \, dt \right)^{\frac{1}{p}}.

    Where:
        - :math:`\| \cdot \|_{\mathbb{R}^D}^p` is a vectorial norm applied
        pointwise, typically an Lp norm or the Euclidean norm.
        - :math:`w(x)` is the weighting function.
        - :math:`p \geq 1` is the order of the norm.
        - :math:`D` is the domain of the function.

    If no weighting function is provided, the norm reduces to the standard
    (unweighted) Lp norm.

    The integration is performed using Simpson's rule for `FDataGrid` and
    `nquad_vec` for `FDataBasis`.

    Args:
        p: Exponent of the Lp norm. Must be greater than or equal to 1. If set
            to `math.inf`, the norm becomes the L-infinity norm.
        vector_norm: Norm to apply pointwise to multivariate functions.
            If a float is passed, it is interpreted as an Lp norm index.
            If `None`, defaults to the value of `p`.
        lp_weight: Optional weight to apply in the integral. It can be a float
            (applied uniformly) or a callable function taking domain points as
            input and returning weights.

    Raises:
        ValueError: If `p` is less than 1 and not infinite.
        NotImplementedError: If the input data type is unsupported.

    Examples:

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x)), x], x)
        >>> from skfda.misc.metrics._weighted_lp_norm import WeightedLpNorm
        >>> weighted_norm = WeightedLpNorm(p=2, lp_weight=2.0)
        >>> weighted_norm(fd).round(2)
        array([1.41, 0.82])

    See Also:
        :class:`LpNorm`: A subclass that sets a default weight of 1.0.
        :func:`weighted_lp_norm`: Functional wrapper.
        :func:`vectorial_norm`: Helper function used for pointwise norm
        evaluation.
    """
    def __init__(
        self,
        p: float,
        vector_norm: Norm[NDArrayFloat] | float | None = None,
        lp_weight: (
            Callable[[GridPointsLike], NDArrayFloat] | float | None
        ) = None,
    ) -> None:

        # Checks that the lp normed is well defined
        if not np.isinf(p) and p < 1:
            msg = f"p (={p}) must be equal or greater than 1."
            raise ValueError(msg)

        self.p = p
        self.vector_norm = vector_norm
        self.lp_weight = lp_weight

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(p={self.p},"
            f" vector_norm={self.vector_norm})"
        )

    def __call__(self, vector: NDArrayFloat | FData) -> NDArrayFloat:  # noqa: C901, PLR0912
        """Compute the Lp norm of a functional data object."""
        from ...misc import inner_product

        if isinstance(vector, np.ndarray):
            if isinstance(self.lp_weight, (float, int)):
                vector = vector * self.lp_weight
            return np.linalg.norm(  # type: ignore[no-any-return]
                vector,
                ord=self.p,
                axis=-1,
            )

        vector_norm = self.vector_norm
        lp_weight = self.lp_weight

        if vector_norm is None:
            vector_norm = self.p
        if lp_weight is None:
            lp_weight = 1.0

        if lp_weight ==1.0 and self.p ==vector_norm==2:  # noqa: PLR2004
            return np.sqrt(inner_product(vector, vector))

        if isinstance(vector, FDataBasis):
            domain = vector.basis.domain_range
            call = vector

            def integrand(*args: GridPointsLike) -> NDArrayFloat:
                f_args = np.asarray(args)

                try:
                    f1 = call(f_args)[:, 0, :]
                except Exception:  # noqa: BLE001
                    f1 = call(f_args)
                weight = (
                    lp_weight
                    if isinstance(lp_weight, (float, int))
                    else lp_weight(f_args)
                )
                return np.asarray(
                    np.power(np.abs(f1), self.p) * weight,
                    dtype=np.float64,
                )

            integral = nquad_vec(
                integrand,
                domain,
            )

            res = (np.sum(integral, axis=-1)) ** (1 / self.p)

        elif isinstance(vector, FDataGrid):
            data_matrix = vector.data_matrix

            if isinstance(vector_norm, (float, int)):
                data_matrix = np.linalg.norm(
                    vector.data_matrix,
                    ord=vector_norm,
                    axis=-1,
                    keepdims=True,
                )
            else:
                original_shape = data_matrix.shape
                data_matrix = data_matrix.reshape(-1, original_shape[-1])
                data_matrix = vector_norm(data_matrix)
                data_matrix = data_matrix.reshape(original_shape[:-1] + (1,))

            data_matrix = (
                data_matrix * lp_weight
                if isinstance(lp_weight, (float, int))
                else lp_weight(vector.grid_points) * data_matrix
            )

            if np.isinf(self.p):
                res = np.max(
                    data_matrix,
                    axis=tuple(range(1, data_matrix.ndim)),
                )

            else:
                integrand = vector.copy(
                    data_matrix=data_matrix**self.p,
                    coordinate_names=(None,),
                )
                # Computes the norm, approximating the integral with Simpson's
                # rule.
                res = integrand.integrate().ravel() ** (1 / self.p)
        else:
            msg = f"LpNorm not implemented for type {type(vector)}"
            raise NotImplementedError(msg)

        if len(res) == 1:
            return res[0]  # type: ignore[no-any-return]

        return res  # type: ignore[no-any-return]


def weighted_lp_norm(
    vector: NDArrayFloat | FData,
    *,
    p: float,
    vector_norm: Norm[NDArrayFloat] | float | None = None,
    lp_weight: Callable[[GridPointsLike], NDArrayFloat] | float | None = None,
) -> NDArrayFloat:
    """
    Functional wrapper for computing weighted Lp norms.

    See :class:`~skfda.misc.metrics.WeightedLpNorm` for full documentation.
    """
    return WeightedLpNorm(p=p, vector_norm=vector_norm, lp_weight=lp_weight)(
        vector,
    )

"""Implementation of Lp norms."""
import math
from builtins import isinstance
from typing import Union

import numpy as np
import scipy.integrate
from typing_extensions import Final

from ...representation import FData, FDataBasis
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat


class LpNorm():
    r"""
    Norm of all the observations in a FDataGrid object.

    For each observation f the Lp norm is defined as:

    .. math::
        \| f \| = \left( \int_D \| f \|^p dx \right)^{
        \frac{1}{p}}

    Where D is the :term:`domain` over which the functions are defined.

    The integral is approximated using Simpson's rule.

    In general, if f is a multivariate function :math:`(f_1, ..., f_d)`, and
    :math:`D \subset \mathbb{R}^n`, it is applied the following generalization
    of the Lp norm.

    .. math::
        \| f \| = \left( \int_D \| f \|_{*}^p dx \right)^{
        \frac{1}{p}}

    Where :math:`\| \cdot \|_*` denotes a vectorial norm. See
    :func:`vectorial_norm` to more information.

    For example, if :math:`f: \mathbb{R}^2 \rightarrow \mathbb{R}^2`, and
    :math:`\| \cdot \|_*` is the euclidean norm
    :math:`\| (x,y) \|_* = \sqrt{x^2 + y^2}`, the lp norm applied is

    .. math::
        \| f \| = \left( \int \int_D \left ( \sqrt{ \| f_1(x,y)
        \|^2 + \| f_2(x,y) \|^2 } \right )^p dxdy \right)^{
        \frac{1}{p}}

    The objects ``l1_norm``, ``l2_norm`` and ``linf_norm`` are instances of
    this class with commonly used values of ``p``, namely 1, 2 and infinity.

    Args:
        p: p of the lp norm. Must be greater or equal
            than 1. If ``p=math.inf`` it is used the L infinity metric.
            Defaults to 2.
        vector_norm: vector norm to apply. If it is a float, is the index of
            the multivariate lp norm. Defaults to the same as ``p``.

    Examples:
        Calculates the norm of a FDataGrid containing the functions y = 1
        and y = x defined in the interval [0,1].

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x)), x] ,x)
        >>> norm = skfda.misc.metrics.LpNorm(2)
        >>> norm(fd).round(2)
        array([ 1.  ,  0.58])

        As the norm with `p=2` is a common choice, one can use `l2_norm`
        directly:

        >>> skfda.misc.metrics.l2_norm(fd).round(2)
        array([ 1.  ,  0.58])

        The lp norm is only defined if p >= 1.

        >>> norm = skfda.misc.metrics.LpNorm(0.5)
        Traceback (most recent call last):
            ....
        ValueError: p (=0.5) must be equal or greater than 1.

    """

    def __init__(
        self,
        p: float,
        vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
    ) -> None:

        # Checks that the lp normed is well defined
        if not np.isinf(p) and p < 1:
            raise ValueError(f"p (={p}) must be equal or greater than 1.")

        self.p = p
        self.vector_norm = vector_norm

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p={self.p}, vector_norm={self.vector_norm})"
        )

    def __call__(self, vector: Union[NDArrayFloat, FData]) -> NDArrayFloat:
        """Compute the Lp norm of a functional data object."""
        from ...misc import inner_product

        if isinstance(vector, np.ndarray):
            return np.linalg.norm(  # type: ignore[no-any-return]
                vector,
                ord=self.p,
                axis=-1,
            )

        vector_norm = self.vector_norm

        if vector_norm is None:
            vector_norm = self.p

        # Special case, the inner product is heavily optimized
        if self.p == vector_norm == 2:
            return np.sqrt(inner_product(vector, vector))

        if isinstance(vector, FDataBasis):
            if self.p != 2:
                raise NotImplementedError

            start, end = vector.domain_range[0]
            integral = scipy.integrate.quad_vec(
                lambda x: np.power(np.abs(vector(x)), self.p),
                start,
                end,
            )
            res = np.sqrt(integral[0]).flatten()

        else:
            data_matrix = vector.data_matrix
            original_shape = data_matrix.shape
            data_matrix = data_matrix.reshape(-1, original_shape[-1])

            data_matrix = (np.linalg.norm(
                vector.data_matrix,
                ord=vector_norm,
                axis=-1,
                keepdims=True,
            ) if isinstance(vector_norm, (float, int))
                else vector_norm(data_matrix)
            )
            data_matrix = data_matrix.reshape(original_shape[:-1] + (1,))

            if np.isinf(self.p):

                res = np.max(
                    data_matrix,
                    axis=tuple(range(1, data_matrix.ndim)),
                )

            elif vector.dim_domain == 1:

                # Computes the norm, approximating the integral with Simpson's
                # rule.
                res = scipy.integrate.simps(
                    data_matrix[..., 0] ** self.p,
                    x=vector.grid_points,
                ) ** (1 / self.p)

            else:
                # Needed to perform surface integration
                return NotImplemented

        if len(res) == 1:
            return res[0]  # type: ignore[no-any-return]

        return res  # type: ignore[no-any-return]


l1_norm: Final = LpNorm(1)
l2_norm: Final = LpNorm(2)
linf_norm: Final = LpNorm(math.inf)


def lp_norm(
    vector: Union[NDArrayFloat, FData],
    *,
    p: float,
    vector_norm: Union[Norm[NDArrayFloat], float, None] = None,
) -> NDArrayFloat:
    r"""Calculate the norm of all the observations in a FDataGrid object.

    For each observation f the Lp norm is defined as:

    .. math::
        \| f \| = \left( \int_D \| f \|^p dx \right)^{
        \frac{1}{p}}

    Where D is the :term:`domain` over which the functions are defined.

    The integral is approximated using Simpson's rule.

    In general, if f is a multivariate function :math:`(f_1, ..., f_d)`, and
    :math:`D \subset \mathbb{R}^n`, it is applied the following generalization
    of the Lp norm.

    .. math::
        \| f \| = \left( \int_D \| f \|_{*}^p dx \right)^{
        \frac{1}{p}}

    Where :math:`\| \cdot \|_*` denotes a vectorial norm. See
    :func:`vectorial_norm` to more information.

    For example, if :math:`f: \mathbb{R}^2 \rightarrow \mathbb{R}^2`, and
    :math:`\| \cdot \|_*` is the euclidean norm
    :math:`\| (x,y) \|_* = \sqrt{x^2 + y^2}`, the lp norm applied is

    .. math::
        \| f \| = \left( \int \int_D \left ( \sqrt{ \| f_1(x,y)
        \|^2 + \| f_2(x,y) \|^2 } \right )^p dxdy \right)^{
        \frac{1}{p}}

    Note:
        This function is a wrapper of :class:`LpNorm`, available only for
        convenience. As the parameter ``p`` is mandatory, it cannot be used
        where a fully-defined norm is required: use an instance of
        :class:`LpNorm` in those cases.

    Args:
        vector: Vector object.
        p: p of the lp norm. Must be greater or equal
            than 1. If ``p=math.inf`` it is used the L infinity metric.
            Defaults to 2.
        vector_norm: vector norm to apply. If it is a float, is the index of
            the multivariate lp norm. Defaults to the same as ``p``.

    Returns:
        numpy.darray: Matrix with as many rows as observations in the first
        object and as many columns as observations in the second one. Each
        element (i, j) of the matrix is the inner product of the ith
        observation of the first object and the jth observation of the second
        one.

    Examples:
        Calculates the norm of a FDataGrid containing the functions y = 1
        and y = x defined in the interval [0,1].

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0,1,1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x)), x] ,x)
        >>> skfda.misc.metrics.lp_norm(fd, p=2).round(2)
        array([ 1.  ,  0.58])

        As the norm with ``p=2`` is a common choice, one can use ``l2_norm``
        directly:

        >>> skfda.misc.metrics.l2_norm(fd).round(2)
        array([ 1.  ,  0.58])

        The lp norm is only defined if p >= 1.

        >>> skfda.misc.metrics.lp_norm(fd, p=0.5)
        Traceback (most recent call last):
            ....
        ValueError: p (=0.5) must be equal or greater than 1.

    See also:
        :class:`LpNorm`

    """
    return LpNorm(p=p, vector_norm=vector_norm)(vector)

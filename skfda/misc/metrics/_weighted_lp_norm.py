"""Implementation of Lp norms."""

import math
from builtins import isinstance
from typing import Union, Callable, List

import numpy as np
import scipy.integrate
from typing_extensions import Final

from ...representation import FData, FDataBasis, FDataGrid
from ...typing._metric import Norm
from ...typing._numpy import NDArrayFloat


class WeightedLpNorm:
    r"""
    Norm of all the observations in a FDataGrid object.

    For each observation :math:`\mathbf{X}` the Lp norm is defined as:

    .. math::
        \| \mathbf{X} \| = \left( \int_D \| \mathbf{X} \|^p dt \right)^{
        \frac{1}{p}}

    Where :math:`\Omega` is the :term:`domain` over which the functions are defined.

    The integral is approximated using Simpson's rule.

    In general, if :math:`\mathbf{X}` is a multivariate function :math:`(X^{(1)}, ..., X^{(D)})`, and
    :math:`\Omega \subset \mathbb{R}^n`, it is applied the following generalization
    of the Lp norm.

    .. math::
        \| \mathbf{X} \| = \left( \int_\Omega \| \mathbf{X} \|_{*}^p dt \right)^{
        \frac{1}{p}}

    Where :math:`\| \cdot \|_*` denotes a vectorial norm. See
    :func:`vectorial_norm` to more information.

    For example, if :math:`\mathbf{X}: \mathbb{R}^2 \rightarrow \mathbb{R}^2`, and
    :math:`\| \cdot \|_*` is the euclidean norm
    :math:`\| (t,s) \|_* = \sqrt{t^2 + s^2}`, the lp norm applied is

    .. math::
        \| \mathbf{X} \| = \left( \int \int_\Omega \left ( \sqrt{ \| X^{(1)}(t,s)
        \|^2 + \| X^{(2)}(t,s) \|^2 } \right )^p dtds \right)^{
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
        >>> norm = skfda.misc.metrics.WeightedLpNorm(2)
        >>> norm(fd).round(2)
        array([ 1.  ,  0.58])

        As the norm with `p=2` is a common choice, one can use `l2_norm`
        directly:

        >>> skfda.misc.metrics.l2_norm(fd).round(2)
        array([ 1.  ,  0.58])

        The lp norm is only defined if p >= 1.

        >>> norm = skfda.misc.metrics.WeightedLpNorm(0.5)
        Traceback (most recent call last):
            ....
        ValueError: p (=0.5) must be equal or greater than 1.

    """

    def __init__(
        self,
        p: float,
        pointwise_weights: Union[NDArrayFloat, None] = None,
        weights: Union[
            List[Callable[[NDArrayFloat], NDArrayFloat]],
            None,
        ] = None,
    ) -> None:

        # Checks that the lp normed is well defined
        if not np.isinf(p) and p < 1:
            raise ValueError(f"p (={p}) must be equal or greater than 1.")

        self.p = p
        self.lp_weight = lp_weight
        self.filter = filter

    def __repr__(self) -> str:
        return f"{type(self).__name__}(" f"p={self.p}, measure={self.lp_weight}"

    def __call__(self, vector: Union[NDArrayFloat, FData]) -> NDArrayFloat:
        """Compute the Lp norm of a functional data object."""
        from ...misc import weighted_inner_product

        lp_weight = self.lp_weight
        filter = self.filter
        if isinstance(vector, np.ndarray):
            if isinstance(lp_weight, (float, int)):
                vector = vector * lp_weight
                return np.linalg.norm(  # type: ignore[no-any-return]
                    vector,
                    ord=self.p,
                    axis=-1,
                )

        # Special case, the inner product is heavily optimized: TODO: Is it insteresting to optimize the inner product with weights and ...
        """ if self.p == 2:
            return np.sqrt(weighted_inner_product(vector, vector, lp_weight)) """

        D = vector.dim_codomain

        if D == 1:
            lp_weight = lp_weight if lp_weight else 1
            filter = filter if filter else lambda x: np.ones_like(x)
        else:
            lp_weight = lp_weight if lp_weight else np.ones(D)

            aux = np.ones(D)
            aux[: len(lp_weight)] = lp_weight
            lp_weight = aux

            filter = filter if filter else [lambda x: 1.0 for _ in range(D)]

        if isinstance(vector, FDataBasis):
            if self.p != 2:
                raise NotImplementedError

            start, end = vector.domain_range[0]
            if D == 1:
                X = lambda x: (filter(x) * lp_weight) * np.power(
                    np.abs(vector(x)), self.p
                )
            else:
                X = lambda x: sum(
                    [
                        lp_weight[d]
                        * filter[d](x)
                        * np.power(np.abs(vector[d](x)), self.p)
                        for d in range(D)
                    ]
                )

            integral = scipy.integrate.quad_vec(X, start, end)
            res = np.sqrt(integral[0]).flatten()

        elif isinstance(vector, FDataGrid):
            data_matrix = vector.data_matrix

            if np.isinf(self.p):
                modified_matrix = (
                    data_matrix
                    * np.stack([f(vector.grid_points[0]) for f in filter], axis=-1)
                    * lp_weight
                )
                res = np.max(
                    modified_matrix,
                    axis=tuple(range(1, data_matrix.ndim)),
                )

            else:
                modified_matrix = (
                    data_matrix**self.p
                    * np.stack([f(vector.grid_points[0]) for f in filter], axis=-1)
                    * lp_weight
                )
                integrand = vector.copy(
                    data_matrix=modified_matrix, coordinate_names=(None,)
                )
            # Computes the norm, approximating the integral with Simpson's
            # rule.
            res = integrand.integrate().ravel() ** (1 / self.p)
        else:
            raise NotImplementedError(
                f"WeightedLpNorm not implemented for type {type(vector)}",
            )

        if len(res) == 1:
            return res[0]  # type: ignore[no-any-return]

        return res  # type: ignore[no-any-return]


def weighted_lp_norm(
    vector: Union[NDArrayFloat, FData],
    *,
    p: float,
    lp_weight: Union[Callable[[NDArrayFloat], NDArrayFloat], NDArrayFloat, None] = None,
) -> NDArrayFloat:
    r"""Calculate the norm of all the observations in a FDataGrid object.

    For each observation :math:`\mathbf{X}` the Lp norm is defined as:

    .. math::
        \|`\mathbf{X}` \| = \left( \int_\Omega \|`\mathbf{X}` \|^p dt \right)^{
        \frac{1}{p}}

    Where :math:`\Omega` is the :term:`domain` over which the functions are defined.

    The integral is approximated using Simpson's rule.

    In general, if :math:`\mathbf{X}` is a multivariate function :math:`(X^{(1)}, ..., X^{(D)})`, and
    :math:`\Omega \subset \mathbb{R}^n`, it is applied the following generalization
    of the Lp norm.

    .. math::
        \| \mathbf{X} \| = \left( \int_\Omega \| \mathbf{X} \|_{*}^p dt \right)^{
        \frac{1}{p}}

    Where :math:`\| \cdot \|_*` denotes a vectorial norm. See
    :func:`vectorial_norm` to more information.

    For example, if :math:`\mathbf{X}: \mathbb{R}^2 \rightarrow \mathbb{R}^2`, and
    :math:`\| \cdot \|_*` is the euclidean norm
    :math:`\| (t,s) \|_* = \sqrt{t^2 + s^2}`, the lp norm applied is

    .. math::
        \| \mathbf{X} \| = \left( \int \int_\Omega \left ( \sqrt{ \| X^{(1)}(t,s)
        \|^2 + \| X^{(2)}(t,s) \|^2 } \right )^p dtds \right)^{
        \frac{1}{p}}

    Note:
        This function is a wrapper of :class:`WeightedLpNorm`, available only for
        convenience. As the parameter ``p`` is mandatory, it cannot be used
        where a fully-defined norm is required: use an instance of
        :class:`WeightedLpNorm` in those cases.

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
        >>> skfda.misc.metrics.weighted_lp_norm(fd, p=2).round(2)
        array([ 1.  ,  0.58])

        As the norm with ``p=2`` is a common choice, one can use ``l2_norm``
        directly:

        >>> skfda.misc.metrics.l2_norm(fd).round(2)
        array([ 1.  ,  0.58])

        The lp norm is only defined if p >= 1.

        >>> skfda.misc.metrics.weighted_lp_norm(fd, p=0.5)
        Traceback (most recent call last):
            ....
        ValueError: p (=0.5) must be equal or greater than 1.

    See also:
        :class:`WeightedLpNorm`

    """
    return WeightedLpNorm(p=p, lp_weight=lp_weight)(vector)

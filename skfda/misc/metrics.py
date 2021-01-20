"""Norms and metrics for functional data objects."""
import math
from abc import abstractmethod
from builtins import isinstance
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import multimethod
import numpy as np
import scipy.integrate
from typing_extensions import Protocol

from .._utils import _pairwise_symmetric
from ..preprocessing.registration import ElasticRegistration, normalize_warping
from ..preprocessing.registration._warping import _normalize_scale
from ..preprocessing.registration.elastic import SRSF
from ..representation import FData, FDataBasis, FDataGrid
from ..representation._typing import Vector

T = TypeVar("T", bound=FData)
VectorType = TypeVar("VectorType", contravariant=True, bound=Vector)
MetricElementType = TypeVar("MetricElementType", contravariant=True)


class Norm(Protocol[VectorType]):
    """Protocol for a norm of a vector."""

    @abstractmethod
    def __call__(self, __vector: VectorType) -> np.ndarray:  # noqa: WPS112
        """Compute the norm of a vector."""


class Metric(Protocol[MetricElementType]):
    """Protocol for a metric between two elements of a metric space."""

    @abstractmethod
    def __call__(
        self,
        __e1: MetricElementType,  # noqa: WPS112
        __e2: MetricElementType,  # noqa: WPS112
    ) -> np.ndarray:
        """Compute the norm of a vector."""


def _check_compatible(fdata1: T, fdata2: T) -> None:

    if isinstance(fdata1, FData) and isinstance(fdata2, FData):
        if (
            fdata2.dim_codomain != fdata1.dim_codomain
            or fdata2.dim_domain != fdata1.dim_domain
        ):
            raise ValueError("Objects should have the same dimensions")

        if not np.array_equal(fdata1.domain_range, fdata2.domain_range):
            raise ValueError("Domain ranges for both objects must be equal")


def _cast_to_grid(
    fdata1: FData,
    fdata2: FData,
    eval_points: np.ndarray = None,
    _check: bool = True,
) -> Tuple[FDataGrid, FDataGrid]:
    """Convert fdata1 and fdata2 to FDatagrid.

    Checks if the fdatas passed as argument are unidimensional and compatible
    and converts them to FDatagrid to compute their distances.

    Args:
        fdata1: First functional object.
        fdata2: Second functional object.
        eval_points: Evaluation points.

    Returns:
        Tuple with two :obj:`FDataGrid` with the same grid points.
    """
    # Dont perform any check
    if not _check:
        return fdata1, fdata2

    _check_compatible(fdata1, fdata2)

    # Case new evaluation points specified
    if eval_points is not None:  # noqa: WPS223
        fdata1 = fdata1.to_grid(eval_points)
        fdata2 = fdata2.to_grid(eval_points)

    elif not isinstance(fdata1, FDataGrid) and isinstance(fdata2, FDataGrid):
        fdata1 = fdata1.to_grid(fdata2.grid_points[0])

    elif not isinstance(fdata2, FDataGrid) and isinstance(fdata1, FDataGrid):
        fdata2 = fdata2.to_grid(fdata1.grid_points[0])

    elif (
        not isinstance(fdata1, FDataGrid)
        and not isinstance(fdata2, FDataGrid)
    ):
        domain = fdata1.domain_range[0]
        grid_points = np.linspace(*domain)
        fdata1 = fdata1.to_grid(grid_points)
        fdata2 = fdata2.to_grid(grid_points)

    elif not np.array_equal(
        fdata1.grid_points,
        fdata2.grid_points,
    ):
        raise ValueError(
            "Grid points for both objects must be equal or"
            "a new list evaluation points must be specified",
        )

    return fdata1, fdata2


class LpNorm(Norm[FData]):
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
        vector_norm: Union[Norm[np.ndarray], float, None] = None,
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

    def __call__(self, fdata: FData) -> np.ndarray:
        """Compute the Lp norm of a functional data object."""
        from ..misc import inner_product

        vector_norm = self.vector_norm

        if vector_norm is None:
            vector_norm = self.p

        # Special case, the inner product is heavily optimized
        if self.p == vector_norm == 2:
            return np.sqrt(inner_product(fdata, fdata))

        if isinstance(fdata, FDataBasis):
            if self.p != 2:
                raise NotImplementedError

            start, end = fdata.domain_range[0]
            integral = scipy.integrate.quad_vec(
                lambda x: np.power(np.abs(fdata(x)), self.p),
                start,
                end,
            )
            res = np.sqrt(integral[0]).flatten()

        else:
            data_matrix = fdata.data_matrix
            original_shape = data_matrix.shape
            data_matrix = data_matrix.reshape(-1, original_shape[-1])

            data_matrix = (np.linalg.norm(
                fdata.data_matrix,
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

            elif fdata.dim_domain == 1:

                # Computes the norm, approximating the integral with Simpson's
                # rule.
                res = scipy.integrate.simps(
                    data_matrix[..., 0] ** self.p,
                    x=fdata.grid_points,
                ) ** (1 / self.p)

            else:
                # Needed to perform surface integration
                return NotImplemented

        if len(res) == 1:
            return res[0]

        return res


l1_norm = LpNorm(1)
l2_norm = LpNorm(2)
linf_norm = LpNorm(math.inf)


def lp_norm(
    fdata: FData,
    *,
    p: float,
    vector_norm: Union[Norm[np.ndarray], float, None] = None,
) -> np.ndarray:
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
        fdata: FData object.
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
    return LpNorm(p=p, vector_norm=vector_norm)(fdata)


class NormInducedMetric(Metric[VectorType]):
    r"""
    Metric induced by a norm.

    Given a norm :math:`\| \cdot \|: X \rightarrow \mathbb{R}`,
    returns the metric :math:`d: X \times X \rightarrow \mathbb{R}` induced
    by the norm:

    .. math::
        d(f,g) = \|f - g\|

    Args:
        norm: Norm used to induce the metric.

    Examples:
        Computes the :math:`\mathbb{L}^2` distance between an object containing
        functional data corresponding to the function :math:`y(x) = x` defined
        over the interval [0, 1] and another one containing data of the
        function :math:`y(x) = x/2`.

        Firstly we create the functional data.

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([x], x)
        >>> fd2 = skfda.FDataGrid([x/2], x)

        To construct the :math:`\mathbb{L}^2` distance it is used the
        :math:`\mathbb{L}^2` norm wich it is used to compute the distance.

        >>> l2_distance = skfda.misc.metrics.NormInducedMetric(l2_norm)
        >>> d = l2_distance(fd, fd2)
        >>> float('%.3f'% d)
        0.289

    """

    def __init__(self, norm: Norm[VectorType]):
        self.norm = norm

    def __call__(self, elem1: VectorType, elem2: VectorType) -> np.ndarray:
        """Compute the induced norm between two vectors."""
        return self.norm(elem1 - elem2)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(norm={self.norm})"


@multimethod.multidispatch
def pairwise_metric_optimization(
    metric: Any,
    elem1: Any,
    elem2: Optional[Any],
) -> np.ndarray:
    r"""
    Optimized computation of a pairwise metric.

    This is a generic function that can be subclassed for different
    combinations of metric and operators in order to provide a more
    efficient implementation for the pairwise metric matrix.
    """
    return NotImplemented


class PairwiseMetric(Generic[MetricElementType]):
    r"""Pairwise metric function.

    Computes a given metric pairwise. The matrix returned by the pairwise
    metric is a matrix with as many rows as observations in the first object
    and as many columns as observations in the second one. Each element
    (i, j) of the matrix is the distance between the ith observation of the
    first object and the jth observation of the second one.

    Args:
        metric: Metric between two elements of a metric
            space.

    """

    def __init__(
        self,
        metric: Metric[MetricElementType],
    ):
        self.metric = metric

    def __call__(
        self,
        elem1: MetricElementType,
        elem2: Optional[MetricElementType] = None,
    ) -> np.ndarray:
        """Evaluate the pairwise metric."""
        optimized = pairwise_metric_optimization(self.metric, elem1, elem2)

        return (
            _pairwise_symmetric(self.metric, elem1, elem2)
            if optimized is NotImplemented
            else optimized
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metric={self.metric})"


class LpDistance(NormInducedMetric[FData]):
    r"""Lp distance for FDataGrid objects.

    Calculates the distance between two functional objects.

    For each pair of observations f and g the distance between them is defined
    as:

    .. math::
        d(f, g) = d(g, f) = \| f - g \|_p

    where :math:`\| {}\cdot{} \|_p` denotes the :func:`Lp norm <lp_norm>`.

    The objects ``l1_distance``, ``l2_distance`` and ``linf_distance`` are
    instances of this class with commonly used values of ``p``, namely 1, 2 and
    infinity.

    Args:
        p: p of the lp norm. Must be greater or equal
            than 1. If ``p=math.inf`` it is used the L infinity metric.
            Defaults to 2.
        vector_norm: vector norm to apply. If it is a float, is the index of
            the multivariate lp norm. Defaults to the same as ``p``.

    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array 2x2 with the computed
        l2 distance between every pair of functions.

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x))], x)
        >>> fd2 =  skfda.FDataGrid([np.zeros(len(x))], x)
        >>>
        >>> distance = skfda.misc.metrics.LpDistance(p=2)
        >>> distance(fd, fd2).round(2)
        array([ 1.])


        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = np.linspace(0, 2, 1001)
        >>> fd2 =  FDataGrid([np.zeros(len(x)), x/2 + 0.5], x)
        >>> distance = skfda.misc.metrics.LpDistance(p=2)
        >>> distance(fd, fd2)
        Traceback (most recent call last):
            ...
        ValueError: ...

    """  # noqa: P102

    def __init__(
        self,
        p: float,
        vector_norm: Union[Norm[np.ndarray], float, None] = None,
    ) -> None:

        self.p = p
        self.vector_norm = vector_norm
        norm = LpNorm(p=p, vector_norm=vector_norm)

        super().__init__(norm)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"p={self.p}, vector_norm={self.vector_norm})"
        )


l1_distance = LpDistance(p=1)
l2_distance = LpDistance(p=2)
linf_distance = LpDistance(p=math.inf)


@pairwise_metric_optimization.register
def _pairwise_metric_optimization_lp_fdata(
    metric: LpDistance,
    elem1: FData,
    elem2: Optional[FData],
) -> np.ndarray:
    from ..misc import inner_product, inner_product_matrix

    vector_norm = metric.vector_norm

    if vector_norm is None:
        vector_norm = metric.p

    # Special case, the inner product is heavily optimized
    if metric.p == vector_norm == 2:
        diag1 = inner_product(elem1, elem1)
        diag2 = diag1 if elem2 is None else inner_product(elem2, elem2)

        if elem2 is None:
            elem2 = elem1

        inner_matrix = inner_product_matrix(elem1, elem2)

        distance_matrix_sqr = (
            -2 * inner_matrix
            + diag1[:, np.newaxis]
            + diag2[np.newaxis, :]
        )

        np.clip(
            distance_matrix_sqr,
            a_min=0,
            a_max=None,
            out=distance_matrix_sqr,
        )

        return np.sqrt(distance_matrix_sqr)

    return NotImplemented


def lp_distance(
    fdata1: T,
    fdata2: T,
    *,
    p: float,
    vector_norm: Union[Norm[np.ndarray], float, None] = None,
) -> np.ndarray:
    r"""
    Lp distance for FDataGrid objects.

    Calculates the distance between two functional objects.

    For each pair of observations f and g the distance between them is defined
    as:

    .. math::
        d(f, g) = d(g, f) = \| f - g \|_p

    where :math:`\| {}\cdot{} \|_p` denotes the :func:`Lp norm <lp_norm>`.

    Note:
        This function is a wrapper of :class:`LpDistance`, available only for
        convenience. As the parameter ``p`` is mandatory, it cannot be used
        where a fully-defined metric is required: use an instance of
        :class:`LpDistance` in those cases.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        p: p of the lp norm. Must be greater or equal
            than 1. If ``p=math.inf`` it is used the L infinity metric.
            Defaults to 2.
        vector_norm: vector norm to apply. If it is a float, is the index of
            the multivariate lp norm. Defaults to the same as ``p``.

    Returns:
        Numpy vector where the i-th coordinate has the distance between the
        i-th element of the first object and the i-th element of the second
        one.

    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array 2x2 with the computed
        l2 distance between every pair of functions.

        >>> import skfda
        >>> import numpy as np
        >>>
        >>> x = np.linspace(0, 1, 1001)
        >>> fd = skfda.FDataGrid([np.ones(len(x))], x)
        >>> fd2 =  skfda.FDataGrid([np.zeros(len(x))], x)
        >>>
        >>> skfda.misc.metrics.lp_distance(fd, fd2, p=2).round(2)
        array([ 1.])

        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = np.linspace(0, 2, 1001)
        >>> fd2 =  FDataGrid([np.zeros(len(x)), x/2 + 0.5], x)
        >>> skfda.misc.metrics.lp_distance(fd, fd2, p=2)
        Traceback (most recent call last):
            ...
        ValueError: ...

    See also:
        :class:`~skfda.misc.metrics.LpDistance`

    """  # noqa: P102
    return LpDistance(p=p, vector_norm=vector_norm)(fdata1, fdata2)


def fisher_rao_distance(
    fdata1: T,
    fdata2: T,
    *,
    eval_points: np.ndarray = None,
    _check: bool = True,
) -> np.ndarray:
    r"""Compute the Fisher-Rao distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`q_i` and :math:`q_j` be the corresponding SRSF
    (see :class:`SRSF`), the fisher rao distance is defined as

    .. math::
        d_{FR}(f_i, f_j) = \| q_i - q_j \|_2 =
        \left ( \int_0^1 sgn(\dot{f_i}(t))\sqrt{|\dot{f_i}(t)|} -
        sgn(\dot{f_j}(t))\sqrt{|\dot{f_j}(t)|} dt \right )^{\frac{1}{2}}

    If the observations are distributions of random variables the distance will
    match with the usual fisher-rao distance in non-parametric form for
    probability distributions [S11-2]_.

    If the observations are defined in a :term:`domain` different than (0,1)
    their domains are normalized to this interval with an affine
    transformation.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        eval_points: Array with points of evaluation.

    Returns:
        Fisher rao distance.

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        .. [S11-2] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Function Representation and
            Metric* (pp. 5-7). arXiv:1103.3817v2.

    """
    fdata1, fdata2 = _cast_to_grid(
        fdata1,
        fdata2,
        eval_points=eval_points,
        _check=_check,
    )

    # Both should have the same grid points
    eval_points_normalized = _normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )
    fdata2 = fdata2.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )

    srsf = SRSF(initial_value=0)
    fdata1_srsf = srsf.fit_transform(fdata1)
    fdata2_srsf = srsf.transform(fdata2)

    # Return the L2 distance of the SRSF
    return l2_distance(fdata1_srsf, fdata2_srsf)


def amplitude_distance(
    fdata1: T,
    fdata2: T,
    *,
    lam: float = 0.0,
    eval_points: np.ndarray = None,
    _check: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    r"""Compute the amplitude distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`q_i` and :math:`q_j` be the corresponding SRSF
    (see :class:`SRSF`), the amplitude distance is defined as

    .. math::
        d_{A}(f_i, f_j)=min_{\gamma \in \Gamma}d_{FR}(f_i \circ \gamma,f_j)

    A penalty term could be added to restrict the ammount of elasticity in the
    alignment used.

    .. math::
        d_{\lambda}^2(f_i, f_j) =min_{\gamma \in \Gamma} \{
        d_{FR}^2(f_i \circ \gamma, f_j) + \lambda \mathcal{R}(\gamma) \}


    Where :math:`d_{FR}` is the Fisher-Rao distance and the penalty term is
    given by

    .. math::
        \mathcal{R}(\gamma) = \|\sqrt{\dot{\gamma}}- 1 \|_{\mathbb{L}^2}^2

    See [SK16-4-10-1]_ for a detailed explanation.

    If the observations are defined in a :term:`domain` different than (0,1)
    their domains are normalized to this interval with an affine
    transformation.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        lam: Penalty term to restric the elasticity.
        eval_points: Array with points of evaluation.
        kwargs: Name arguments to be passed to
            :func:`elastic_registration_warping`.

    Returns:
        Elastic distance.

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        ..  [SK16-4-10-1] Srivastava, Anuj & Klassen, Eric P. (2016).
            Functional and shape data analysis. In *Amplitude Space and a
            Metric Structure* (pp. 107-109). Springer.
    """
    fdata1, fdata2 = _cast_to_grid(
        fdata1,
        fdata2,
        eval_points=eval_points,
        _check=_check,
    )

    # Both should have the same grid points
    eval_points_normalized = _normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )
    fdata2 = fdata2.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )

    elastic_registration = ElasticRegistration(
        template=fdata2,
        penalty=lam,
        output_points=eval_points_normalized,
        **kwargs,
    )

    fdata1_reg = elastic_registration.fit_transform(fdata1)

    srsf = SRSF(initial_value=0)
    fdata1_reg_srsf = srsf.fit_transform(fdata1_reg)
    fdata2_srsf = srsf.transform(fdata2)
    distance = l2_distance(fdata1_reg_srsf, fdata2_srsf)

    if lam != 0.0:
        # L2 norm || sqrt(Dh) - 1 ||^2
        warping_deriv = elastic_registration.warping_.derivative()
        penalty = warping_deriv(eval_points_normalized)[0, ..., 0]
        penalty = np.sqrt(penalty, out=penalty)
        penalty -= 1
        penalty = np.square(penalty, out=penalty)
        penalty = scipy.integrate.simps(penalty, x=eval_points_normalized)

        distance = np.sqrt(distance**2 + lam * penalty)

    return distance


def phase_distance(
    fdata1: T,
    fdata2: T,
    *,
    lam: float = 0.0,
    eval_points: np.ndarray = None,
    _check: bool = True,
) -> np.ndarray:
    r"""Compute the phase distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`\gamma_{ij}` the corresponding warping used in the elastic
    registration to align :math:`f_i` to :math:`f_j` (see
    :func:`elastic_registration`). The phase distance between :math:`f_i`
    and :math:`f_j` is defined as

    .. math::
        d_{P}(f_i, f_j) = d_{FR}(\gamma_{ij}, \gamma_{id}) =
        arcos \left ( \int_0^1 \sqrt {\dot \gamma_{ij}(t)} dt \right )

    See [SK16-4-10-2]_ for a detailed explanation.

    If the observations are defined in a :term:`domain` different than (0,1)
    their domains are normalized to this interval with an affine
    transformation.

    Args:
        fdata1: First FData object.
        fdata2: Second FData object.
        lam: Penalty term to restric the elasticity.
        eval_points (array_like, optional): Array with points of evaluation.

    Returns:
        Phase distance between the objects.

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        ..  [SK16-4-10-2] Srivastava, Anuj & Klassen, Eric P. (2016).
            Functional and shape data analysis. In *Phase Space and a Metric
            Structure* (pp. 109-111). Springer.
    """
    fdata1, fdata2 = _cast_to_grid(
        fdata1,
        fdata2,
        eval_points=eval_points,
        _check=_check,
    )

    # Rescale in the interval (0,1)
    eval_points_normalized = _normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )
    fdata2 = fdata2.copy(
        grid_points=eval_points_normalized,
        domain_range=(0, 1),
    )

    elastic_registration = ElasticRegistration(
        penalty=lam,
        template=fdata2,
        output_points=eval_points_normalized,
    )

    elastic_registration.fit_transform(fdata1)

    warping_deriv = elastic_registration.warping_.derivative()
    derivative_warping = warping_deriv(eval_points_normalized)[0, ..., 0]

    derivative_warping = np.sqrt(derivative_warping, out=derivative_warping)

    d = scipy.integrate.simps(derivative_warping, x=eval_points_normalized)
    d = np.clip(d, -1, 1)

    return np.arccos(d)


def warping_distance(
    warping1: T,
    warping2: T,
    *,
    eval_points: np.ndarray = None,
    _check: bool = True,
) -> np.ndarray:
    r"""Compute the distance between warpings functions.

    Let :math:`\gamma_i` and :math:`\gamma_j` be two warpings, defined in
    :math:`\gamma_i:[a,b] \rightarrow [a,b]`. The distance in the
    space of warping functions, :math:`\Gamma`, with the riemannian metric
    given by the fisher-rao inner product can be computed using the structure
    of hilbert sphere in their srsf's.

    .. math::
        d_{\Gamma}(\gamma_i, \gamma_j) = cos^{-1} \left ( \int_0^1
        \sqrt{\dot \gamma_i(t)\dot \gamma_j(t)}dt \right )

    See [SK16-4-11-2]_ for a detailed explanation.

    If the warpings are not defined in [0,1], an affine transformation is maked
    to change the :term:`domain`.

    Args:
        warping1: First warping.
        warping2: Second warping.
        eval_points: Array with points of evaluation.

    Returns:
        Distance between warpings:

    Raises:
        ValueError: If the objects are not unidimensional.

    References:
        ..  [SK16-4-11-2] Srivastava, Anuj & Klassen, Eric P. (2016).
            Functional and shape data analysis. In *Probability Density
            Functions* (pp. 113-117). Springer.

    """
    warping1, warping2 = _cast_to_grid(
        warping1,
        warping2,
        eval_points=eval_points,
        _check=_check,
    )

    # Normalization of warping to (0,1)x(0,1)
    warping1 = normalize_warping(warping1, (0, 1))
    warping2 = normalize_warping(warping2, (0, 1))

    warping1_data = warping1.derivative().data_matrix[0, ..., 0]
    warping2_data = warping2.derivative().data_matrix[0, ..., 0]

    # Derivative approximations can have negatives, specially in the
    # borders.
    warping1_data[warping1_data < 0] = 0
    warping2_data[warping2_data < 0] = 0

    # In this case the srsf is the sqrt(gamma')
    srsf_warping1 = np.sqrt(warping1_data, out=warping1_data)
    srsf_warping2 = np.sqrt(warping2_data, out=warping2_data)

    product = np.multiply(srsf_warping1, srsf_warping2, out=srsf_warping1)

    d = scipy.integrate.simps(product, x=warping1.grid_points[0])
    d = np.clip(d, -1, 1)

    return np.arccos(d)

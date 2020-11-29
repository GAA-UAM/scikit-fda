from builtins import isinstance

import scipy.integrate

import numpy as np

from .._utils import _pairwise_commutative
from ..preprocessing.registration import normalize_warping, ElasticRegistration
from ..preprocessing.registration._warping import _normalize_scale
from ..preprocessing.registration.elastic import SRSF
from ..representation import FData, FDataGrid, FDataBasis


def _check_compatible(fdata1, fdata2):

    if isinstance(fdata1, FData) and isinstance(fdata2, FData):
        if (fdata2.dim_codomain != fdata1.dim_codomain or
                fdata2.dim_domain != fdata1.dim_domain):
            raise ValueError("Objects should have the same dimensions")

        if not np.array_equal(fdata1.domain_range, fdata2.domain_range):
            raise ValueError("Domain ranges for both objects must be equal")


def _cast_to_grid(fdata1, fdata2, eval_points=None, _check=True, **kwargs):
    """Convert fdata1 and fdata2 to FDatagrid.

    Checks if the fdatas passed as argument are unidimensional and compatible
    and converts them to FDatagrid to compute their distances.

    Args:
        fdata1: (:obj:`FData`): First functional object.
        fdata2: (:obj:`FData`): Second functional object.

    Returns:
        tuple: Tuple with two :obj:`FDataGrid` with the same grid points.
    """
    # Dont perform any check
    if not _check:
        return fdata1, fdata2

    _check_compatible(fdata1, fdata2)

    # Case new evaluation points specified
    if eval_points is not None:
        fdata1 = fdata1.to_grid(eval_points)
        fdata2 = fdata2.to_grid(eval_points)

    elif not isinstance(fdata1, FDataGrid) and isinstance(fdata2, FDataGrid):
        fdata1 = fdata1.to_grid(fdata2.grid_points[0])

    elif not isinstance(fdata2, FDataGrid) and isinstance(fdata1, FDataGrid):
        fdata2 = fdata2.to_grid(fdata1.grid_points[0])

    elif (not isinstance(fdata1, FDataGrid) and
          not isinstance(fdata2, FDataGrid)):
        domain = fdata1.domain_range[0]
        grid_points = np.linspace(*domain)
        fdata1 = fdata1.to_grid(grid_points)
        fdata2 = fdata2.to_grid(grid_points)

    elif not np.array_equal(fdata1.grid_points,
                            fdata2.grid_points):
        raise ValueError("Grid points for both objects must be equal or"
                         "a new list evaluation points must be specified")

    return fdata1, fdata2


def distance_from_norm(norm, **kwargs):
    r"""Return the distance induced by a norm.

    Given a norm :math:`\| \cdot \|: X \rightarrow \mathbb{R}`,
    returns the distance :math:`d: X \times X \rightarrow \mathbb{R}` induced
    by the norm:

    .. math::
        d(f,g) = \|f - g\|

    Args:
        norm (:obj:`Function`): Norm function `norm(fdata, **kwargs)`.
        **kwargs (dict, optional): Named parameters to be passed to the norm
            function.

    Returns:
        :obj:`Function`: Distance function `norm_distance(fdata1, fdata2)`.

    Examples:
        Computes the :math:`\mathbb{L}^2` distance between an object containing
        functional data corresponding to the function :math:`y(x) = x` defined
        over the interval [0, 1] and another one containing data of the
        function :math:`y(x) = x/2`.

        Firstly we create the functional data.

        >>> x = np.linspace(0, 1, 1001)
        >>> fd = FDataGrid([x], x)
        >>> fd2 =  FDataGrid([x/2], x)

        To construct the :math:`\mathbb{L}^2` distance it is used the
        :math:`\mathbb{L}^2` norm wich it is used to compute the distance.

        >>> l2_distance = distance_from_norm(lp_norm, p=2)
        >>> d = l2_distance(fd, fd2)
        >>> float('%.3f'% d)
        0.289

    """
    def norm_distance(fdata1, fdata2):
        # Substract operation checks if objects are compatible
        return norm(fdata1 - fdata2, **kwargs)

    norm_distance.__name__ = f"{norm.__name__}_distance"

    return norm_distance


def pairwise_distance(distance, **kwargs):
    r"""Return a pairwise distance function for FData objects.

    Given a distance it returns the corresponding pairwise distance function.

    The returned pairwise distance function calculates the distance between
    all possible pairs consisting of one observation of the first FDataGrid
    object and one of the second one.

    The matrix returned by the pairwise distance is a matrix with as many rows
    as observations in the first object and as many columns as observations in
    the second one. Each element (i, j) of the matrix is the distance between
    the ith observation of the first object and the jth observation of the
    second one.

    Args:
        distance (:obj:`Function`): Distance functions between two functional
            objects `distance(fdata1, fdata2, **kwargs)`.
        **kwargs (:obj:`dict`, optional): parameters dictionary to be passed
            to the distance function.

    Returns:
        :obj:`Function`: Pairwise distance function, wich accepts two
            functional data objects and returns the pairwise distance matrix.
    """
    def pairwise(fdata1, fdata2=None):

        return _pairwise_commutative(distance, fdata1, fdata2)

    pairwise.__name__ = f"pairwise_{distance.__name__}"

    return pairwise


def lp_norm(fdata, p=2, p2=None):
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


    Args:
        fdata (FData): FData object.
        p (int, optional): p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.
        p2 (int, optional): p index of the vectorial norm applied in case of
            multivariate objects. Defaults to 2.

    Returns:
        numpy.darray: Matrix with as many rows as observations in the first
        object and as many columns as observations in the second one. Each
        element (i, j) of the matrix is the inner product of the ith
        observation of the first object and the jth observation of the second
        one.

    Examples:
        Calculates the norm of a FDataGrid containing the functions y = 1
        and y = x defined in the interval [0,1].


        >>> x = np.linspace(0,1,1001)
        >>> fd = FDataGrid([np.ones(len(x)), x] ,x)
        >>> lp_norm(fd).round(2)
        array([ 1.  ,  0.58])

        The lp norm is only defined if p >= 1.

        >>> lp_norm(fd, p = 0.5)
        Traceback (most recent call last):
            ....
        ValueError: p must be equal or greater than 1.

    """
    from ..misc import inner_product

    if p2 is None:
        p2 = p

    # Special case, the inner product is heavily optimized
    if p == p2 == 2:
        return np.sqrt(inner_product(fdata, fdata))

    # Checks that the lp normed is well defined
    if not (p == 'inf' or np.isinf(p)) and p < 1:
        raise ValueError(f"p must be equal or greater than 1.")

    if isinstance(fdata, FDataBasis):
        if fdata.dim_codomain > 1 or p != 2:
            raise NotImplementedError

        start, end = fdata.domain_range[0]
        integral = scipy.integrate.quad_vec(
            lambda x: np.power(np.abs(fdata(x)), p), start, end)
        res = np.sqrt(integral[0]).flatten()

    else:
        if fdata.dim_codomain > 1:
            if p2 == 'inf':
                p2 = np.inf
            data_matrix = np.linalg.norm(fdata.data_matrix, ord=p2, axis=-1,
                                         keepdims=True)
        else:
            data_matrix = np.abs(fdata.data_matrix)

        if p == 'inf' or np.isinf(p):

            if fdata.dim_domain == 1:
                res = np.max(data_matrix[..., 0], axis=1)
            else:
                res = np.array([np.max(observation)
                                for observation in data_matrix])

        elif fdata.dim_domain == 1:

            # Computes the norm, approximating the integral with Simpson's
            # rule.
            res = scipy.integrate.simps(data_matrix[..., 0] ** p,
                                        x=fdata.grid_points) ** (1 / p)

        else:
            # Needed to perform surface integration
            return NotImplemented

    if len(res) == 1:
        return res[0]

    return res


def lp_distance(fdata1, fdata2, p=2, p2=2, *, eval_points=None, _check=True):
    r"""Lp distance for FDataGrid objects.

    Calculates the distance between two functional objects.

    For each pair of observations f and g the distance between them is defined
    as:

    .. math::
        d(f, g) = d(g, f) = \| f - g \|_p

    where :math:`\| {}\cdot{} \|_p` denotes the :func:`Lp norm <lp_norm>`.

    Args:
        fdatagrid (FDataGrid): FDataGrid object.
        p (int, optional): p of the lp norm. Must be greater or equal
            than 1. If p='inf' or p=np.inf it is used the L infinity metric.
            Defaults to 2.
        p2 (int, optional): p index of the vectorial norm applied in case of
            multivariate objects. Defaults to 2. See :func:`lp_norm`.

    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array 2x2 with the computed
        l2 distance between every pair of functions.

        >>> x = np.linspace(0, 1, 1001)
        >>> fd = FDataGrid([np.ones(len(x))], x)
        >>> fd2 =  FDataGrid([np.zeros(len(x))], x)
        >>> lp_distance(fd, fd2).round(2)
        array([ 1.])


        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = np.linspace(0, 2, 1001)
        >>> fd2 =  FDataGrid([np.zeros(len(x)), x/2 + 0.5], x)
        >>> lp_distance(fd, fd2)
        Traceback (most recent call last):
            ....
        ValueError: ...

    See also:
        :func:`~skfda.misc.metrics.l1_distance
        :func:`~skfda.misc.metrics.l2_distance
        :func:`~skfda.misc.metrics.linf_distance

    """
    _check_compatible(fdata1, fdata2)

    return lp_norm(fdata1 - fdata2, p=p, p2=p2)


def l1_distance(fdata1, fdata2, *, eval_points=None, _check=True):
    r"""L1 distance for FDataGrid objects.

    Calculates the L1 distance between fdata1 and fdata2:
    .. math::
        d(fdata1, fdata2) =
            \left( \int_D \| fdata1(x)-fdata2(x) \| dx
            \right)

    See also:
        :func:`~skfda.misc.metrics.lp_distance
        :func:`~skfda.misc.metrics.l2_distance
        :func:`~skfda.misc.metrics.linf_distance
    """
    return lp_distance(fdata1, fdata2, p=1, p2=1,
                       eval_points=eval_points, _check=_check)


def l2_distance(fdata1, fdata2, *, eval_points=None, _check=True):
    r"""L2 distance for FDataGrid objects.

    Calculates the euclidean distance between fdata1 and fdata2:
    .. math::
        d(fdata1, fdata2) =
            \left( \int_D \| fdata1(x)-fdata2(x) \|^2 dx
            \right)^{\frac{1}{2}}

    See also:
        :func:`~skfda.misc.metrics.lp_distance
        :func:`~skfda.misc.metrics.l1_distance
        :func:`~skfda.misc.metrics.linf_distance
    """
    return lp_distance(fdata1, fdata2, p=2, p2=2,
                       eval_points=eval_points, _check=_check)


def linf_distance(fdata1, fdata2, *, eval_points=None, _check=True):
    r"""L_infinity distance for FDataGrid objects.

    Calculates the L_infinity distance between fdata1 and fdata2:
    .. math::
        d(fdata1, fdata2) \equiv \inf \{ C\ge 0 : |fdata1(x)-fdata2(x)|
                                                                \le C a.e. \}.

    See also:
        :func:`~skfda.misc.metrics.lp_distance
        :func:`~skfda.misc.metrics.l1_distance
        :func:`~skfda.misc.metrics.l2_distance
    """
    return lp_distance(fdata1, fdata2, p=np.inf, p2=np.inf,
                       eval_points=eval_points, _check=_check)


def fisher_rao_distance(fdata1, fdata2, *, eval_points=None, _check=True):
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
        fdata1 (FData): First FData object.
        fdata2 (FData): Second FData object.
        eval_points (array_like, optional): Array with points of evaluation.

    Returns:
        Fisher rao distance.

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        .. [S11-2] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Function Representation and
            Metric* (pp. 5-7). arXiv:1103.3817v2.

    """
    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points,
                                   _check=_check)

    # Both should have the same grid points
    eval_points_normalized = _normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(grid_points=eval_points_normalized,
                         domain_range=(0, 1))
    fdata2 = fdata2.copy(grid_points=eval_points_normalized,
                         domain_range=(0, 1))

    srsf = SRSF(initial_value=0)
    fdata1_srsf = srsf.fit_transform(fdata1)
    fdata2_srsf = srsf.transform(fdata2)

    # Return the L2 distance of the SRSF
    return lp_distance(fdata1_srsf, fdata2_srsf, p=2)


def amplitude_distance(fdata1, fdata2, *, lam=0., eval_points=None,
                       _check=True, **kwargs):
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
        fdata1 (FData): First FData object.
        fdata2 (FData): Second FData object.
        lam (float, optional): Penalty term to restric the elasticity.
        eval_points (array_like, optional): Array with points of evaluation.
        **kwargs (dict): Name arguments to be passed to
            :func:`elastic_registration_warping`.

    Returns:
        float: Elastic distance.

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        ..  [SK16-4-10-1] Srivastava, Anuj & Klassen, Eric P. (2016).
            Functional and shape data analysis. In *Amplitude Space and a
            Metric Structure* (pp. 107-109). Springer.
    """
    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points,
                                   _check=_check)

    # Both should have the same grid points
    eval_points_normalized = _normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(grid_points=eval_points_normalized,
                         domain_range=(0, 1))
    fdata2 = fdata2.copy(grid_points=eval_points_normalized,
                         domain_range=(0, 1))

    elastic_registration = ElasticRegistration(
        template=fdata2,
        penalty=lam,
        output_points=eval_points_normalized,
        **kwargs)

    fdata1_reg = elastic_registration.fit_transform(fdata1)

    srsf = SRSF(initial_value=0)
    fdata1_reg_srsf = srsf.fit_transform(fdata1_reg)
    fdata2_srsf = srsf.transform(fdata2)
    distance = lp_distance(fdata1_reg_srsf, fdata2_srsf)

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


def phase_distance(fdata1, fdata2, *, lam=0., eval_points=None, _check=True,
                   **kwargs):
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
        fdata1 (FData): First FData object.
        fdata2 (FData): Second FData object.
        lambda (float, optional): Penalty term to restric the elasticity.
        eval_points (array_like, optional): Array with points of evaluation.
        **kwargs (dict): Name arguments to be passed to
            :func:`elastic_registration_warping`.

    Returns:
        float: Phase distance between the objects.

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        ..  [SK16-4-10-2] Srivastava, Anuj & Klassen, Eric P. (2016).
            Functional and shape data analysis. In *Phase Space and a Metric
            Structure* (pp. 109-111). Springer.
    """
    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points,
                                   _check=_check)

    # Rescale in (0,1)
    eval_points_normalized = _normalize_scale(fdata1.grid_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(grid_points=eval_points_normalized,
                         domain_range=(0, 1))
    fdata2 = fdata2.copy(grid_points=eval_points_normalized,
                         domain_range=(0, 1))

    elastic_registration = ElasticRegistration(
        penalty=lam, template=fdata2,
        output_points=eval_points_normalized)

    elastic_registration.fit_transform(fdata1)

    warping_deriv = elastic_registration.warping_.derivative()
    derivative_warping = warping_deriv(eval_points_normalized)[0, ..., 0]

    derivative_warping = np.sqrt(derivative_warping, out=derivative_warping)

    d = scipy.integrate.simps(derivative_warping, x=eval_points_normalized)
    d = np.clip(d, -1, 1)

    return np.arccos(d)


def warping_distance(warping1, warping2, *, eval_points=None, _check=True):
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
        fdata1 (:obj:`FData`): First warping.
        fdata2 (:obj:`FData`): Second warping.
        eval_points (array_like, optional): Array with points of evaluation.

    Returns:
        float: Distance between warpings:

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        ..  [SK16-4-11-2] Srivastava, Anuj & Klassen, Eric P. (2016).
            Functional and shape data analysis. In *Probability Density
            Functions* (pp. 113-117). Springer.

    """
    warping1, warping2 = _cast_to_grid(warping1, warping2,
                                       eval_points=eval_points, _check=_check)

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

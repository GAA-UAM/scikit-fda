
import scipy.integrate
import numpy


from ..representation import FData
from ..representation import FDataGrid
from ..preprocessing.registration import (
    normalize_warping, _normalize_scale, to_srsf,
    elastic_registration_warping)


def _cast_to_grid(fdata1, fdata2, eval_points=None):
    """Checks if the fdatas passed as argument are unidimensional and compatible
    and converts them to FDatagrid to compute their distances.


    Args:
        fdata1: (:obj:`FData`): First functional object.
        fdata2: (:obj:`FData`): Second functional object.

    Returns:
        tuple: Tuple with two :obj:`FDataGrid` with the same sample points.
    """

    # To allow use numpy arrays internally
    if (not isinstance(fdata1, FData) and not isinstance(fdata2, FData)
        and eval_points is not None):
        fdata1 = FDataGrid([fdata1], sample_points=eval_points)
        fdata2 = FDataGrid([fdata1], sample_points=eval_points)

    # Checks dimension
    elif (fdata2.ndim_image != fdata1.ndim_image or
          fdata2.ndim_domain != fdata1.ndim_domain):
        raise ValueError("Objects should have the same dimensions")

    # Case different domain ranges
    elif not numpy.array_equal(fdata1.domain_range, fdata2.domain_range):
        raise ValueError("Domain ranges for both objects must be equal")

    # Case new evaluation points specified
    elif eval_points is not None:
        if not numpy.array_equal(eval_points, fdata1.sample_points[0]):
            fdata1 = fdata1.to_grid(eval_points)
        if not numpy.array_equal(eval_points, fdata2.sample_points[0]):
            fdata2 = fdata2.to_grid(eval_points)

    elif not isinstance(fdata1, FDataGrid) and isinstance(fdata2, FDataGrid):
        fdata1 = fdata1.to_grid(fdata2.eval_points)

    elif not isinstance(fdata2, FDataGrid) and isinstance(fdata1, FDataGrid):
        fdata2 = fdata2.to_grid(fdata1.eval_points)

    elif not isinstance(fdata1, FDataGrid) and not isinstance(fdata2, FDataGrid):
        fdata1 = fdata1.to_grid(eval_points)
        fdata2 = fdata2.to_grid(eval_points)

    elif not numpy.array_equal(fdata1.sample_points,
                                      fdata2.sample_points):
        raise ValueError("Sample points for both objects must be equal or"
                         "a new list evaluation points must be specified")

    return fdata1, fdata2

def vectorial_norm(fdatagrid, p=2):
    r"""Apply a vectorial norm to a multivariate function.

    Given a multivariate function :math:`f:\mathbb{R}^n\rightarrow \mathbb{R}^d`
    applies a vectorial norm :math:`\| \cdot \|` to produce a function
    :math:`\|f\|:\mathbb{R}^n\rightarrow \mathbb{R}`.

    For example, let :math:`f:\mathbb{R} \rightarrow \mathbb{R}^2` be
    :math:`f(t)=(f_1(t), f_2(t))` and :math:`\| \cdot \|_2` the euclidian norm.

    .. math::
        \|f\|_2(t) = \sqrt { |f_1(t)|^2 + |f_2(t)|^2 }

    In general if :math:`p \neq \pm \infty` and :math:`f:\mathbb{R}^n
    \rightarrow \mathbb{R}^d`

    .. math::
        \|f\|_p(x_1, ... x_n) = \left ( \sum_{k=1}^{d} |f_k(x_1, ..., x_n)|^p
        \right )^{(1/p)}

    Args:
        fdatagrid (:class:`FDatagrid`): Functional object to be transformed.
        p (int, optional): Exponent in the lp norm. If p is a number then
            it is applied sum(abs(x)**p)**(1./p), if p is inf then max(abs(x)),
            and if p is -inf it is applied min(abs(x)). See numpy.linalg.norm
            to more information. Defaults to 2.

    Returns:
        (:class:`FDatagrid`): FDatagrid with image dimension equal to 1.

    Examples:

        >>> from skfda.datasets import make_multimodal_samples
        >>> from skfda.misc.metrics import vectorial_norm

        First we will construct an example dataset with curves in
        :math:`\mathbb{R}^2`.

        >>> fd = make_multimodal_samples(ndim_image=2, random_state=0)
        >>> fd.ndim_image
        2

        We will apply the euclidean norm

        >>> fd = vectorial_norm(fd, p=2)
        >>> fd.ndim_image
        1

    """
    data_matrix = numpy.linalg.norm(fdatagrid.data_matrix, ord=p, axis=-1,
                                    keepdims=True)

    return fdatagrid.copy(data_matrix=data_matrix)


def distance_from_norm(norm, **kwargs):
    r"""Returns the distance induced by a norm.

    Given a norm :math:`\| \cdot \|: X \rightarrow \mathbb{R}`,
    returns the distance :math:`d: X \times X \rightarrow \mathbb{R}` induced by
    the norm:

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
        over the interval [0, 1] and another one containing data of the function
        :math:`y(x) = x/2`.

        Firstly we create the functional data.

        >>> x = numpy.linspace(0, 1, 1001)
        >>> fd = FDataGrid([x], x)
        >>> fd2 =  FDataGrid([x/2], x)

        To construct the :math:`\mathbb{L}^2` distance it is used the
        :math:`\mathbb{L}^2` norm wich it is used to compute the distance.

        >>> l2_distance = distance_from_norm(norm_lp, p=2)
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
    r"""Return pairwise distance for FDataGrid objects.

    Given a distance returns the corresponding pairwise distance function.

    The pairwise distance calculates the distance between all possible pairs of
    one sample of the first FDataGrid object and one of the second one.

    The matrix returned by the pairwise distance is a matrix with as many rows
    as samples in the first object and as many columns as samples in the second
    one. Each element (i, j) of the matrix is the distance between the ith
    sample of the first object and the jth sample of the second one.

    Args:
        distance (:obj:`Function`): Distance functions between two functional
            objects `distance(fdata1, fdata2, **kwargs)`.
        **kwargs (:obj:`dict`, optional): parameters dictionary to be passed
            to the distance function.

    Returns:
        :obj:`Function`: Pairwise distance function, wich accepts two functional
        data objects and returns the pairwise distance matrix.
    """
    def pairwise(fdata1, fdata2):

        # Checks
        if not numpy.array_equal(fdata1.domain_range, fdata2.domain_range):
            raise ValueError("Domain ranges for both objects must be equal")

        # Creates an empty matrix with the desired size to store the results.
        matrix = numpy.empty((fdata1.nsamples, fdata2.nsamples))

        # Iterates over the different samples of both objects.
        for i in range(fdata1.nsamples):
            for j in range(fdata2.nsamples):
                matrix[i, j] = distance(fdata1[i], fdata2[j], **kwargs)
        # Computes the metric between all piars of x and y.
        return matrix

    pairwise.__name__ = f"pairwise_{distance.__name__}"

    return pairwise


def norm_lp(fdatagrid, p=2, p2=2):
    r"""Calculate the norm of all the samples in a FDataGrid object.

    For each sample sample f the Lp norm is defined as:

    .. math::
        \lVert f \rVert = \left( \int_D \lvert f \rvert^p dx \right)^{
        \frac{1}{p}}

    Where D is the domain over which the functions are defined.

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
        \lVert f \rVert = \left( \int \int_D \left ( \sqrt{ \lvert f_1(x,y)
        \rvert^2 + \lvert f_2(x,y) \rvert^2 } \right )^p dxdy \right)^{
        \frac{1}{p}}


    Args:
        fdatagrid (FDataGrid): FDataGrid object.
        p (int, optional): p of the lp norm. Must be greater or equal
            than 1. Defaults to 2.
        p2 (int, optional): p index of the vectorial norm applied in case of
            multivariate objects. Defaults to 2.

    Returns:
        numpy.darray: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the inner product of the ith sample
        of the first object and the jth sample of the second one.

    Examples:
        Calculates the norm of a FDataGrid containing the functions y = 1
        and y = x defined in the interval [0,1].


        >>> x = numpy.linspace(0,1,1001)
        >>> fd = FDataGrid([numpy.ones(len(x)), x] ,x)
        >>> norm_lp(fd).round(2)
        array([ 1.  ,  0.58])

        The lp norm is only defined if p >= 1.

        >>> norm_lp(fd, p = 0.5)
        Traceback (most recent call last):
            ....
        ValueError: p must be equal or greater than 1.

    """
    # Checks that the lp normed is well defined
    if p < 1:
        raise ValueError(f"p must be equal or greater than 1.")

    if fdatagrid.ndim_image > 1:
        data_matrix = numpy.linalg.norm(fdatagrid.data_matrix, ord=p2, axis=-1,
                                        keepdims=True)
    else:
        data_matrix = numpy.abs(fdatagrid.data_matrix)

    if fdatagrid.ndim_domain == 1:

        # Computes the norm, approximating the integral with Simpson's rule.
        res = scipy.integrate.simps(data_matrix[..., 0] ** p,
                                    x=fdatagrid.sample_points) ** (1 / p)

    else:
        # Needed to perform surface integration
        return NotImplemented

    if len(res) == 1:
        return res[0]

    return res


def lp_distance(fdata1, fdata2, p=2, *, eval_points=None):
    r"""Lp distance for FDataGrid objects.

    Calculates the distance between all possible pairs of one sample of
    the first FDataGrid object and one of the second one.

    For each pair of samples f and g the distance between them is defined as:

    .. math::
        d(f, g) = d(f, g) = \lVert f - g \rVert

    The norm is specified as a parameter but defaults to the l2 norm.

    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array 2x2 with the computed
        l2 distance between every pair of functions.

        >>> x = numpy.linspace(0, 1, 1001)
        >>> fd = FDataGrid([numpy.ones(len(x))], x)
        >>> fd2 =  FDataGrid([numpy.zeros(len(x))], x)
        >>> lp_distance(fd, fd2).round(2)
        1.0


        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = numpy.linspace(0, 2, 1001)
        >>> fd2 =  FDataGrid([numpy.zeros(len(x)), x/2 + 0.5], x)
        >>> lp_distance(fd, fd2)
        Traceback (most recent call last):
            ....
        ValueError: Domain ranges for both objects must be equal

    """
    # Checks

    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points)

    return norm_lp(fdata1 - fdata2, p=p)




def fisher_rao_distance(fdata1, fdata2, *, eval_points=None):
    """Compute the Fisher-Rao distance btween two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`q_i` and :math:`q_j` be the corresponding SRSF (see :func:`to_srsf`),
    the fisher rao distance is defined as

    .. math::
        d_{FR}(f_i, f_j) = \\| q_i - q_j \\|_2 =
        \\left ( \\int_0^1 sgn(\\dot{f_i}(t))\\sqrt{|\\dot{f_i}(t)|} -
        sgn(\\dot{f_j}(t))\\sqrt{|\\dot{f_j}(t)|} dt \\right )^{\\frac{1}{2}}

    If the observations are distributions of random variables the distance will
    match with the usual fisher-rao distance in non-parametric form for
    probability distributions [S11-2]_.

    If the samples are defined in a domain different than (0,1) their domains
    are normalized to this interval with an affine transformation.

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

    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points)

    # Both should have the same sample points
    eval_points_normalized = _normalize_scale(fdata1.sample_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(sample_points=eval_points_normalized,
                         domain_range=(0,1))
    fdata2 = fdata2.copy(sample_points=eval_points_normalized,
                         domain_range=(0,1))

    fdata1_srsf = to_srsf(fdata1)
    fdata2_srsf = to_srsf(fdata2)

    # Return the L2 distance of the SRSF
    return lp_distance(fdata1_srsf, fdata2_srsf, p=2)

def amplitude_distance(fdata1, fdata2, *, lam=0., eval_points=None, **kwargs):
    """Compute the amplitude distance between two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`q_i` and :math:`q_j` be the corresponding SRSF (see :func:`to_srsf`),
    the amplitude distance is defined as

    .. math::
        d_{A}(f_i, f_j)=min_{\\gamma \\in \\Gamma}d_{FR}(f_i \\circ \\gamma,f_j)

    A penalty term could be added to restrict the ammount of elasticity in the
    alignment used.

    .. math::
        d_{\\lambda}^2(f_i, f_j) =min_{\\gamma \\in \\Gamma} \\{
        d_{FR}^2(f_i \\circ \\gamma, f_j) + \\lambda \\mathcal{R}(\\gamma) \\}


    Where :math:`d_{FR}` is the Fisher-Rao distance and the penalty term is
    given by

    .. math::
        \\mathcal{R}(\\gamma) = \\|\\sqrt{\\dot{\\gamma}}- 1 \\|_{\\mathbb{L}^2}^2

    See [SK16-4-10-1]_ for a detailed explanation.

    If the samples are defined in a domain different than (0,1) their domains
    are normalized to this interval with an affine transformation.

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
        ..  [SK16-4-10-1] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Amplitude Space and a Metric Structure*
            (pp. 107-109). Springer.
    """

    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points)

    # Both should have the same sample points
    eval_points_normalized = _normalize_scale(fdata1.sample_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(sample_points=eval_points_normalized,
                         domain_range=(0,1))
    fdata2 = fdata2.copy(sample_points=eval_points_normalized,
                         domain_range=(0,1))

    fdata1_srsf = to_srsf(fdata1)
    fdata2_srsf = to_srsf(fdata2)

    warping = elastic_registration_warping(fdata1,
                                             template=fdata2,
                                             lam=lam,
                                             eval_points=eval_points_normalized,
                                             fdatagrid_srsf=fdata1_srsf,
                                             template_srsf=fdata2_srsf,
                                             **kwargs)

    fdata1_reg = fdata1.compose(warping)

    distance = lp_distance(to_srsf(fdata1_reg), fdata2_srsf)

    if lam != 0.0:
        # L2 norm || sqrt(Dh) - 1 ||^2
        penalty = warping(eval_points_normalized, derivative=1,
                          keepdims=False)[0]
        penalty = numpy.sqrt(penalty, out=penalty)
        penalty -= 1
        penalty = numpy.square(penalty, out=penalty)
        penalty = scipy.integrate.simps(penalty, x=eval_points_normalized)

        distance = numpy.sqrt(distance**2 + lam*penalty)

    return distance

def phase_distance(fdata1, fdata2, *, lam=0., eval_points=None, **kwargs):
    """Compute the amplitude distance btween two functional objects.

    Let :math:`f_i` and :math:`f_j` be two functional observations, and let
    :math:`\\gamma_{ij}` the corresponding warping used in the elastic
    registration to align :math:`f_i` to :math:`f_j` (see
    :func:`elastic_registration`). The phase distance between :math:`f_i`
    and :math:`f_j` is defined as

    .. math::
        d_{P}(f_i, f_j) = d_{FR}(\\gamma_{ij}, \\gamma_{id}) =
        arcos \\left ( \\int_0^1 \\sqrt {\\dot \\gamma_{ij}(t)} dt \\right )

    See [SK16-4-10-2]_ for a detailed explanation.

    If the samples are defined in a domain different than (0,1) their domains
    are normalized to this interval with an affine transformation.

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
        ..  [SK16-4-10-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Phase Space and a Metric Structure*
            (pp. 109-111). Springer.

    """

    fdata1, fdata2 = _cast_to_grid(fdata1, fdata2, eval_points=eval_points)

    # Rescale in (0,1)
    eval_points_normalized = _normalize_scale(fdata1.sample_points[0])

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(sample_points=eval_points_normalized,
                         domain_range=(0,1))
    fdata2 = fdata2.copy(sample_points=eval_points_normalized,
                         domain_range=(0,1))

    warping = elastic_registration_warping(fdata1, template=fdata2,
                                             lam=lam,
                                             eval_points=eval_points_normalized,
                                             **kwargs)

    derivative_warping = warping(eval_points_normalized, keepdims=False,
                                 derivative=1)[0]

    derivative_warping = numpy.sqrt(derivative_warping, out=derivative_warping)

    d = scipy.integrate.simps(derivative_warping, x=eval_points_normalized)

    return numpy.arccos(d)


def warping_distance(warping1, warping2, *, eval_points=None):
    """Compute the distance between warpings functions.

    Let :math:`\\gamma_i` and :math:`\\gamma_j` be two warpings, defined in
    :math:`\\gamma_i:[a,b] \\rightarrow [a,b]`. The distance in the
    space of warping functions, :math:`\\Gamma`, with the riemannian metric
    given by the fisher-rao inner product can be computed using the structure
    of hilbert sphere in their srsf's.

    .. math::
        d_{\\Gamma}(\\gamma_i, \\gamma_j) = cos^{-1} \\left ( \\int_0^1
        \\sqrt{\\dot \\gamma_i(t)\\dot \\gamma_j(t)}dt \\right )

    See [SK16-4-11-2]_ for a detailed explanation.

    If the warpings are not defined in [0,1], an affine transformation is maked
    to change the domain.

    Args:
        fdata1 (:obj:`FData`): First warping.
        fdata2 (:obj:`FData`): Second warping.
        eval_points (array_like, optional): Array with points of evaluation.

    Returns:
        float: Distance between warpings:

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        ..  [SK16-4-11-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Probability Density Functions*
            (pp. 113-117). Springer.

    """

    warping1, warping2 = _cast_to_grid(warping1, warping2,
                                       eval_points=eval_points)

    # Normalization of warping to (0,1)x(0,1)
    warping1 = normalize_warping(warping1, (0,1))
    warping2 = normalize_warping(warping2, (0,1))

    warping1_data = warping1.derivative().data_matrix[0, ..., 0]
    warping2_data = warping2.derivative().data_matrix[0, ..., 0]

    # In this case the srsf is the sqrt(gamma')
    srsf_warping1 = numpy.sqrt(warping1_data, out=warping1_data)
    srsf_warping2 = numpy.sqrt(warping2_data, out=warping2_data)

    product = numpy.multiply(srsf_warping1, srsf_warping2, out=srsf_warping1)
    d = scipy.integrate.simps(product, x=warping1.sample_points[0])

    return numpy.arccos(d)

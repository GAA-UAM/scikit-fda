
import scipy.integrate
import numpy

from . import FDataGrid
from .registration import (normalize_warping, _normalize_scale,
                           to_srsf, elastic_registration_warping)



def norm_lp(fdatagrid, p=2):
    r"""Calculate the norm of all the samples in a FDataGrid object.

    For each sample sample f the lp norm is defined as:

    .. math::
        \lVert f \rVert = \left( \int_D \lvert f \rvert^p dx \right)^{
        \frac{1}{p}}

    Where D is the domain over which the functions are defined.

    The integral is approximated using Simpson's rule.

    Args:
        fdatagrid (FDataGrid): FDataGrid object.
        p (int, optional): p of the lp norm. Must be greater or equal
            than 1. Defaults to 2.

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
        array([ 1.  , 0.58])

        The lp norm is only defined if p >= 1.

        >>> norm_lp(fd, p = 0.5)
        Traceback (most recent call last):
            ....
        ValueError: p must be equal or greater than 1.

    """
    # Checks that the lp normed is well defined
    if p < 1:
        raise ValueError("p must be equal or greater than 1.")

    if fdatagrid.ndim_image > 1:
        raise ValueError("Not implemented for image with "
                         "dimension greater than 1")

    # Computes the norm, approximating the integral with Simpson's rule.
    return scipy.integrate.simps(numpy.abs(fdatagrid.data_matrix[..., 0]) ** p,
                                 x=fdatagrid.sample_points
                                 ) ** (1 / p)


def metric(fdatagrid, fdatagrid2, norm=norm_lp, **kwargs):
    r"""Return distance for FDataGrid obejcts.

    Calculates the distance between all possible pairs of one sample of
    the first FDataGrid object and one of the second one.

    For each pair of samples f and g the distance between them is defined as:

    .. math::
        d(f, g) = d(f, g) = \lVert f - g \rVert

    The norm is specified as a parameter but defaults to the l2 norm.

    Args:
        fdatagrid (FDataGrid): First FDataGrid object.
        fdatagrid2 (FDataGrid): Second FDataGrid object.
        norm (:obj:`Function`, optional): Norm function used in the definition
            of the distance.
        **kwargs (:obj:`dict`, optional): parameters dictionary to be passed
            to the norm function.

    Returns:
        :obj:`numpy.darray`: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the distance between the ith sample
        of the first object and the jth sample of the second one.


    Examples:
        Computes the distances between an object containing functional data
        corresponding to the functions y = 1 and y = x defined over the
        interval [0, 1] and another ones containing data of the functions y
        = 0 and y = x/2. The result then is an array 2x2 with the computed
        l2 distance between every pair of functions.

        >>> x = numpy.linspace(0, 1, 1001)
        >>> fd = FDataGrid([numpy.ones(len(x)), x], x)
        >>> fd2 =  FDataGrid([numpy.zeros(len(x)), x/2 + 0.5], x)
        >>> metric(fd, fd2).round(2)
        array([[ 1.  , 0.29],
               [ 0.58, 0.29]])


        If the functional data are defined over a different set of points of
        discretisation the functions returns an exception.

        >>> x = numpy.linspace(0, 2, 1001)
        >>> fd2 =  FDataGrid([numpy.zeros(len(x)), x/2 + 0.5], x)
        >>> metric(fd, fd2)
        Traceback (most recent call last):
            ....
        ValueError: Sample points for both objects must be equal

    """
    # Checks
    if not numpy.array_equal(fdatagrid.sample_points,
                             fdatagrid2.sample_points):
        raise ValueError("Sample points for both objects must be equal")
        # Creates an empty matrix with the desired size to store the results.
    matrix = numpy.empty([fdatagrid.nsamples, fdatagrid2.nsamples])
    # Iterates over the different samples of both objects.
    for i in range(fdatagrid.nsamples):
        for j in range(fdatagrid2.nsamples):
            matrix[i, j] = norm(fdatagrid[i] - fdatagrid2[j], **kwargs)
    # Computes the metric between x and y as norm(x -y).
    return matrix



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
        :obj:`numpy.darray`: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the distance between the ith sample
        of the first object and the jth sample of the second one.

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        .. [S11-2] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Function Representation and
            Metric* (pp. 5-7). arXiv:1103.3817v2.

    """
    if (fdata1.ndim_image != 1 or fdata1.ndim_domain != 1 or
        fdata2.ndim_image != 1 or fdata2.ndim_domain != 1):
        raise ValueError("Objects should be unidimensional")


    if not isinstance(fdata1, FDataGrid):
        fdata1 = fdata1.to_grid(eval_points=eval_points)

    if not isinstance(fdata2, FDataGrid):
        fdata2 = fdata2.to_grid(eval_points=eval_points)

    if eval_points is None:
        eval_points = fdata1.sample_points[0]

    eval_points_normalized = _normalize_scale(eval_points)

    # Calculate the corresponding srsf and normalize to (0,1)
    fdata1 = fdata1.copy(sample_points=_normalize_scale(fdata1.sample_points[0]))
    fdata1_srsf = to_srsf(fdata1, eval_points=eval_points_normalized)

    fdata2 = fdata2.copy(sample_points=_normalize_scale(fdata2.sample_points[0]))
    fdata2_srsf = to_srsf(fdata2, eval_points=eval_points_normalized)

    # Return the L2 distance of the SRSF
    return metric(fdata1_srsf, fdata2_srsf, norm=norm_lp)

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
        \\mathcal{R}(\\gamma) = \|\\sqrt{\\dot{\\gamma}}- 1 \|_{\\mathbb{L}^2}^2

    See [SK16-4-10-1]_ for a detailed explanation.

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
        :obj:`numpy.darray`: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the distance between the ith sample
        of the first object and the jth sample of the second one.

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        ..  [SK16-4-10-1] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Amplitude Space and a Metric Structure*
            (pp. 107-109). Springer.
    """

    if (fdata1.ndim_image != 1 or fdata1.ndim_domain != 1 or
        fdata2.ndim_image != 1 or fdata2.ndim_domain != 1):
        raise ValueError("Objects should be unidimensional")

    if not isinstance(fdata1, FDataGrid):
        fdata1 = fdata1.to_grid(eval_points=eval_points)

    if not isinstance(fdata2, FDataGrid):
        fdata2 = fdata2.to_grid(eval_points=eval_points)

    if eval_points is None:
        eval_points = fdata1.sample_points[0]

    # For optimization, fdata1 will be the object with more samples
    if fdata1.nsamples < fdata2.nsamples:
        fdata_aux = fdata1
        fdata1 = fdata2
        fdata2 = fdata_aux
        transpose = True
    else:
        transpose = False

    matrix = numpy.empty((fdata1.nsamples, fdata2.nsamples))

    eval_points_normalized = _normalize_scale(eval_points)

    # Normalization of scale to (0,1)
    fdata1 = fdata1.copy(sample_points=_normalize_scale(fdata1.sample_points[0]))
    fdata1_srsf = to_srsf(fdata1, eval_points=eval_points_normalized)
    fdata2 = fdata2.copy(sample_points=_normalize_scale(fdata2.sample_points[0]))
    fdata2_srsf = to_srsf(fdata2, eval_points=eval_points_normalized)

    # Iterate over the smallest FData
    for j in range(fdata2.nsamples):

        fdataj = fdata2[j]

        warping_j = elastic_registration_warping(fdata1,
                                                 template=fdataj,
                                                 lam=lam,
                                                 eval_points=eval_points_normalized,
                                                 fdatagrid_srsf=fdata1_srsf,
                                                 template_srsf=fdata2_srsf[j],
                                                 **kwargs)
        f_register_j  = fdata1.compose(warping_j)

        # Distance without penalty term
        matrix[:, j] = norm_lp(f_register_j - fdataj, p=2)


        if lam != 0.0:
            numpy.square(matrix[:, j], out=matrix[:, j])

            # L2 norm || sqrt(Dh) - 1 ||^2
            warps_values = warping_j(eval_points_normalized, derivative=1)
            warps_values = numpy.sqrt(warps_values)
            warps_values -= 1
            warps_values = numpy.square(warps_values)


            matrix[:, j] += lam*scipy.integrate.simps(warps_values,
                                                  x=eval_points_normalized, axis=1)

            numpy.sqrt(matrix[:, j], out=matrix[:, j])


    return matrix

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
        :obj:`numpy.darray`: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the distance between the ith sample
        of the first object and the jth sample of the second one.

    Raises:
        ValueError: If the objects are not unidimensional.


    Refereces:
        ..  [SK16-4-10-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Phase Space and a Metric Structure*
            (pp. 109-111). Springer.

    """

    if (fdata1.ndim_image != 1 or fdata1.ndim_domain != 1 or
        fdata2.ndim_image != 1 or fdata2.ndim_domain != 1):
        raise ValueError("Objects should be unidimensional")

    if not isinstance(fdata1, FDataGrid):
        fdata1 = fdata1.to_grid(eval_points=eval_points)

    if not isinstance(fdata2, FDataGrid):
        fdata2 = fdata2.to_grid(eval_points=eval_points)

    if eval_points is None:
        eval_points = fdata1.sample_points[0]

    # For optimization, fdata1 will be the object with more samples
    if fdata1.nsamples < fdata2.nsamples:
        fdata_aux = fdata1
        fdata1 = fdata2
        fdata2 = fdata_aux
        transpose = True
    else:
        transpose = False

    matrix = numpy.empty((fdata1.nsamples, fdata2.nsamples))

    eval_points_normalized = _normalize_scale(eval_points)

    fdata1 = fdata1.copy(sample_points=_normalize_scale(fdata1.sample_points[0]))
    fdata1_srsf = to_srsf(fdata1, eval_points=eval_points_normalized)

    fdata2 = fdata2.copy(sample_points=_normalize_scale(fdata2.sample_points[0]))
    fdata2_srsf = to_srsf(fdata2, eval_points=eval_points_normalized)

    # Iterate over the smallest FData
    for j in range(fdata2.nsamples):

        fdataj = fdata2[j]
        warping_j = elastic_registration_warping(fdata1,
                                                 template=fdataj,
                                                 lam=lam,
                                                 eval_points=eval_points_normalized,
                                                 fdatagrid_srsf=fdata1_srsf,
                                                 template_srsf=fdata2_srsf[j])

        derivative_warping_j = warping_j(eval_points_normalized, keepdims=False,
                                         derivative=1)

        d = scipy.integrate.simps(numpy.sqrt(derivative_warping_j),
                                  x=eval_points_normalized, axis=1)

        d[d > 1] = 1

        matrix[:,j] = numpy.arccos(d)

    # Undo the tranposition due to the swap of fdatas
    if transpose:
        matrix = matrix.T

    return matrix


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
        fdata1 (FData): First FData object.
        fdata2 (FDataGrid): Second FData object.
        eval_points (array_like, optional): Array with points of evaluation.

    Returns:
        :obj:`numpy.darray`: Matrix with as many rows as samples in the first
        object and as many columns as samples in the second one. Each
        element (i, j) of the matrix is the distance between the ith sample
        of the first object and the jth sample of the second one.

    Raises:
        ValueError: If the objects are not unidimensional.

    Refereces:
        ..  [SK16-4-11-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Probability Density Functions*
            (pp. 113-117). Springer.

    """
    if (warping1.ndim_image != 1 or warping1.ndim_domain != 1 or
        warping2.ndim_image != 1 or warping2.ndim_domain != 1):
        raise ValueError("Objects should be unidimensional")


    if not isinstance(warping1, FDataGrid):
        warping1 = warping1.to_grid(eval_points=eval_points)

    if not isinstance(warping2, FDataGrid):
        warping2 = warping2.to_grid(eval_points=eval_points)

    if eval_points is None:
        eval_points = warping1.sample_points[0]

    eval_points_normalized = _normalize_scale(eval_points)

    n = warping1.nsamples
    m = warping2.nsamples
    matrix = numpy.empty((n, m))

    # Normalization of warping to (0,1)x(0,1)
    warping1 = normalize_warping(warping1)
    warping2 = normalize_warping(warping2)

    warping1 = warping1.derivative()(eval_points_normalized, keepdims=False)
    warping2 = warping2.derivative()(eval_points_normalized, keepdims=False)

    # In this case the srsf is the sqrt(gamma')
    srsf_warping1 = numpy.sqrt(warping1)
    srsf_warping2 = numpy.sqrt(warping2).T


    for i in range(n):

        product_i = numpy.multiply(srsf_warping1[i][:,numpy.newaxis], srsf_warping2)
        d = scipy.integrate.simps(product_i, x=eval_points_normalized, axis=0)

        d[d>1]=1
        matrix[i] = numpy.arccos(d)

    return matrix

"""Defines methods for the validation of the smoothing."""
import numpy

from . import kernel_smoothers


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


def cv(fdatagrid, s_matrix):
    r"""Cross validation scoring method.

    It calculates the cross validation score for every sample in a FDataGrid
    object given a smoothing matrix :math:`\hat{H}^\nu` calculated with a
    parameter :math:`\nu`:

    .. math::
        CV(\nu)=\frac{1}{n} \sum_i \left(y_i - \hat{y}_i^{\nu(
        -i)}\right)^2

    Where :math:`\hat{y}_i^{\nu(-i)}` is the adjusted :math:`y_i` when the
    the pair of values :math:`(x_i,y_i)` are excluded in the smoothing. This
    would require to recalculate the smoothing matrix n times. Fortunately
    the above formula can be expressed in a way where the smoothing matrix
    does not need to be calculated again.

    .. math::
        CV(\nu)=\frac{1}{n} \sum_i \left(\frac{y_i - \hat{y}_i^\nu}{1 -
        \hat{H}_{ii}^\nu}\right)^2

    Args:
        fdatagrid (FDataGrid): Object over which the CV score is calculated.
        s_matrix (numpy.darray): Smoothig matrix.

    Returns:
        float: Cross validation score.

    """
    y = fdatagrid.data_matrix[..., 0]
    y_est = numpy.dot(s_matrix, y.T).T
    return numpy.mean(((y - y_est) / (1 - s_matrix.diagonal())) ** 2)


def gcv(fdatagrid, s_matrix, penalisation_function=None):
    r"""General cross validation scoring method.

    It calculates the general cross validation score for every sample in a
    FDataGrid object given a smoothing matrix :math:`\hat{H}^\nu`
    calculated with a parameter :math:`\nu`:

    .. math::
        GCV(\nu)=\Xi(\nu,n)\frac{1}{n} \sum_i \left(y_i - \hat{
        y}_i^\nu\right)^2

    Where :math:`\hat{y}_i^{\nu}` is the adjusted :math:`y_i` and
    :math:`\Xi` is a penalisation function. By default the penalisation
    function is:

    .. math::
        \Xi(\nu,n) = \left( 1 - \frac{tr(\hat{H}^\nu)}{n} \right)^{-2}

    But others such as the Akaike's information criterion can be considered.

    Args:
        fdatagrid (FDataGrid): Object over which the CV score is calculated.
        s_matrix (numpy.darray): Smoothig matrix.
        penalisation_function (Function): Function taking a smoothing matrix
            and returing a penalisation score. If None the general cross
            validation penalisation is applied. Defaults to None.

    Returns:
        float: Cross validation score.

    """
    y = fdatagrid.data_matrix[..., 0]
    y_est = numpy.dot(s_matrix, y.T).T
    if penalisation_function is not None:
        return (numpy.mean(((y - y_est) / (1 - s_matrix.diagonal())) ** 2)
                * penalisation_function(s_matrix))
    return (numpy.mean(((y - y_est) / (1 - s_matrix.diagonal())) ** 2)
            * (1 - s_matrix.diagonal().mean()) ** -2)


def minimise(fdatagrid, parameters,
             smoothing_method=kernel_smoothers.nw, cv_method=gcv,
             penalisation_function=None, **kwargs):
    """Chooses the best smoothness parameter and performs smoothing.

    Performs the smoothing of a FDataGrid object choosing the best
    parameter of a given list using a cross validation scoring method.

    Args:
        fdatagrid (FDataGrid): FDataGrid object.
        parameters (list of double): List of parameters to be tested.
        smoothing_method (Function): Function that takes a list of
            discretised points, a parameter, an optionally a weights matrix
            and returns a hat matrix or smoothing matrix.
        cv_method (Function): Function that takes a matrix,
            a smoothing matrix, and optionally a weights matrix and
            calculates a cross validation score.
        penalisation_function(Fuction): if gcv is selected as cv_method a
            penalisation function can be specified through this parameter.

    Returns:
        dict: A dictionary containing the following:

            {
                'scores': (list of double) List of the scores for each
                    parameter.

                'best_score': (double) Minimum score.

                'best_parameter': (double) Parameter that produces the
                    lesser score.

                'hat_matrix': (numpy.darray) Hat matrix built with the best
                    parameter.

                'fdatagrid': (FDataGrid) Smoothed FDataGrid object.

            }

    Examples:
        Creates a FDataGrid object of the function :math:`y=x^2` and peforms
        smoothing by means of the k-nearest neighbours method.

        >>> import skfda
        >>> x = numpy.linspace(-2, 2, 5)
        >>> fd = skfda.FDataGrid(x ** 2, x)
        >>> res = minimise(fd, [2,3], smoothing_method=kernel_smoothers.knn)
        >>> numpy.array(res['scores']).round(2)
        array([ 11.67, 12.37])
        >>> round(res['best_score'], 2)
        11.67
        >>> res['best_parameter']
        2
        >>> res['hat_matrix'].round(2)
        array([[ 0.5 , 0.5 , 0.  , 0.  , 0.  ],
               [ 0.33, 0.33, 0.33, 0.  , 0.  ],
               [ 0.  , 0.33, 0.33, 0.33, 0.  ],
               [ 0.  , 0.  , 0.33, 0.33, 0.33],
               [ 0.  , 0.  , 0.  , 0.5 , 0.5 ]])
        >>> res['fdatagrid'].round(2)
        FDataGrid(
            array([[[ 2.5 ],
                    [ 1.67],
                    [ 0.67],
                    [ 1.67],
                    [ 2.5 ]]]),
            sample_points=[array([-2., -1.,  0.,  1.,  2.])],
            domain_range=array([[-2.,  2.]]),
            ...)

        Other validation methods can be used such as cross-validation or
        general cross validation using other penalisation functions.

        >>> res = minimise(fd, [2,3], smoothing_method=kernel_smoothers.knn,
        ...                cv_method=cv)
        >>> numpy.array(res['scores']).round(2)
        array([ 4.2, 5.5])
        >>> res = minimise(fd, [2,3], smoothing_method=kernel_smoothers.knn,
        ...                penalisation_function=aic)
        >>> numpy.array(res['scores']).round(2)
        array([ 9.35, 10.71])
        >>> res = minimise(fd, [2,3], smoothing_method=kernel_smoothers.knn,
        ...                penalisation_function=fpe)
        >>> numpy.array(res['scores']).round(2)
        array([ 9.8, 11. ])
        >>> res = minimise(fd, [2,3], smoothing_method=kernel_smoothers.knn,
        ...                penalisation_function=shibata)
        >>> numpy.array(res['scores']).round(2)
        array([ 7.56, 9.17])
        >>> res = minimise(fd, [2,3], smoothing_method=kernel_smoothers.knn,
        ...                penalisation_function=rice)
        >>> numpy.array(res['scores']).round(2)
        array([ 21. , 16.5])

    """
    if fdatagrid.ndim_domain != 1:
        raise NotImplementedError("This method only works when the dimension "
                                  "of the domain of the FDatagrid object is "
                                  "one.")
    if fdatagrid.ndim_image != 1:
        raise NotImplementedError("This method only works when the dimension "
                                  "of the image of the FDatagrid object is "
                                  "one.")
    # Reduce one dimension the sample points.
    sample_points = fdatagrid.sample_points[0]
    scores = []
    # Calculates the scores for each parameter.
    if penalisation_function is not None:
        for h in parameters:
            s = smoothing_method(sample_points, h, **kwargs)
            scores.append(
                cv_method(fdatagrid, s,
                          penalisation_function=penalisation_function))
    else:
        for h in parameters:
            s = smoothing_method(sample_points, h, **kwargs)
            scores.append(
                cv_method(fdatagrid, s))
    # gets the best parameter.
    h = parameters[int(numpy.argmin(scores))]
    s = smoothing_method(sample_points, h, **kwargs)
    fdatagrid_adjusted = fdatagrid.copy(
        data_matrix=numpy.dot(fdatagrid.data_matrix[..., 0], s.T))

    return {'scores': scores,
            'best_score': numpy.min(scores),
            'best_parameter': h,
            'hat_matrix': s,
            'fdatagrid': fdatagrid_adjusted,
            }


def aic(s_matrix):
    r"""Akaike's information criterion for cross validation.

    .. math::
        \Xi(\nu,n) = \exp\left(2 * \frac{tr(\hat{H}^\nu)}{n}\right)

    Args:
        s_matrix (numpy.darray): Smoothing matrix whose penalisation
            score is desired.

    Returns:
         float: Penalisation given by the Akaike's information criterion.

    """
    return numpy.exp(2 * s_matrix.diagonal().mean())


def fpe(s_matrix):
    r"""Finite prediction error for cross validation.

    .. math::
        \Xi(\nu,n) = \frac{1 + \frac{tr(\hat{H}^\nu)}{n}}{1 -
        \frac{tr(\hat{H}^\nu)}{n}}

    Args:
        s_matrix (numpy.darray): Smoothing matrix whose penalisation
            score is desired.

    Returns:
         float: Penalisation given by the finite prediction error.

    """
    return (1 + s_matrix.diagonal().mean()) / (1 - s_matrix.diagonal().mean())


def shibata(s_matrix):
    r"""Shibata's model selector for cross validation.

    .. math::
        \Xi(\nu,n) = 1 + 2 * \frac{tr(\hat{H}^\nu)}{n}

    Args:
        s_matrix (numpy.darray): Smoothing matrix whose penalisation
            score is desired.

    Returns:
         float: Penalisation given by the Shibata's model selector.

    """
    return 1 + 2 * s_matrix.diagonal().mean()


def rice(s_matrix):
    r"""Rice's bandwidth selector for cross validation.

    .. math::
        \Xi(\nu,n) = \left(1 - 2 * \frac{tr(\hat{H}^\nu)}{n}\right)^{-1}

    Args:
        s_matrix (numpy.darray): Smoothing matrix whose penalisation
            score is desired.

    Returns:
         float: Penalisation given by the Rice's bandwidth selector.

    """
    return (1 - 2 * s_matrix.diagonal().mean()) ** -1

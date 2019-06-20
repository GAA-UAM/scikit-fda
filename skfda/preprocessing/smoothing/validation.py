"""Defines methods for the validation of the smoothing."""
import numpy as np

from . import kernel_smoothers
from sklearn.model_selection import GridSearchCV


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


class LinearSmootherLeaveOneOutScorer():
    r"""Leave-one-out cross validation scoring method for linear smoothers.

    It calculates the cross validation score for every sample in a FDataGrid
    object given a linear smoother with a smoothing matrix :math:`\hat{H}^\nu`
    calculated with a parameter :math:`\nu`:

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
        estimator (Estimator): Linear smoothing estimator.
        X (FDataGrid): Functional data to smooth.
        y (FDataGrid): Functional data target. Should be the same as X.

    Returns:
        float: Cross validation score, with negative sign, as it is a
        penalization.

    """

    def __call__(self, estimator, X, y):
        y_est = estimator.transform(X)

        hat_matrix = estimator.hat_matrix_

        return -np.mean(((y.data_matrix[..., 0] - y_est.data_matrix[..., 0])
                         / (1 - hat_matrix.diagonal())) ** 2)


class LinearSmootherGeneralizedCVScorer():
    r"""Generalized cross validation scoring method for linear smoothers.

    It calculates the general cross validation score for every sample in a
    FDataGrid object given a smoothing matrix :math:`\hat{H}^\nu`
    calculated with a parameter :math:`\nu`:

    .. math::
        GCV(\nu)=\Xi(\nu,n)\frac{1}{n} \sum_i \left(y_i - \hat{
        y}_i^\nu\right)^2

    Where :math:`\hat{y}_i^{\nu}` is the adjusted :math:`y_i` and
    :math:`\Xi` is a penalization function. By default the penalization
    function is:

    .. math::
        \Xi(\nu,n) = \left( 1 - \frac{tr(\hat{H}^\nu)}{n} \right)^{-2}

    but others such as the Akaike's information criterion can be considered.

    Args:
        estimator (Estimator): Linear smoothing estimator.
        X (FDataGrid): Functional data to smooth.
        y (FDataGrid): Functional data target. Should be the same as X.

    Returns:
        float: Cross validation score, with negative sign, as it is a
        penalization.

    """
    def __init__(self, penalization_function=None):
        self.penalization_function = penalization_function

    def __call__(self, estimator, X, y):
        y_est = estimator.transform(X)

        hat_matrix = estimator.hat_matrix_

        if self.penalization_function is None:
            def penalization_function(hat_matrix):
                return (1 - hat_matrix.diagonal().mean()) ** -2
        else:
            penalization_function = self.penalization_function

        return -(np.mean(((y.data_matrix[..., 0] - y_est.data_matrix[..., 0])
                          / (1 - hat_matrix.diagonal())) ** 2)
                 * penalization_function(hat_matrix))


def optimize_smoothing_parameter(fdatagrid, parameter_values,
                                 smoothing_method=None,
                                 cv_method=None):
    """Chooses the best smoothing parameter and performs smoothing.

    Performs the smoothing of a FDataGrid object choosing the best
    parameter of a given list using a cross validation scoring method.

    Note:
        This is similar to fitting a scikit-learn GridSearchCV over the
        data, using the cv_method as a scorer.

    Args:
        fdatagrid (FDataGrid): FDataGrid object.
        parameters (list of double): List of parameters to be tested.
        smoothing_method (Function): Function that takes a list of
            discretised points, a parameter, an optionally a weights matrix
            and returns a hat matrix or smoothing matrix.
        cv_method (Function): Function that takes a matrix,
            a smoothing matrix, and optionally a weights matrix and
            calculates a cross validation score.
        penalization_function(Fuction): if gcv is selected as cv_method a
            penalization function can be specified through this parameter.

    Returns:
        grid: A scikit-learn GridSearchCV estimator, properly fitted.

    Examples:
        Creates a FDataGrid object of the function :math:`y=x^2` and peforms
        smoothing by means of the k-nearest neighbours method.

        >>> import skfda
        >>> x = np.linspace(-2, 2, 5)
        >>> fd = skfda.FDataGrid(x ** 2, x)
        >>> grid = optimize_smoothing_parameter(fd, [2,3],
        ...            smoothing_method=kernel_smoothers.KNeighborsSmoother())
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-11.67, -12.37])
        >>> round(grid.best_score_, 2)
        -11.67
        >>> grid.best_params_['smoothing_parameter']
        2
        >>> grid.best_estimator_.hat_matrix_.round(2)
        array([[ 0.5 , 0.5 , 0.  , 0.  , 0.  ],
               [ 0.33, 0.33, 0.33, 0.  , 0.  ],
               [ 0.  , 0.33, 0.33, 0.33, 0.  ],
               [ 0.  , 0.  , 0.33, 0.33, 0.33],
               [ 0.  , 0.  , 0.  , 0.5 , 0.5 ]])
        >>> grid.best_estimator_.transform(fd).round(2)
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
        general cross validation using other penalization functions.

        >>> grid = optimize_smoothing_parameter(fd, [2,3],
        ...         smoothing_method=kernel_smoothers.KNeighborsSmoother(),
        ...         cv_method=LinearSmootherLeaveOneOutScorer())
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-4.2, -5.5])
        >>> grid = optimize_smoothing_parameter(fd, [2,3],
        ...         smoothing_method=kernel_smoothers.KNeighborsSmoother(),
        ...         cv_method=LinearSmootherGeneralizedCVScorer(
        ...                         akaike_information_criterion))
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([ -9.35, -10.71])
        >>> grid = optimize_smoothing_parameter(fd, [2,3],
        ...         smoothing_method=kernel_smoothers.KNeighborsSmoother(),
        ...         cv_method=LinearSmootherGeneralizedCVScorer(
        ...                         finite_prediction_error))
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([ -9.8, -11. ])
        >>> grid = optimize_smoothing_parameter(fd, [2,3],
        ...         smoothing_method=kernel_smoothers.KNeighborsSmoother(),
        ...         cv_method=LinearSmootherGeneralizedCVScorer(shibata))
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-7.56, -9.17])
        >>> grid = optimize_smoothing_parameter(fd, [2,3],
        ...         smoothing_method=kernel_smoothers.KNeighborsSmoother(),
        ...         cv_method=LinearSmootherGeneralizedCVScorer(rice))
        >>> np.array(grid.cv_results_['mean_test_score']).round(2)
        array([-21. , -16.5])

    """
    if fdatagrid.ndim_domain != 1:
        raise NotImplementedError("This method only works when the dimension "
                                  "of the domain of the FDatagrid object is "
                                  "one.")
    if fdatagrid.ndim_image != 1:
        raise NotImplementedError("This method only works when the dimension "
                                  "of the image of the FDatagrid object is "
                                  "one.")

    if smoothing_method is None:
        smoothing_method = kernel_smoothers.NadarayaWatsonSmoother()

    if cv_method is None:
        cv_method = LinearSmootherGeneralizedCVScorer()

    grid = GridSearchCV(estimator=smoothing_method,
                        param_grid={'smoothing_parameter': parameter_values},
                        scoring=cv_method, cv=[(slice(None), slice(None))])
    grid.fit(fdatagrid, fdatagrid)

    return grid


def akaike_information_criterion(hat_matrix):
    r"""Akaike's information criterion for cross validation.

    .. math::
        \Xi(\nu,n) = \exp\left(2 * \frac{tr(\hat{H}^\nu)}{n}\right)

    Args:
        hat_matrix (numpy.darray): Smoothing matrix whose penalization
            score is desired.

    Returns:
         float: penalization given by the Akaike's information criterion.

    """
    return np.exp(2 * hat_matrix.diagonal().mean())


def finite_prediction_error(hat_matrix):
    r"""Finite prediction error for cross validation.

    .. math::
        \Xi(\nu,n) = \frac{1 + \frac{tr(\hat{H}^\nu)}{n}}{1 -
        \frac{tr(\hat{H}^\nu)}{n}}

    Args:
        hat_matrix (numpy.darray): Smoothing matrix whose penalization
            score is desired.

    Returns:
         float: penalization given by the finite prediction error.

    """
    return ((1 + hat_matrix.diagonal().mean())
            / (1 - hat_matrix.diagonal().mean()))


def shibata(hat_matrix):
    r"""Shibata's model selector for cross validation.

    .. math::
        \Xi(\nu,n) = 1 + 2 * \frac{tr(\hat{H}^\nu)}{n}

    Args:
        hat_matrix (numpy.darray): Smoothing matrix whose penalization
            score is desired.

    Returns:
         float: penalization given by the Shibata's model selector.

    """
    return 1 + 2 * hat_matrix.diagonal().mean()


def rice(hat_matrix):
    r"""Rice's bandwidth selector for cross validation.

    .. math::
        \Xi(\nu,n) = \left(1 - 2 * \frac{tr(\hat{H}^\nu)}{n}\right)^{-1}

    Args:
        hat_matrix (numpy.darray): Smoothing matrix whose penalization
            score is desired.

    Returns:
         float: penalization given by the Rice's bandwidth selector.

    """
    return (1 - 2 * hat_matrix.diagonal().mean()) ** -1

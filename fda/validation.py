"""This module defines methods for the validation of the smoothing.

"""
import fda
from fda import kernel_smoothers
import numpy

__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


def cv(fdatagrid, s_matrix, w_matrix=None):
    """ Cross validation scoring method.

    """
    # TODO check method and compare it to R library.
    y = fdatagrid.data_matrix
    y_est = numpy.dot(s_matrix, y.T).T
    return numpy.mean(((y - y_est)/(1 - s_matrix.diagonal()))**2)


def gcv(data_matrix, s_matrix, w_matrix=None):
    raise NotImplementedError


def minimise(fdatagrid, parameters,
             smoothing_method=kernel_smoothers.nw, cv_method=gcv,
             w=None, **kwargs):
    """ Performs the smoothing of a FDataGrid object choosing the best
    parameter of a given list using a cross validation scoring method.

    Args:
        fdatagrid (FDataGrid): FDataGrid object.
        parameters (list of double): List of parameters to be tested.
        smoothing_method (Function): Function that takes a list of
            discretised points, a parameter, an optionally a weights matrix
            and returns a hat matrix or smoothing matrix.
        cv_method (function): Function that takes a matrix,
            a smoothing matrix, and optionally a weights matrix and
            calculates a cross validation score.
        w (numpy.darray): weights matrix.


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

    """
    scores = []
    # Calculates the scores for each parameter.
    for h in parameters:
        s = smoothing_method(fdatagrid.sample_points, h, **kwargs)
        scores.append(cv_method(fdatagrid, s, w))
    # gets the best parameter.
    h = parameters[numpy.argmin(scores)]
    s = smoothing_method(fdatagrid.sample_points, h, **kwargs)
    fdatagrid_adjusted = fda.FDataGrid(numpy.dot(fdatagrid.data_matrix, s),
                                       fdatagrid.sample_points,
                                       fdatagrid.sample_range,
                                       fdatagrid.names)
    return {'scores': scores,
            'best_score': numpy.min(scores),
            'best_parameter': h,
            'hat_matrix': s,
            'fdatagrid': fdatagrid_adjusted,
            }

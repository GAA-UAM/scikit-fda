"""Functional data descriptive statistics.
"""
from ..depth import modified_band_depth

def mean(fdata, weights=None):
    """Compute the mean of all the samples in a FData object.

    Computes the mean of all the samples in a FDataGrid or FDataBasis object.

    Args:
        fdata (FDataGrid or FDataBasis): Object containing all the samples
            whose mean is wanted.
        weight (array-like, optional): List of weights.


    Returns:
        FDataGrid or FDataBasis: A FDataGrid or FDataBasis object with just
        one sample representing the mean of all the samples in the original
        object.

    """
    return fdata.mean(weights)


def var(fdatagrid):
    """Compute the variance of a set of samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the set of samples
        whose variance is desired.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        variance of all the samples in the original FDataGrid object.

    """
    return fdatagrid.var()


def gmean(fdatagrid):
    """Compute the geometric mean of all the samples in a FDataGrid object.

    Args:
        fdatagrid (FDataGrid): Object containing all the samples whose
            geometric mean is wanted.

    Returns:
        FDataGrid: A FDataGrid object with just one sample representing the
        geometric mean of all the samples in the original FDataGrid object.

    """
    return fdatagrid.gmean()


def cov(fdatagrid):
    """Compute the covariance.

    Calculates the covariance matrix representing the covariance of the
    functional samples at the observation points.

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
            functional variable.

    Returns:
        numpy.darray: Matrix of covariances.

    """
    return fdatagrid.cov()


def depth_based_median(fdatagrid, depth_method=modified_band_depth):
    """Compute the median based on a depth measure.

    The depth based median is the deepest curve given a certain
    depth measure

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
            functional variable.
        depth_method (:ref:`depth measure <depth-measures>`, optional):
                Method used to order the data. Defaults to :func:`modified
                band depth <fda.depth_measures.modified_band_depth>`.

    Returns:
        FDataGrid: object containing the computed depth_based median.

    """
    depth = depth_method(fdatagrid)
    indices_descending_depth = (-depth).argsort(axis=0)

    # The median is the deepest curve
    return fdatagrid[indices_descending_depth[0]]


def trim_mean(fdatagrid,
              proportiontocut,
              depth_method=modified_band_depth):
    """Compute the trimmed means based on a depth measure.

    The trimmed means consists in computing the mean function without a
    percentage of least deep curves. That is, we first remove the least deep
    curves and then we compute the mean as usual.

    Note that in scipy the leftmost and rightmost proportiontocut data are
    removed. In this case, as we order the data by the depth, we only remove
    those that have the least depth values.

    Args:
        fdatagrid (FDataGrid): Object containing different samples of a
            functional variable.
        proportiontocut (float): indicates the percentage of functions to
            remove. It is not easy to determine as it varies from dataset to
            dataset.
        depth_method (:ref:`depth measure <depth-measures>`, optional):
            Method used to order the data. Defaults to :func:`modified
            band depth <fda.depth_measures.modified_band_depth>`.

    Returns:
        FDataGrid: object containing the computed trimmed mean.

    """
    n_samples_to_keep = (fdatagrid.n_samples -
                         int(fdatagrid.n_samples * proportiontocut))

    # compute the depth of each curve and store the indexes in descending order
    depth = depth_method(fdatagrid)
    indices_descending_depth = (-depth).argsort(axis=0)

    trimmed_curves = fdatagrid[indices_descending_depth[:n_samples_to_keep]]

    return trimmed_curves.mean()

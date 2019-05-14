"""Functional principal component analysis.
"""

from ....exploratory.stats import mean
import numpy as np


def fpca(fdatagrid, n=2):
    """Compute Functional Principal Components Analysis.

    Performs Functional Principal Components Analysis to reduce
    dimensionality and obtain the principal modes of variation for a
    functional data object.

    It uses SVD numpy implementation to compute PCA.

    Args:
        fdatagrid (FDataGrid): functional data object.
        n (int, optional): Number of principal components. Defaults to 2.

    Returns:
        tuple: (scores, principal directions, eigenvalues)

    """
    fdatagrid = fdatagrid - mean(fdatagrid)  # centers the data
    # singular value decomposition
    u, s, v = np.linalg.svd(fdatagrid.data_matrix)
    principal_directions = v.T  # obtain the eigenvectors matrix
    eigenvalues = (np.diag(s) ** 2) / (fdatagrid.nsamples - 1)
    scores = u @ s  # functional principal scores

    return scores, principal_directions, eigenvalues

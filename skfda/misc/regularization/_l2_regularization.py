import numpy as np

from ._regularization import Regularization


class L2Regularization(Regularization):
    """
    Regularization using a sum of coefficient squares.

    """

    def penalty_matrix(self, coef_info):
        return np.identity(coef_info.shape[0])

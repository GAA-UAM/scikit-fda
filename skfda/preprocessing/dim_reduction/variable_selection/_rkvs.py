from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.linalg as linalg
import sklearn.utils.validation

from ...._utils import _classifier_get_classes
from ...._utils._sklearn_adapter import (
    BaseEstimator,
    InductiveTransformerMixin,
)
from ....representation import FDataGrid
from ....typing._numpy import NDArrayFloat, NDArrayInt


def _rkhs_vs(
    X: NDArrayFloat,
    Y: NDArrayInt,
    n_features_to_select: int = 1,
) -> Tuple[NDArrayInt, NDArrayFloat]:
    """
    RKHS-VS implementation.

    Parameters:
        X: Matrix of trajectories
        Y: Vector of class labels
        n_features_to_select: Number of selected features

    Returns:
        Selected features and vector of scores.

    """
    X = np.atleast_2d(X)
    assert n_features_to_select >= 1
    assert n_features_to_select <= X.shape[1]

    _, Y = _classifier_get_classes(Y)

    selected_features = np.zeros(n_features_to_select, dtype=int)
    score = np.zeros(n_features_to_select)
    indexes = np.arange(0, X.shape[1])

    # Calculate means and covariance matrix
    class_1_trajectories = X[Y.ravel() == 1]
    class_0_trajectories = X[Y.ravel() == 0]

    means = (
        np.mean(class_1_trajectories, axis=0)
        - np.mean(class_0_trajectories, axis=0)
    )

    class_1_count = sum(Y)
    class_0_count = Y.shape[0] - class_1_count

    class_1_proportion = class_1_count / Y.shape[0]
    class_0_proportion = class_0_count / Y.shape[0]

    # The result should be casted to 2D because of bug #11502 in numpy
    variances = (
        class_1_proportion * np.atleast_2d(
            np.cov(class_1_trajectories, rowvar=False, bias=True),
        )
        + class_0_proportion * np.atleast_2d(
            np.cov(class_0_trajectories, rowvar=False, bias=True),
        )
    )

    # The first variable maximizes |mu(t)|/sigma(t)
    mu_sigma = np.abs(means) / np.sqrt(np.diag(variances))

    selected_features[0] = np.argmax(mu_sigma)
    score[0] = mu_sigma[selected_features[0]]
    indexes = np.delete(indexes, selected_features[0])

    for i in range(1, n_features_to_select):
        aux = np.zeros_like(indexes, dtype=np.float_)

        for j in range(0, indexes.shape[0]):
            new_selection = np.concatenate([
                selected_features[:i],
                [indexes[j]],
            ])

            new_means = np.atleast_2d(means[new_selection])

            lstsq_solution = linalg.lstsq(
                variances[new_selection[:, np.newaxis], new_selection],
                new_means.T,
                rcond=None,
            )[0]

            aux[j] = new_means @ lstsq_solution

        aux2 = np.argmax(aux)
        selected_features[i] = indexes[aux2]
        score[i] = aux[aux2]
        indexes = np.delete(indexes, aux2)

    return selected_features, score


class RKHSVariableSelection(
    BaseEstimator,
    InductiveTransformerMixin[FDataGrid, NDArrayFloat, NDArrayInt],
):
    r"""
    Reproducing kernel variable selection.

    This is a filter variable selection method for binary classification
    problems. With a fixed number :math:`d` of variables to select, it aims to
    find the variables :math:`X(t_1), \ldots, X(t_d)` for the values
    :math:`t_1, \ldots, t_d` that maximize the separation of
    the class means in the reduced space, measured using the Mahalanobis
    distance

    .. math::
        \phi(t_1, \ldots, t_d) = m_{t_1, \ldots, t_d}^T
        K_{t_1, \ldots, t_d}^{-1} m_{t_1, \ldots, t_d}

    where :math:`m_{t_1, \ldots, t_d}` is the difference of the mean
    functions of both classes evaluated at points :math:`t_1, \ldots, t_d`
    and :math:`K_{t_1, \ldots, t_d}` is the common covariance function
    evaluated at the same points.

    This method is optimal, with a fixed value of :math:`d`, for variable
    selection in Gaussian binary classification problems with the same
    covariance in both classes (homoscedasticity), when all possible
    combinations of points are taken into account. That means that for all
    possible selections of :math:`t_1, \ldots, t_d`, the one in which
    :math:`\phi(t_1, \ldots, t_d)` is greater minimizes the optimal
    misclassification error of all the classification problems with the
    reduced dimensionality. For a longer discussion about the optimality and
    consistence of this method, we refer the reader to the original
    article [1]_.

    In practice the points are selected one at a time, using
    a greedy approach, so this optimality is not always guaranteed.

    Parameters:
        n_features_to_select: number of features to select.

    Examples:
        >>> from skfda.preprocessing.dim_reduction import variable_selection
        >>> from skfda.datasets import make_gaussian_process
        >>> import skfda
        >>> import numpy as np

        We create trajectories from two classes, one with zero mean and the
        other with a peak-like mean. Both have Brownian covariance.

        >>> n_samples = 10000
        >>> n_features = 200
        >>>
        >>> def mean_1(t):
        ...     return (np.abs(t - 0.25)
        ...             - 2 * np.abs(t - 0.5)
        ...             + np.abs(t - 0.75))
        >>>
        >>> X_0 = make_gaussian_process(n_samples=n_samples // 2,
        ...                             n_features=n_features,
        ...                             random_state=0)
        >>> X_1 = make_gaussian_process(n_samples=n_samples // 2,
        ...                             n_features=n_features,
        ...                             mean=mean_1,
        ...                             random_state=1)
        >>> X = skfda.concatenate((X_0, X_1))
        >>>
        >>> y = np.zeros(n_samples)
        >>> y [n_samples // 2:] = 1

        Select the relevant points to distinguish the two classes

        >>> rkvs = variable_selection.RKHSVariableSelection(
        ...                               n_features_to_select=3)
        >>> _ = rkvs.fit(X, y)
        >>> point_mask = rkvs.get_support()
        >>> points = X.grid_points[0][point_mask]
        >>> np.allclose(points, [0.25, 0.5, 0.75], rtol=1e-2)
        True

        Apply the learned dimensionality reduction

        >>> X_dimred = rkvs.transform(X)
        >>> len(X.grid_points[0])
        200
        >>> X_dimred.shape
        (10000, 3)

    References:
        .. [1] J. R. Berrendero, A. Cuevas, and J. L. Torrecilla, «On the Use
               of Reproducing Kernel Hilbert Spaces in Functional
               Classification», Journal of the American Statistical
               Association, vol. 113, no. 523, pp. 1210-1218, jul. 2018,
               doi: 10.1080/01621459.2017.1320287.

    """

    def __init__(self, n_features_to_select: int = 1) -> None:
        self.n_features_to_select = n_features_to_select

    def fit(  # type: ignore[override] # noqa: D102
        self,
        X: FDataGrid,
        y: NDArrayInt,
    ) -> RKHSVariableSelection:

        n_unique_labels = len(np.unique(y))
        if n_unique_labels != 2:
            raise ValueError(
                f"RK-VS can only be used when there are only "
                f"two different labels, but there are "
                f"{n_unique_labels}",
            )

        if X.dim_domain != 1 or X.dim_codomain != 1:
            raise ValueError("Domain and codomain dimensions must be 1")

        X, y = sklearn.utils.validation.check_X_y(X.data_matrix[..., 0], y)

        self._features_shape_ = X.shape[1:]

        features, scores = _rkhs_vs(
            X=X,
            Y=y,
            n_features_to_select=self.n_features_to_select,
        )

        self._features_ = features
        self._scores_ = scores

        return self

    def transform(  # noqa: D102
        self,
        X: FDataGrid,
        Y: None = None,
    ) -> NDArrayFloat:

        sklearn.utils.validation.check_is_fitted(self)

        X_matrix = sklearn.utils.validation.check_array(X.data_matrix[..., 0])

        if X_matrix.shape[1:] != self._features_shape_:
            raise ValueError(
                "The trajectories have a different number of "
                "points than the ones fitted",
            )

        return X_matrix[:, self._features_]  # type: ignore[no-any-return]

    def get_support(self, indices: bool = False) -> NDArrayInt:
        """
        Get a mask, or integer index, of the features selected.

        Parameters:
            indices: If True, the return value will be an array of integers,
                rather than a boolean mask.

        Returns:
            An index that selects the retained features from a `FDataGrid`
            object.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices`
            is True, this is an integer array of shape [# output features]
            whose values are indices into the input feature vector.

        """
        features = self._features_
        if indices:
            return features

        mask = np.zeros(self._features_shape_[0], dtype=bool)
        mask[features] = True
        return mask

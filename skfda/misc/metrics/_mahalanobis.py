"""Functional Mahalanobis Distance Module."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..._utils._sklearn_adapter import BaseEstimator
from ...representation import FData
from ...representation.basis import Basis
from ...typing._numpy import ArrayLike, NDArrayFloat
from .._math import inner_product_matrix
from ..regularization._regularization import TikhonovRegularization

WeightsCallable = Callable[[np.ndarray], np.ndarray]


class MahalanobisDistance(BaseEstimator):
    """Functional Mahalanobis distance.

    Class that implements functional Mahalanobis distance for both
    basis and grid representations of the data
    :footcite:`berrendero+bueno-larraz+cuevas_2020_mahalanobis`.

    Parameters:
        n_components: Number of eigenvectors to keep from
            functional principal component analysis. Defaults to 10.
        centering: If ``True`` then calculate the mean of the functional
            data object and center the data first. Defaults to ``True``.
        regularization: Regularization object to be applied.
        weights: The weights vector used for discrete integration.
            If none then Simpson's rule is used for computing the
            weights. If a callable object is passed, then the weight
            vector will be obtained by evaluating the object at the sample
            points of the passed FDataGrid object in the fit method. This
            parameter is only used when fitting a FDataGrid.
        components_basis: The basis in which we want the eigenvectors.
            We can use a different basis than the basis contained in the
            passed FDataBasis object. This parameter is only used when
            fitting a FDataBasis.
        alpha: Hyperparameter that controls the smoothness of the
            aproximation.
        eigenvectors: Eigenvectors of the covariance operator.
        eigenvalues: Eigenvalues of the covariance operator.

    Examples:
        >>> from skfda.misc.metrics import MahalanobisDistance
        >>> import numpy as np
        >>> from skfda.representation.grid import FDataGrid
        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> mahalanobis = MahalanobisDistance(2)
        >>> mahalanobis.fit(fd)
        MahalanobisDistance(n_components=2)
        >>> mahalanobis(fd[0], fd[1])
        array([ 1.99680384])

    References:
        .. footbibliography::
    """

    def __init__(
        self,
        n_components: int = 10,
        centering: bool = True,
        regularization: Optional[TikhonovRegularization[FData]] = None,
        weights: Optional[Union[ArrayLike, WeightsCallable]] = None,
        components_basis: Optional[Basis] = None,
        alpha: float = 0.001,
        eigenvalues: Optional[NDArrayFloat] = None,
        eigenvectors: Optional[FData] = None,
    ) -> None:
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = components_basis
        self.alpha = alpha
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

    def fit(
        self,
        X: FData,
        y: None = None,
    ) -> MahalanobisDistance:
        """Fit the functional Mahalanobis distance to X.

        We extract the eigenvectors and corresponding eigenvalues
        of the covariance operator by using FPCA.

        Args:
            X: Stochastic process.
            y: Ignored.

        Returns:
            self
        """
        from ...preprocessing.dim_reduction import FPCA

        if self.eigenvalues is None or self.eigenvectors is None:
            fpca = FPCA(
                n_components=self.n_components,
                centering=self.centering,
                regularization=self.regularization,
                components_basis=self.components_basis,
                _weights=self.weights,
            )
            fpca.fit(X)
            self.eigenvalues_ = fpca.explained_variance_
            self.eigenvectors_ = fpca.components_

        else:
            self.eigenvalues_ = self.eigenvalues
            self.eigenvectors_ = self.eigenvectors

        return self

    def __call__(
        self,
        e1: FData,
        e2: FData,
    ) -> NDArrayFloat:
        """Compute the squared functional Mahalanobis distances of given observations.

        Args:
            e1: First object.
            e2: Second object.

        Returns:
            Squared functional Mahalanobis distance between two observations.
        """
        try:
            check_is_fitted(self)
            eigenvalues = self.eigenvalues_
            eigenvectors = self.eigenvectors_
        except NotFittedError:
            if self.eigenvalues is not None and self.eigenvectors is not None:
                eigenvalues = self.eigenvalues
                eigenvectors = self.eigenvectors
            else:
                raise

        return np.sum(  # type: ignore[no-any-return]
            eigenvalues
            * inner_product_matrix(e1 - e2, eigenvectors) ** 2
            / (eigenvalues + self.alpha)**2,
            axis=1,
        )

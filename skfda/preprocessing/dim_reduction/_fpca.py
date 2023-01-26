"""Functional Principal Component Analysis Module."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar, Union

import numpy as np
import scipy.integrate
from scipy.linalg import solve_triangular
from sklearn.decomposition import PCA

from ..._utils._sklearn_adapter import BaseEstimator, InductiveTransformerMixin
from ...misc.regularization import L2Regularization, compute_penalty_matrix
from ...representation import FData
from ...representation.basis import Basis, FDataBasis
from ...representation.grid import FDataGrid
from ...typing._numpy import ArrayLike, NDArrayFloat

Function = TypeVar("Function", bound=FData)
WeightsCallable = Callable[[np.ndarray], np.ndarray]


class FPCA(  # noqa: WPS230 (too many public attributes)
    BaseEstimator,
    InductiveTransformerMixin[FData, NDArrayFloat, object],
):
    r"""
    Principal component analysis.

    Class that implements functional principal component analysis for both
    basis and grid representations of the data. The parameters are shared
    when fitting a FDataBasis or FDataGrid, except for
    ``components_basis``.

    Parameters:
        n_components: Number of principal components to keep from
            functional principal component analysis. Defaults to 3.
        centering: Set to ``False`` when the functional data is already known
            to be centered and there is no need to center it. Otherwise,
            the mean of the functional data object is calculated and the data
            centered before fitting . Defaults to ``True``.
        regularization: Regularization object to be applied.
        components_basis: The basis in which we want the principal
            components. We can use a different basis than the basis contained
            in the passed FDataBasis object. This parameter is only used when
            fitting a FDataBasis.

    Attributes:
        components\_: this contains the principal components.
        explained_variance\_ : The amount of variance explained by
            each of the selected components.
        explained_variance_ratio\_ : this contains the percentage
            of variance explained by each principal component.
        singular_values\_: The singular values corresponding to each of the
            selected components.
        mean\_: mean of the train data.

    Examples:
        Construct an artificial FDataBasis object and run FPCA with this
        object. The resulting principal components are not compared because
        there are several equivalent possibilities.

        >>> import skfda
        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> basis = skfda.representation.basis.MonomialBasis(
        ...     domain_range=(0,1), n_basis=2
        ... )
        >>> basis_fd = fd.to_basis(basis)
        >>> fpca_basis = FPCA(2)
        >>> fpca_basis = fpca_basis.fit(basis_fd)

        In this example we apply discretized functional PCA with some simple
        data to illustrate the usage of this class. We initialize the
        FPCA object, fit the artificial data and obtain the scores.
        The results are not tested because there are several equivalent
        possibilities.

        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fpca_grid = FPCA(2)
        >>> fpca_grid = fpca_grid.fit(fd)
    """

    def __init__(
        self,
        n_components: int = 3,
        *,
        centering: bool = True,
        regularization: Optional[L2Regularization[FData]] = None,
        components_basis: Optional[Basis] = None,
        _weights: Optional[Union[ArrayLike, WeightsCallable]] = None,
    ) -> None:
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self._weights = _weights
        self.components_basis = components_basis

    def _center_if_necessary(
        self,
        X: Function,
        *,
        learn_mean: bool = True,
    ) -> Function:

        if learn_mean:
            self.mean_ = X.mean()

        return X - self.mean_ if self.centering else X

    def _fit_basis(
        self,
        X: FDataBasis,
        y: object = None,
    ) -> FPCA:
        """
        Compute the first n_components principal components and saves them.

        The eigenvalues associated with these principal components are also
        saved. For more details about how it is implemented please view the
        referenced book.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self

        References:
            .. [RS05-8-4-2] Ramsay, J., Silverman, B. W. (2005). Basis function
                expansion of the functions. In *Functional Data Analysis*
                (pp. 161-164). Springer.

        """
        # the maximum number of components is established by the target basis
        # if the target basis is available.
        n_basis = (
            self.components_basis.n_basis
            if self.components_basis
            else X.basis.n_basis
        )
        # necessary in inverse_transform
        self.n_samples_ = X.n_samples

        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError(
                "The sample size must be bigger than the "
                "number of components",
            )

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the basis
        if self.n_components > n_basis:
            raise AttributeError(
                "The number of components should be "
                "smaller than the number of attributes of "
                "target principal components' basis.",
            )

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # setup principal component basis if not given
        components_basis = self.components_basis
        if components_basis is not None:
            # First fix domain range if not already done
            components_basis = components_basis.copy(
                domain_range=X.basis.domain_range,
            )
            g_matrix = components_basis.gram_matrix()
            # The matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            j_matrix = X.basis.inner_product_matrix(components_basis)
        else:
            # If no other basis is specified we use the same basis as the
            # passed FDataBasis object
            components_basis = X.basis.copy()
            g_matrix = components_basis.gram_matrix()
            j_matrix = g_matrix

        self._X_basis = X.basis
        self._j_matrix = j_matrix

        # Apply regularization / penalty if applicable
        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(components_basis,),
            regularization_parameter=1,
            regularization=self.regularization,
        )

        # apply regularization
        if regularization_matrix is not None:
            # using += would have a different behavior
            g_matrix = g_matrix + regularization_matrix  # noqa: WPS350

        # obtain triangulation using cholesky
        l_matrix = np.linalg.cholesky(g_matrix)

        # we need L^{-1} for a multiplication, there are two possible ways:
        # using solve to get the multiplication result directly or just invert
        # the matrix. We choose solve because it is faster and more stable.
        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = solve_triangular(
            l_matrix,
            np.transpose(j_matrix),
            lower=True,
        )

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA
        final_matrix = X.coefficients @ np.transpose(l_inv_j_t)

        # initialize the pca module provided by scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        component_coefficients = solve_triangular(
            np.transpose(l_matrix),
            np.transpose(pca.components_),
            lower=False,
        )

        self.explained_variance_ratio_: NDArrayFloat = (
            pca.explained_variance_ratio_
        )
        self.explained_variance_: NDArrayFloat = pca.explained_variance_
        self.singular_values_: NDArrayFloat = pca.singular_values_
        self.components_: FData = X.copy(
            basis=components_basis,
            coefficients=component_coefficients.T,
            sample_names=(None,) * self.n_components,
        )

        return self

    def _transform_basis(
        self,
        X: FDataBasis,
        y: object = None,
    ) -> NDArrayFloat:
        """Compute the n_components first principal components score.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            Principal component scores.

        """
        if X.basis != self._X_basis:
            raise ValueError(
                "The basis used in fit is different from "
                "the basis used in transform.",
            )

        # in this case it is the inner product of our data with the components
        return (  # type: ignore[no-any-return]
            X.coefficients @ self._j_matrix
            @ self.components_.coefficients.T
        )

    def _fit_grid(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> FPCA:
        r"""
        Compute the n_components first principal components and saves them.

        The eigenvalues associated with these principal
        components are also saved. For more details about how it is implemented
        please view the referenced book, chapter 8.

        In summary, we are performing standard multivariate PCA over
        :math:`\mathbf{X} \mathbf{W}^{1/2}` where :math:`\mathbf{X}` is the
        data matrix and :math:`\mathbf{W}` is the weight matrix (this matrix
        defines the numerical integration). By default the weight matrix is
        obtained using the trapezoidal rule.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self.

        References:
            .. [RS05-8-4-1] Ramsay, J., Silverman, B. W. (2005). Discretizing
                the functions. In *Functional Data Analysis* (p. 161).
                Springer.

        """
        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError(
                "The sample size must be bigger than the "
                "number of components",
            )

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the funcional data object
        if self.n_components > X.data_matrix.shape[1]:
            raise AttributeError(
                "The number of components should be "
                "smaller than the number of discretization "
                "points of the functional data object.",
            )

        # data matrix initialization
        fd_data = X.data_matrix.reshape(X.data_matrix.shape[:-1])

        # get the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # necessary for inverse_transform
        self.n_samples_ = n_samples

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # establish weights for each point of discretization
        if self._weights is None:
            # grid_points is a list with one array in the 1D case
            identity = np.eye(len(X.grid_points[0]))
            self._weights = scipy.integrate.simps(identity, X.grid_points[0])
        elif callable(self._weights):
            self._weights = self._weights(X.grid_points[0])
            # if its a FDataGrid then we need to reduce the dimension to 1-D
            # array
            if isinstance(self._weights, FDataGrid):
                self._weights = np.squeeze(self._weights.data_matrix)
        else:
            self._weights = self._weights

        weights_matrix = np.diag(self._weights)

        basis = FDataGrid(
            data_matrix=np.identity(n_points_discretization),
            grid_points=X.grid_points,
        )

        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(basis,),
            regularization_parameter=1,
            regularization=self.regularization,
        )

        # See issue #497 for more information about this approach
        factorization_matrix = weights_matrix.astype(float)
        if self.regularization is not None:
            factorization_matrix += regularization_matrix

        # Tranpose of the Cholesky decomposition
        Lt = np.linalg.cholesky(factorization_matrix).T

        new_data_matrix = fd_data @ weights_matrix
        new_data_matrix = np.linalg.solve(Lt.T, new_data_matrix.T).T

        pca = PCA(n_components=self.n_components)
        pca.fit(new_data_matrix)

        components = pca.components_
        components = np.linalg.solve(Lt, pca.components_.T).T

        self.components_ = X.copy(
            data_matrix=components,
            sample_names=(None,) * self.n_components,
        )

        self.explained_variance_ratio_ = (
            pca.explained_variance_ratio_
        )
        self.explained_variance_ = pca.explained_variance_
        self.singular_values_ = pca.singular_values_

        return self

    def _transform_grid(
        self,
        X: FDataGrid,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components score.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            Principal component scores.

        """
        # in this case its the coefficient matrix multiplied by the principal
        # components as column vectors

        return (  # type: ignore[no-any-return]
            X.data_matrix.reshape(X.data_matrix.shape[:-1])
            * self._weights
            @ np.transpose(
                self.components_.data_matrix.reshape(
                    self.components_.data_matrix.shape[:-1],
                ),
            )
        )

    def fit(
        self,
        X: FData,
        y: object = None,
    ) -> FPCA:
        """
        Compute the n_components first principal components and saves them.

        Args:
            X: The functional data object to be analysed.
            y: Ignored.

        Returns:
            self

        """
        if isinstance(X, FDataGrid):
            return self._fit_grid(X, y)
        elif isinstance(X, FDataBasis):
            return self._fit_basis(X, y)

        raise AttributeError("X must be either FDataGrid or FDataBasis")

    def transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the ``n_components`` first principal components scores.

        Args:
            X: The functional data object to be analysed.
            y: Only present because of fit function convention

        Returns:
            Principal component scores.

        """
        X = self._center_if_necessary(X, learn_mean=False)

        if isinstance(X, FDataGrid):
            return self._transform_grid(X, y)
        elif isinstance(X, FDataBasis):
            return self._transform_basis(X, y)

        raise AttributeError("X must be either FDataGrid or FDataBasis")

    def fit_transform(
        self,
        X: FData,
        y: object = None,
    ) -> NDArrayFloat:
        """
        Compute the n_components first principal components and their scores.

        Args:
            X: The functional data object to be analysed.
            y: Ignored

        Returns:
            Principal component scores.

        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(
        self,
        pc_scores: NDArrayFloat,
    ) -> FData:
        """
        Compute the recovery from the fitted principal components scores.

        In other words,
        it maps ``pc_scores``, from the fitted functional PCs' space,
        back to the input functional space.
        ``pc_scores`` might be an array returned by ``transform`` method.

        Args:
            pc_scores: ndarray (n_samples, n_components).

        Returns:
            A FData object.

        """
        # check the instance is fitted.

        # input format check:
        if isinstance(pc_scores, np.ndarray):
            if pc_scores.ndim == 1:
                pc_scores = pc_scores[np.newaxis, :]

            if pc_scores.shape[1] != self.n_components:
                raise AttributeError(
                    "pc_scores must be a numpy array "
                    "with n_samples rows and n_components columns.",
                )
        else:
            raise AttributeError("pc_scores is not a numpy array.")

        # inverse_transform is slightly different whether
        # .fit was applied to FDataGrid or FDataBasis object
        # Does not work (boundary problem in x_hat and bias reconstruction)
        if isinstance(self.components_, FDataGrid):

            additional_args = {
                "data_matrix": np.einsum(
                    "nc,c...->n...",
                    pc_scores,
                    self.components_.data_matrix,
                ),
            }

        elif isinstance(self.components_, FDataBasis):

            additional_args = {
                "coefficients": pc_scores @ self.components_.coefficients,
            }

        return (
            self.mean_.copy(
                **additional_args,
                sample_names=(None,) * len(pc_scores),
            )
            + self.mean_
        )

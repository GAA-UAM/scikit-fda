"""Functional Principal Component Analysis Module."""

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from skfda.misc.regularization import compute_penalty_matrix
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid

__author__ = "Yujian Hong"
__email__ = "yujian.hong@estudiante.uam.es"


class FPCA(BaseEstimator, TransformerMixin):
    """Class that implements functional principal component analysis for both
    basis and grid representations of the data. Most parameters are shared
    when fitting a FDataBasis or FDataGrid, except weights and components_basis.

    Parameters:
        n_components (int): number of principal components to obtain from
            functional principal component analysis. Defaults to 3.
        centering (bool): if True then calculate the mean of the functional data
            object and center the data first. Defaults to True. If True the
            passed FDataBasis object is modified.
        regularization (Regularization):
            Regularization object to be applied.
        components_basis (Basis): the basis in which we want the principal
            components. We can use a different basis than the basis contained in
            the passed FDataBasis object. This parameter is only used when
            fitting a FDataBasis.
        weights (numpy.array or callable): the weights vector used for
            discrete integration. If none then the trapezoidal rule is used for
            computing the weights. If a callable object is passed, then the
            weight vector will be obtained by evaluating the object at the
            sample points of the passed FDataGrid object in the fit method.
            This parameter is only used when fitting a FDataGrid.

    Attributes:
        components_ (FData): this contains the principal components in a
            basis representation.
        explained_variance_ (array_like): The amount of variance explained by
            each of the selected components.
        explained_variance_ratio_ (array_like): this contains the percentage of
            variance explained by each principal component.
        mean_ (FData): mean of the train data.


    Examples:
        Construct an artificial FDataBasis object and run FPCA with this object.
        The resulting principal components are not compared because there are
        several equivalent possibilities.

        >>> import skfda
        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> grid_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> basis = skfda.representation.basis.Monomial(
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

    def __init__(self,
                 n_components=3,
                 centering=True,
                 regularization=None,
                 weights=None,
                 components_basis=None
                 ):
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.weights = weights
        self.components_basis = components_basis

    def _center_if_necessary(self, X, *, learn_mean=True):

        if learn_mean:
            self.mean_ = X.mean()

        return X - self.mean_ if self.centering else X

    def _fit_basis(self, X: FDataBasis, y=None):
        """Computes the first n_components principal components and saves them.
        The eigenvalues associated with these principal components are also
        saved. For more details about how it is implemented please view the
        referenced book.

        Args:
            X (FDataBasis):
                the functional data object to be analysed in basis
                representation
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)

        References:
            .. [RS05-8-4-2] Ramsay, J., Silverman, B. W. (2005). Basis function
                expansion of the functions. In *Functional Data Analysis*
                (pp. 161-164). Springer.

        """

        # the maximum number of components is established by the target basis
        # if the target basis is available.
        n_basis = (self.components_basis.n_basis if self.components_basis
                   else X.basis.n_basis)
        n_samples = X.n_samples

        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError("The sample size must be bigger than the "
                                 "number of components")

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the basis
        if self.n_components > n_basis:
            raise AttributeError("The number of components should be "
                                 "smaller than the number of attributes of "
                                 "target principal components' basis.")

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # setup principal component basis if not given
        components_basis = self.components_basis
        if components_basis is not None:
            # First fix domain range if not already done
            components_basis = components_basis.copy(
                domain_range=X.basis.domain_range)
            g_matrix = components_basis.gram_matrix()
            # the matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            j_matrix = X.basis.inner_product_matrix(components_basis)
        else:
            # if no other basis is specified we use the same basis as the passed
            # FDataBasis Object
            components_basis = X.basis.copy()
            g_matrix = components_basis.gram_matrix()
            j_matrix = g_matrix

        self._X_basis = X.basis
        self._j_matrix = j_matrix

        # Apply regularization / penalty if applicable
        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(components_basis,),
            regularization_parameter=1,
            regularization=self.regularization)

        # apply regularization
        g_matrix = (g_matrix + regularization_matrix)

        # obtain triangulation using cholesky
        l_matrix = np.linalg.cholesky(g_matrix)

        # we need L^{-1} for a multiplication, there are two possible ways:
        # using solve to get the multiplication result directly or just invert
        # the matrix. We choose solve because it is faster and more stable.
        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = solve_triangular(l_matrix, np.transpose(j_matrix),
                                     lower=True)

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA
        final_matrix = (X.coefficients @ np.transpose(l_inv_j_t) /
                        np.sqrt(n_samples))

        # initialize the pca module provided by scikit-learn
        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        component_coefficients = solve_triangular(np.transpose(l_matrix),
                                                  np.transpose(
                                                      pca.components_),
                                                  lower=False)

        component_coefficients = np.transpose(component_coefficients)

        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.explained_variance_ = pca.explained_variance_
        self.components_ = X.copy(basis=components_basis,
                                  coefficients=component_coefficients,
                                  sample_names=(None,) * self.n_components)

        return self

    def _transform_basis(self, X, y=None):
        """Computes the n_components first principal components score and
        returns them.

        Args:
            X (FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present because of fit function convention

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """

        if X.basis != self._X_basis:
            raise ValueError("The basis used in fit is different from "
                             "the basis used in transform.")

        # in this case it is the inner product of our data with the components
        return (X.coefficients @ self._j_matrix
                @ self.components_.coefficients.T)

    def _fit_grid(self, X: FDataGrid, y=None):
        r"""Computes the n_components first principal components and saves them.

        The eigenvalues associated with these principal
        components are also saved. For more details about how it is implemented
        please view the referenced book, chapter 8.

        In summary, we are performing standard multivariate PCA over
        :math:`\frac{1}{\sqrt{N}} \mathbf{X} \mathbf{W}^{1/2}` where :math:`N`
        is the number of samples in the dataset, :math:`\mathbf{X}` is the data
        matrix and :math:`\mathbf{W}` is the weight matrix (this matrix
        defines the numerical integration). By default the weight matrix is
        obtained using the trapezoidal rule.

        Args:
            X (FDataGrid):
                the functional data object to be analysed in basis
                representation
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)

        References:
            .. [RS05-8-4-1] Ramsay, J., Silverman, B. W. (2005). Discretizing
            the functions. In *Functional Data Analysis* (p. 161). Springer.
        """

        # check that the number of components is smaller than the sample size
        if self.n_components > X.n_samples:
            raise AttributeError("The sample size must be bigger than the "
                                 "number of components")

        # check that we do not exceed limits for n_components as it should
        # be smaller than the number of attributes of the funcional data object
        if self.n_components > X.data_matrix.shape[1]:
            raise AttributeError("The number of components should be "
                                 "smaller than the number of discretization "
                                 "points of the functional data object.")

        # data matrix initialization
        fd_data = X.data_matrix.reshape(X.data_matrix.shape[:-1])

        # get the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        X = self._center_if_necessary(X)

        # establish weights for each point of discretization
        if not self.weights:
            # grid_points is a list with one array in the 1D case
            # in trapezoidal rule, suppose \deltax_k = x_k - x_{k-1}, the weight
            # vector is as follows: [\deltax_1/2, \deltax_1/2 + \deltax_2/2,
            # \deltax_2/2 + \deltax_3/2, ... , \deltax_n/2]
            differences = np.diff(X.grid_points[0])
            differences = np.concatenate(((0,), differences, (0,)))
            self.weights = (differences[:-1] + differences[1:]) / 2
        elif callable(self.weights):
            self.weights = self.weights(X.grid_points[0])
            # if its a FDataGrid then we need to reduce the dimension to 1-D
            # array
            if isinstance(self.weights, FDataGrid):
                self.weights = np.squeeze(self.weights.data_matrix)

        weights_matrix = np.diag(self.weights)

        basis = FDataGrid(
            data_matrix=np.identity(n_points_discretization),
            grid_points=X.grid_points
        )

        regularization_matrix = compute_penalty_matrix(
            basis_iterable=(basis,),
            regularization_parameter=1,
            regularization=self.regularization)

        fd_data = np.transpose(np.linalg.solve(
            np.transpose(basis.data_matrix[..., 0] + regularization_matrix),
            np.transpose(fd_data)))

        # see docstring for more information
        final_matrix = fd_data @ np.sqrt(weights_matrix) / np.sqrt(n_samples)

        pca = PCA(n_components=self.n_components)
        pca.fit(final_matrix)
        self.components_ = X.copy(data_matrix=np.transpose(
            np.linalg.solve(np.sqrt(weights_matrix),
                            np.transpose(pca.components_))),
            sample_names=(None,) * self.n_components)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.explained_variance_ = pca.explained_variance_

        return self

    def _transform_grid(self, X: FDataGrid, y=None):
        """Computes the n_components first principal components score and
        returns them.

        Args:
            X (FDataGrid):
                the functional data object to be analysed
            y (None, not used):
                only present because of fit function convention

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """

        # in this case its the coefficient matrix multiplied by the principal
        # components as column vectors

        return X.data_matrix.reshape(
            X.data_matrix.shape[:-1]) @ np.transpose(
            self.components_.data_matrix.reshape(
                self.components_.data_matrix.shape[:-1]))

    def fit(self, X, y=None):
        """Computes the n_components first principal components and saves them
        inside the FPCA object, both FDataGrid and FDataBasis are accepted

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)
        """
        if isinstance(X, FDataGrid):
            return self._fit_grid(X, y)
        elif isinstance(X, FDataBasis):
            return self._fit_basis(X, y)
        else:
            raise AttributeError("X must be either FDataGrid or FDataBasis")

    def transform(self, X, y=None):
        """Computes the n_components first principal components score and
        returns them.

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present because of fit function convention

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """
        X = self._center_if_necessary(X, learn_mean=False)

        if isinstance(X, FDataGrid):
            return self._transform_grid(X, y)
        elif isinstance(X, FDataBasis):
            return self._transform_basis(X, y)
        else:
            raise AttributeError("X must be either FDataGrid or FDataBasis")

    def fit_transform(self, X, y=None, **fit_params):
        """Computes the n_components first principal components and their scores
        and returns them.
        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present for convention of a fit function

        Returns:
            (array_like): the scores of the data with reference to the
            principal components
        """
        self.fit(X, y)
        return self.transform(X, y)

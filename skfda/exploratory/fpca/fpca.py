"""Functional Principal Component Analysis Module."""

import numpy as np
import skfda
from abc import ABC, abstractmethod
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, LeaveOneOut

__author__ = "Yujian Hong"
__email__ = "yujian.hong@estudiante.uam.es"


class FPCA(ABC, BaseEstimator, TransformerMixin):
    """Defines the common structure shared between classes that do functional
    principal component analysis

    Attributes:
        n_components (int): number of principal components to obtain from
            functional principal component analysis. Defaults to 3.
        centering (bool): if True then calculate the mean of the functional data
            object and center the data first
        components (FDataGrid or FDataBasis): this contains the principal
            components either in a basis form or discretized form
        component_values (array_like): this contains the values (eigenvalues)
            associated with the principal components
        pca (sklearn.decomposition.PCA): object for principal component analysis.
            In both cases (discretized FPCA and basis FPCA) the problem can be
            reduced to a regular PCA problem and use the framework provided by
            sklearn to continue.
    """

    def     __init__(self, n_components=3, centering=True):
        """FPCA constructor

        Args:
            n_components (int): number of principal components to obtain from
                functional principal component analysis
            centering (bool): if True then calculate the mean of the functional
                data object and center the data first. Defaults to True
        """
        self.n_components = n_components
        self.centering = centering
        self.components = None
        self.component_values = None
        self.pca = PCA(n_components=self.n_components)

    @abstractmethod
    def fit(self, X, y=None):
        """Computes the n_components first principal components and saves them
        inside the FPCA object.

        Args:
            X (FDataGrid or FDataBasis):
                the functional data object to be analysed
            y (None, not used):
                only present for convention of a fit function

        Returns:
            self (object)
        """
        pass

    @abstractmethod
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
        pass

    def fit_transform(self, X, y=None, **fit_params):
        """
        Computes the n_components first principal components and their scores
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


class FPCABasis(FPCA):
    """Funcional principal component analysis for functional data represented
    in basis form.

    Attributes:
        n_components (int): number of principal components to obtain from
            functional principal component analysis. Defaults to 3.
        centering (bool): if True then calculate the mean of the functional data
            object and center the data first. Defaults to True. If True the
            passed FDataBasis object is modified.
        components (FDataBasis): this contains the principal components either
            in a basis form.
        components_basis (Basis): the basis in which we want the principal
            components. We can use a different basis than the basis contained in
            the passed FDataBasis object.
        component_values (array_like): this contains the values (eigenvalues)
            associated with the principal components.
        pca (sklearn.decomposition.PCA): object for principal component analysis.
            In both cases (discretized FPCA and basis FPCA) the problem can be
            reduced to a regular PCA problem and use the framework provided by
            sklearn to continue.

    Examples:
        Construct an artificial FDataBasis object and run FPCA with this object.

        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> sample_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> basis = skfda.representation.basis.Monomial((0,1), n_basis=2)
        >>> basis_fd = fd.to_basis(basis)
        >>> fpca_basis = FPCABasis(2)
        >>> fpca_basis = fpca_basis.fit(basis_fd)
        >>> fpca_basis.components.coefficients
        array([[ 1.        , -3.        ],
               [-1.73205081,  1.73205081]])

    """

    def __init__(self,
                 n_components=3,
                 components_basis=None,
                 centering=True,
                 regularization_derivative_degree=2,
                 regularization_coefficients=None,
                 regularization_parameter=0):
        """FPCABasis constructor

        Args:
            n_components (int): number of principal components to obtain from
                functional principal component analysis
            components_basis (skfda.representation.Basis): the basis in which we
                want the principal components. Defaults to None. If so, the
                basis contained in the passed FDataBasis object for the fit
                function will be used.
            centering (bool): if True then calculate the mean of the functional
                data object and center the data first. Defaults to True
        """
        super().__init__(n_components, centering)
        # basis that we want to use for the principal components
        self.components_basis = components_basis
        # lambda in the regularization / penalization process
        self.regularization_parameter = regularization_parameter
        self.regularization_derivative_degree = regularization_derivative_degree
        self.regularization_coefficients = regularization_coefficients

    def fit(self, X: FDataBasis, y=None):
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
        n_basis = self.components_basis.n_basis if self.components_basis \
            else X.basis.n_basis
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
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # subtract from each row the mean coefficient matrix
            X.coefficients -= meanfd.coefficients

        # setup principal component basis if not given
        if self.components_basis:
            # First fix domain range if not already done
            self.components_basis.domain_range = X.basis.domain_range
            g_matrix = self.components_basis.gram_matrix()
            # the matrix that are in charge of changing the computed principal
            # components to target matrix is essentially the inner product
            # of both basis.
            j_matrix = X.basis.inner_product(self.components_basis)
        else:
            # if no other basis is specified we use the same basis as the passed
            # FDataBasis Object
            self.components_basis = X.basis.copy()
            g_matrix = self.components_basis.gram_matrix()
            j_matrix = g_matrix

        # make g matrix symmetric, referring to Ramsay's implementation
        g_matrix = (g_matrix + np.transpose(g_matrix)) / 2

        # Apply regularization / penalty if applicable
        if self.regularization_parameter > 0:
            # obtain regularization matrix
            regularization_matrix = self.components_basis.penalty(
                self.regularization_derivative_degree,
                self.regularization_coefficients)
            # apply regularization
            g_matrix = g_matrix + self.regularization_parameter \
                * regularization_matrix

        # obtain triangulation using cholesky
        l_matrix = np.linalg.cholesky(g_matrix)

        # we need L^{-1} for a multiplication, there are two possible ways:
        # using solve to get the multiplication result directly or just invert
        # the matrix. We choose solve because it is faster and more stable.
        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = np.linalg.solve(l_matrix, np.transpose(j_matrix))

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for PCA
        final_matrix = X.coefficients @ np.transpose(l_inv_j_t) / \
                       np.sqrt(n_samples)

        self.pca.fit(final_matrix)

        # we choose solve to obtain the component coefficients for the
        # same reason: it is faster and more efficient
        component_coefficients = np.linalg.solve(np.transpose(l_matrix),
                                                 np.transpose(self.pca.components_))

        component_coefficients = np.transpose(component_coefficients)

        # the singular values obtained using SVD are the squares of eigenvalues
        self.component_values = self.pca.singular_values_ ** 2
        self.components = X.copy(basis=self.components_basis,
                                 coefficients=component_coefficients)

        return self

    def transform(self, X, y=None):
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

        # in this case it is the inner product of our data with the components
        return X.inner_product(self.components)


class FPCADiscretized(FPCA):
    """Funcional principal component analysis for functional data represented
    in discretized form.

    Attributes:
        n_components (int): number of principal components to obtain from
            functional principal component analysis. Defaults to 3.
        centering (bool): if True then calculate the mean of the functional data
            object and center the data first. Defaults to True. If True the
            passed FDataBasis object is modified.
        components (FDataBasis): this contains the principal components either
            in a basis form.
        components_basis (Basis): the basis in which we want the principal
            components. We can use a different basis than the basis contained in
            the passed FDataBasis object.
        component_values (array_like): this contains the values (eigenvalues)
            associated with the principal components.
        pca (sklearn.decomposition.PCA): object for principal component analysis.
            In both cases (discretized FPCA and basis FPCA) the problem can be
            reduced to a regular PCA problem and use the framework provided by
            sklearn to continue.

    Examples:
        In this example we apply discretized functional PCA with some simple
        data to illustrate the usage of this class. We initialize the
        FPCADiscretized object, fit the artificial data and obtain the scores.

        >>> data_matrix = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> sample_points = [0, 1]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> fpca_discretized = FPCADiscretized(2)
        >>> fpca_discretized = fpca_discretized.fit(fd)
        >>> fpca_discretized.components.data_matrix
        array([[[-0.4472136 ],
                [ 0.89442719]],
        <BLANKLINE>
               [[-0.89442719],
                [-0.4472136 ]]])
        >>> fpca_discretized.transform(fd)
        array([[-1.11803399e+00,  5.55111512e-17],
               [ 1.11803399e+00, -5.55111512e-17]])
    """

    def __init__(self, n_components=3, weights=None, centering=True):
        """FPCABasis constructor

        Args:
            n_components (int): number of principal components to obtain from
                functional principal component analysis
            weights (numpy.array): the weights vector used for discrete
                integration. If none then the trapezoidal rule is used for
                computing the weights.
            centering (bool): if True then calculate the mean of the functional
                data object and center the data first. Defaults to True
        """
        super().__init__(n_components, centering)
        self.weights = weights

    def fit(self, X: FDataGrid, y=None):
        """Computes the n_components first principal components and saves them
        inside the FPCA object.The eigenvalues associated with these principal
        components are also saved. For more details about how it is implemented
        please view the referenced book, chapter 8.

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
        fd_data = np.squeeze(X.data_matrix)

        # get the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # if centering is True then subtract the mean function to each function
        # in FDataBasis
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # subtract from each row the mean coefficient matrix
            fd_data -= np.squeeze(meanfd.data_matrix)

        # establish weights for each point of discretization
        if not self.weights:
            # sample_points is a list with one array in the 1D case
            # in trapezoidal rule, suppose \deltax_k = x_k - x_{k-1}, the weight
            # vector is as follows: [\deltax_1/2, \deltax_1/2 + \deltax_2/2,
            # \deltax_2/2 + \deltax_3/2, ... , \deltax_n/2]
            differences = np.diff(X.sample_points[0])
            self.weights = [sum(differences[i:i + 2]) / 2 for i in
                            range(len(differences))]
            self.weights = np.concatenate(([differences[0] / 2], self.weights))

        weights_matrix = np.diag(self.weights)

        final_matrix = fd_data @ np.sqrt(weights_matrix) / np.sqrt(n_samples)
        self.pca.fit(final_matrix)
        self.components = X.copy(data_matrix=self.pca.components_)
        self.component_values = self.pca.singular_values_ ** 2

        return self

    def transform(self, X, y=None):
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
        return np.squeeze(X.data_matrix) @ np.transpose(
            np.squeeze(self.components.data_matrix))

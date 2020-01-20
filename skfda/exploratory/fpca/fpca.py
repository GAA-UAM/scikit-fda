import numpy as np
from abc import ABC, abstractmethod
from skfda.representation.basis import FDataBasis
from skfda.representation.grid import FDataGrid
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin


class FPCA(ABC, BaseEstimator, ClassifierMixin):
    """Defines the common structure shared between classes that do functional principal component analysis

    Attributes:
        n_components (int): number of principal components to obtain from functional principal component analysis
        centering (bool): if True then calculate the mean of the functional data object and center the data first
        svd (bool): if True then we use svd to obtain the principal components. Otherwise we use eigenanalysis
        components (FDataGrid or FDataBasis): this contains the principal components either in a basis form or
            discretized form
        component_values (array_like): this contains the values (eigenvalues) associated with the principal components

    """

    def __init__(self, n_components=3, centering=True):
        """ FPCA constructor
        Args:
            n_components (int): number of principal components to obtain from functional principal component analysis
            centering (bool): if True then calculate the mean of the functional data object and center the data first.
                Defaults to True
            svd (bool): if True then we use svd to obtain the principal components. Otherwise we use eigenanalysis.
                Defaults to True as svd is usually more efficient
        """
        self.n_components = n_components
        self.centering = centering
        self.components = None
        self.component_values = None

    @abstractmethod
    def fit(self, X, y=None):
        """Computes the n_components first principal components and saves them inside the FPCA object.

            Args:
                X (FDataGrid or FDataBasis): the functional data object to be analysed
                y (None, not used): only present for convention of a fit function

            Returns:
                self (object)
        """
        pass

    @abstractmethod
    def transform(self, X, y=None):
        """Computes the n_components first principal components score and returns them.

            Args:
                X (FDataGrid or FDataBasis): the functional data object to be analysed
                y (None, not used): only present for convention of a fit function

            Returns:
                (array_like): the scores of the data with reference to the principal components
        """
        pass

    def fit_transform(self, X, y=None):
        """Computes the n_components first principal components and their scores and returns them.

            Args:
                X (FDataGrid or FDataBasis): the functional data object to be analysed
                y (None, not used): only present for convention of a fit function

            Returns:
                (array_like): the scores of the data with reference to the principal components
        """
        self.fit(X, y)
        return self.transform(X, y)


class FPCABasis(FPCA):

    def __init__(self, n_components=3, components_basis=None, centering=True):
        super().__init__(n_components, centering)
        # component_basis is the basis that we want to use for the principal components
        self.components_basis = components_basis

    def fit(self, X: FDataBasis, y=None):
        # initialize pca
        self.pca = PCA(n_components=self.n_components)

        # if centering is True then substract the mean function to each function in FDataBasis
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # substract from each row the mean coefficient matrix
            X.coefficients -= meanfd.coefficients

        # for reference, X.coefficients is the C matrix
        n_samples, n_basis = X.coefficients.shape

        # setup principal component basis if not given
        if self.components_basis:
            # if the principal components are in the same basis, this is essentially the gram matrix
            g_matrix = self.components_basis.gram_matrix()
            j_matrix = X.basis.inner_product(self.components_basis)
        else:
            self.components_basis = X.basis.copy()
            g_matrix = self.components_basis.gram_matrix()
            j_matrix = g_matrix

        l_matrix = np.linalg.cholesky(g_matrix)

        # L^{-1}
        l_matrix_inv = np.linalg.inv(l_matrix)

        # The following matrix is needed: L^{-1}*J^T
        l_inv_j_t = l_matrix_inv @ np.transpose(j_matrix)

        # TODO make the final matrix symmetric, not necessary as the final matrix is not a square matrix?

        # the final matrix, C(L-1Jt)t for svd or (L-1Jt)-1CtC(L-1Jt)t for eigen analysis
        final_matrix = X.coefficients @ np.transpose(l_inv_j_t) / np.sqrt(n_samples)

        self.pca.fit(final_matrix)
        self.component_values = self.pca.singular_values_ ** 2
        self.components = X.copy(basis=self.components_basis,
                                 coefficients=self.pca.components_ @ l_matrix_inv)
        """
        if self.svd:
            # vh contains the eigenvectors transposed
            # s contains the singular values, which are square roots of eigenvalues
            u, s, vh = np.linalg.svd(final_matrix, full_matrices=True, compute_uv=True)
            principal_components = vh @ l_matrix_inv
            self.components = X.copy(basis=self.components_basis,
                                     coefficients=principal_components[:self.n_components, :])
            self.component_values = s ** 2
        else:
            final_matrix = np.transpose(final_matrix) @ final_matrix

            # perform eigenvalue and eigenvector analysis on this matrix
            # eigenvectors is a numpy array, such that its columns are eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(final_matrix)

            # sort the eigenvalues and eigenvectors from highest to lowest
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            principal_components_t = np.transpose(l_matrix_inv) @ eigenvectors

            # we only want the first ones, determined by n_components
            principal_components_t = principal_components_t[:, :self.n_components]

            self.components = X.copy(basis=self.components_basis,
                                     coefficients=np.transpose(principal_components_t))

            self.component_values = eigenvalues
        """

        return self

    def transform(self, X, y=None):
        # in this case it is the inner product of our data with the components
        return X.inner_product(self.components)


class FPCADiscretized(FPCA):
    def __init__(self, n_components=3, weights=None, centering=True):
        super().__init__(n_components, centering)
        self.weights = weights

    # noinspection PyPep8Naming
    def fit(self, X: FDataGrid, y=None):
        # initialize pca module
        self.pca = PCA(n_components=self.n_components)

        # data matrix initialization
        fd_data = np.squeeze(X.data_matrix)

        # obtain the number of samples and the number of points of descretization
        n_samples, n_points_discretization = fd_data.shape

        # if centering is True then subtract the mean function to each function in FDataBasis
        if self.centering:
            meanfd = X.mean()
            # consider moving these lines to FDataBasis as a centering function
            # subtract from each row the mean coefficient matrix
            fd_data -= np.squeeze(meanfd.data_matrix)

        # establish weights for each point of discretization
        if not self.weights:
            # sample_points is a list with one array in the 1D case
            # in trapezoidal rule, suppose \deltax_k = x_k - x_{k-1}, the weight vector is as follows:
            # [\deltax_1/2, \deltax_1/2 + \deltax_2/2, \deltax_2/2 + \deltax_3/2, ... , \deltax_n/2]
            differences = np.diff(X.sample_points[0])
            self.weights = [sum(differences[i:i + 2]) / 2 for i in range(len(differences))]
            self.weights = np.concatenate(([differences[0] / 2], self.weights))

        weights_matrix = np.diag(self.weights)

        # k_estimated is not used for the moment
        # k_estimated = fd_data @ np.transpose(fd_data) / n_samples

        final_matrix = fd_data @ np.sqrt(weights_matrix) / np.sqrt(n_samples)
        self.pca.fit(final_matrix)
        self.components = X.copy(data_matrix=self.pca.components_)
        self.component_values = self.pca.singular_values_**2

        """
        if self.svd:
            # vh contains the eigenvectors transposed
            # s contains the singular values, which are square roots of eigenvalues
            u, s, vh = np.linalg.svd(final_matrix, full_matrices=True, compute_uv=True)
            self.components = X.copy(data_matrix=vh[:self.n_components, :])
            self.component_values = s**2
        else:
            # perform eigenvalue and eigenvector analysis on this matrix
            # eigenvectors is a numpy array, such that its columns are eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(np.transpose(final_matrix) @ final_matrix)

            # sort the eigenvalues and eigenvectors from highest to lowest
            # the eigenvectors are the principal components
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            principal_components_t = eigenvectors[:, idx]

            # we only want the first ones, determined by n_components
            principal_components_t = principal_components_t[:, :self.n_components]

            # prepare the computed principal components
            self.components = X.copy(data_matrix=np.transpose(principal_components_t))
            self.component_values = eigenvalues
        """
        return self

    def transform(self, X, y=None):
        # in this case its the coefficient matrix multiplied by the principal components as column vectors
        return np.squeeze(X.data_matrix) @ np.transpose(np.squeeze(self.components.data_matrix))

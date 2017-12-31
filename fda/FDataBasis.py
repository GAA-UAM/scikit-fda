"""This module defines functional data object in a functional basis
representation and the corresponding basis class.

"""

from abc import ABC, abstractmethod

import numpy
import matplotlib.pyplot
import scipy.interpolate
import scipy.linalg

__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


class Basis(ABC):
    """Defines a functional basis.

    Attributes:
        def_range (tuple): a tuple of length 2 containing the initial and end
            values of the interval over which the basis can be evaluated.
        nbasis (int): number of functions in the basis.

    """

    def __init__(self, def_range=(0, 1), nbasis=1):
        """Basis constructor.

        Args:
            def_range (tuple, optional): Definition of the interval where the
                basis defines a space. Defaults to (0,1).
            nbasis: Number of functions that form the basis. Defaults to 1.
        """
        # Some checks
        if def_range[0] >= def_range[1]:
            raise ValueError("The interval {} is not well-defined.".format(
                def_range))
        if nbasis < 1:
            raise ValueError("The number of basis has to be strictly "
                             "possitive.")
        self.def_range = def_range
        self.nbasis = nbasis
        self._drop_index_lst = []

        super().__init__()

    @abstractmethod
    def compute_matrix(self, eval_points):
        """Computes the basis at a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis at the values specified in eval_points.

        """
        pass

    @abstractmethod
    def evaluate(self, eval_points):
        """Evaluates the basis at a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis at the values specified in eval_points.

        """
        eval_points = numpy.asarray(eval_points)
        if numpy.any(numpy.isnan(eval_points)):
            raise ValueError("The list of points where the function is "
                             "evaluated can not contain nan values.")

        # TODO include evaluation at derivatives
        return self.compute_matrix(eval_points)

    @abstractmethod
    def plot(self, **kwargs):
        """Plots the basis object.

        Args:
            **kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """

        # Number of points where the basis are evaluated
        npoints = max(501, 10 * self.nbasis)
        # List of points where the basis are evaluated
        eval_points = numpy.linspace(self.def_range[0], self.def_range[1],
                                     npoints)
        # Basis evaluated in the previous list of points
        mat = self.evaluate(eval_points)
        # Plot
        return matplotlib.pyplot.plot(eval_points, mat.T, **kwargs)


class Monomial(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:
    .. math::
        1, t, t^2, t^3...

    Examples:
        Defines a monomial base over the interval :math:`[0, 5]` consisting
        on the first 3 powers of :math:`t`: :math:`1, t, t^2`.

        >>> bs_mon = Monomial((0,5), nbasis=3)

        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> bs_mon.evaluate([0, 1, 2])
        array([[ 1.,  1.,  1.],
               [ 0.,  1.,  2.],
               [ 0.,  1.,  4.]])

    """
    def __init__(self, def_range=(0, 1), nbasis=1):
        super().__init__(def_range, nbasis)

    def plot(self, **kwargs):
        return super().plot(**kwargs)

    def evaluate(self, eval_points):
        return super().evaluate(eval_points)

    def compute_matrix(self, eval_points):
        """Computes the basis at a list of values.

        For each of the basis computes its value for each of the points in
        the list passed as argument to the method.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis at the values specified in eval_points.

        """
        # Initialise empty matrix
        mat = numpy.empty((self.nbasis, len(eval_points)))
        
        # For each basis computes its value for each evaluation point
        for i in range(self.nbasis):
            mat[i] = eval_points ** i

        return mat


class BSpline(Basis):
    r"""BSpline basis.

    BSpline basis elements are defined recursively as:
    .. math::
        B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}
        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    Where k indicates the order of the spline.

    Implementation details: In order to work correctly on the ends of the
    interval where the basis is defined is usually necessary to have these
    values at the ends as knots several times. This class handles this so that
    knots don't have to be duplicated.

    Examples:
        Constructs specifying number of basis and order.

        >>> bss = BSpline(nbasis=8, order=4)

        If no order is specified defaults to 4 because cubic splines are
        the most used. So the previous example is the same as:

        >>> bss = BSpline(nbasis=8)

        It is also possible to create a BSpline basis specifying the knots.

        >>> bss = BSpline(knots=[0, 0.2, 0.4, 0.6, 0.8, 1])

        Once we create a basis we can evaluate each of its functions at a
        set of points.

        >>> bss = BSpline(nbasis=3, order=3)
        >>> bss.evaluate([0, 0.5, 1])
        array([[ 1.  ,  0.25,  0.  ],
               [ 0.  ,  0.5 ,  0.  ],
               [ 0.  ,  0.25,  1.  ]])

    """
    def __init__(self, def_range=None, nbasis=None, order=4, knots=None):
        """BSpline basis constructor.

        Args:
            def_range (tuple, optional): Definition of the interval where the
                basis defines a space. Defaults to (0,1) if knots are not
                specified. If knots are specified defaults to the first and
                last element of the knots.
            nbasis (int, optional): Number of splines that form the basis.
            order (int, optional): Order of the splines. One greater that
                their degree. Defaults to 4 which mean cubic splines.
            knots (list): List of knots of the splines. If def_range is
                specified the first and last elements of the knots have to
                match with it.
        """
        # Knots default to equally space points in the def_range
        if knots is None:
            if nbasis is None:
                raise ValueError("Must provide either a list of knots or the"
                                 "number of basis.")
            if def_range is None:
                def_range = (0, 1)
            knots = list(numpy.linspace(def_range[0], def_range[1],
                                        nbasis - order + 2))
        else:
            knots = list(knots)
            knots.sort()

        # nbasis default to number of knots + order of the splines - 2
        if nbasis is None:
            nbasis = len(knots) + order - 2
            if def_range is None:
                def_range = (knots[0], knots[-1])

        if def_range[0] != knots[0] or def_range[1] != knots[-1]:
            raise ValueError("The ends of the knots must be the same as "
                             "the def_range.")

        # Checks
        if nbasis != order + len(knots) - 2:
            raise ValueError("The number of basis has to equal the order "
                             "plus the number of knots minus 2.")

        self.order = order
        self.knots = knots
        super().__init__(def_range, nbasis)

    def plot(self, **kwargs):
        return super().plot(**kwargs)

    def evaluate(self, eval_points):
        return super().evaluate(eval_points)

    def compute_matrix(self, eval_points):
        """Computes the basis at a list of values.

        It uses the scipy implementation of BSplines to compute the values
        for each element of the basis.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis at the values specified in eval_points.

        """
        # Because of how bsplines are defined is necessary to duplicate the
        # values at the ends of the knots as many times as the order.
        knots = numpy.array([self.knots[0]] * (self.order - 1) + self.knots
                            + [self.knots[-1]] * (self.order - 1))
        # c is used the select which spline the function splev below computes
        c = numpy.zeros(len(knots))

        # Initialise empty matrix
        mat = numpy.empty((self.nbasis, len(eval_points)))

        # For each basis computes its value for each evaluation point
        for i in range(self.nbasis):
            # write a 1 in c in the position of the spline calculated in each
            # iteration
            c[i] = 1
            # compute the spline
            mat[i] = scipy.interpolate.splev(eval_points, (knots, c,
                                                           self.order - 1))
            c[i] = 0

        return mat


class FDataBasis:

    def __init__(self, basis, coefficients):
        """

        Args:
            basis:
            coefficients:
        """
        coefficients = numpy.atleast_2d(coefficients)
        if coefficients.shape[1] != basis.nbasis:
            raise ValueError("The length or number of columns of coefficients "
                             "has to be the same equal to the number of "
                             "elements of the basis.")
        self.basis = basis
        self.coefficients = coefficients

    @property
    def nsamples(self):
        return self.coefficients.shape[0]

    @property
    def nbasis(self):
        return self.basis.nbasis

    @property
    def def_range(self):
        return self.basis.def_range

    def evaluate(self, eval_points):

        # each column is the values of one element of the basis
        basis_values = self.basis.evaluate(eval_points).T

        res_matrix = numpy.empty((self.nsamples, len(eval_points)))

        for i in range(self.nsamples):
            _matrix = basis_values * self.coefficients[i]
            res_matrix[i] = _matrix.sum(axis=1)

        return res_matrix

    def plot(self, **kwargs):
        npoints = max(501, 10 * self.nbasis)
        # List of points where the basis are evaluated
        eval_points = numpy.linspace(self.def_range[0], self.def_range[1],
                                     npoints)
        # Basis evaluated in the previous list of points
        mat = self.evaluate(eval_points)
        # Plot
        return matplotlib.pyplot.plot(eval_points, mat.T, **kwargs)


def data_tof_data_basis(data_matrix, sample_points, basis, method='cholesky'):

    # n is the samples
    # m is the observations
    # k is the number of elements of the basis

    # Each sample in a column (m x n)
    data_matrix = data_matrix.T

    # Each basis in a column
    basis_values = basis.evaluate(sample_points).T

    # We need to solve the equation
    # basis_values(B) @ C = data_matrix(D)

    if method == 'cholesky':
        # Resolves the equation
        # B.T @ B @ C = B.T @ D
        # by means of the cholesky decomposition
        left_matrix = basis_values.T @ basis_values
        right_matrix = basis_values.T @ data_matrix
        coefficients = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
            left_matrix, lower=True), right_matrix)

        # The ith column is the coefficients of the ith basis for each sample
        coefficients = coefficients.T

    elif method == 'qr':
        # Resolves the equation
        # B.T @ B @ C = B.T @ D
        # by means of the QR decomposition

        # B = Q @ R
        q, r = scipy.linalg.qr(basis_values)
        right_matrix = q.T @ data_matrix

        # R @ C = Q.T @ D
        coefficients = numpy.linalg.solve(r, right_matrix)
        # The ith column is the coefficients of the ith basis for each sample
        coefficients = coefficients.T

    else:
        raise ValueError("Unknown method.")

    return FDataBasis(basis, coefficients)

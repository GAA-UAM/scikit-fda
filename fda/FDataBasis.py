"""This module defines functional data object in a basis function system
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
    """Defines the structure of a basis function system.

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): number of functions in the basis.

    """

    def __init__(self, domain_range=(0, 1), nbasis=1):
        """Basis constructor.

        Args:
            domain_range (tuple, optional): Definition of the interval where
                the basis defines a space. Defaults to (0,1).
            nbasis: Number of functions that form the basis. Defaults to 1.
        """
        # Some checks
        if domain_range[0] >= domain_range[1]:
            raise ValueError("The interval {} is not well-defined.".format(
                domain_range))
        if nbasis < 1:
            raise ValueError("The number of basis has to be strictly "
                             "possitive.")
        self.domain_range = domain_range
        self.nbasis = nbasis
        self._drop_index_lst = []

        super().__init__()

    @abstractmethod
    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        pass

    def evaluate(self, eval_points, derivative=0):
        """Evaluates the basis function system or its derivatives at a list of
        given values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        eval_points = numpy.asarray(eval_points)
        if numpy.any(numpy.isnan(eval_points)):
            raise ValueError("The list of points where the function is "
                             "evaluated can not contain nan values.")

        return self._compute_matrix(eval_points, derivative)

    def plot(self, ax=None, derivative=0, **kwargs):
        """Plots the basis object or its derivatives.

        Args:
            ax (axis object, optional): axis over with the graphs are plotted.
                Defaults to matplotlib current axis.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            **kwargs: keyword arguments to be [RS05]_passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """
        if ax is None:
            ax = matplotlib.pyplot.gca()
        # Number of points where the basis are evaluated
        npoints = max(501, 10 * self.nbasis)
        # List of points where the basis are evaluated
        eval_points = numpy.linspace(self.domain_range[0],
                                     self.domain_range[1],
                                     npoints)
        # Basis evaluated in the previous list of points
        mat = self.evaluate(eval_points, derivative)
        # Plot
        return ax.plot(eval_points, mat.T, **kwargs)

    def __repr__(self):
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"nbasis={self.nbasis})")


class Monomial(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): number of functions in the basis.

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

        And also evaluates its derivatives

        >>> bs_mon.evaluate([0, 1, 2], derivative=1)
        array([[ 0.,  0.,  0.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  4.]])
        >>> bs_mon.evaluate([0, 1, 2], derivative=2)
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 2.,  2.,  2.]])

    """
    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        For each of the basis computes its value for each of the points in
        the list passed as argument to the method.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        # Initialise empty matrix
        mat = numpy.zeros((self.nbasis, len(eval_points)))
        
        # For each basis computes its value for each evaluation
        if derivative == 0:
            for i in range(self.nbasis):
                mat[i] = eval_points ** i
        else:
            for i in range(self.nbasis):
                if derivative <= i:
                    factor = i
                    for j in range(2, derivative + 1):
                        factor *= (i - j + 1)
                    mat[i] = factor * eval_points ** (i - derivative)

        return mat


class BSpline(Basis):
    r"""BSpline basis.

    BSpline basis elements are defined recursively as:

    .. math::
        B_{i, 1}(x) = 1 \quad \text{if } t_i \le x < t_{i+1},
        \quad 0 \text{ otherwise}

    .. math::
        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
        + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    Where k indicates the order of the spline.

    Implementation details: In order to allow a discontinuous behaviour at
    the boundaries of the domain it is necessary to placing m knots at the
    boundaries [RS05]_. This is automatically done so that the user only has to
    specify a single knot at the boundaries.

    Attributes:
        domain_range (tuple): A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): Number of functions in the basis.
        order (int): Order of the splines. One greather than their degree.
        knots (list): List of knots of the spline functions.

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

        And evaluates first derivative

        >>> bss.evaluate([0, 0.5, 1], derivative=1)
        array([[-2., -1.,  0.],
               [ 2.,  0., -2.],
               [ 0.,  1.,  2.]])

    References:
        .. [RS05] Ramsay, J., Silverman, B. W. (2005). *Functional Data
            Analysis*. Springler. 50-51.

    """
    def __init__(self, domain_range=None, nbasis=None, order=4, knots=None):
        """BSpline basis constructor.

        Args:
            domain_range (tuple, optional): Definition of the interval where
                the basis defines a space. Defaults to (0,1) if knots are not
                specified. If knots are specified defaults to the first and
                last element of the knots.
            nbasis (int, optional): Number of splines that form the basis.
            order (int, optional): Order of the splines. One greater that
                their degree. Defaults to 4 which mean cubic splines.
            knots (array_like): List of knots of the splines. If domain_range
                is specified the first and last elements of the knots have to
                match with it.
        """
        # Knots default to equally space points in the domain_range
        if knots is None:
            if nbasis is None:
                raise ValueError("Must provide either a list of knots or the"
                                 "number of basis.")
            if domain_range is None:
                domain_range = (0, 1)
            knots = list(numpy.linspace(domain_range[0], domain_range[1],
                                        nbasis - order + 2))
        else:
            knots = list(knots)
            knots.sort()

        # nbasis default to number of knots + order of the splines - 2
        if nbasis is None:
            nbasis = len(knots) + order - 2
            if domain_range is None:
                domain_range = (knots[0], knots[-1])

        if domain_range[0] != knots[0] or domain_range[1] != knots[-1]:
            raise ValueError("The ends of the knots must be the same as "
                             "the domain_range.")

        # Checks
        if nbasis != order + len(knots) - 2:
            raise ValueError("The number of basis has to equal the order "
                             "plus the number of knots minus 2.")

        self.order = order
        self.knots = knots
        super().__init__(domain_range, nbasis)

    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        It uses the scipy implementation of BSplines to compute the values
        for each element of the basis.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        Implementation details: In order to allow a discontinuous behaviour at
        the boundaries of the domain it is necessary to placing m knots at the
        boundaries [RS05]_. This is automatically done so that the user only
        has to specify a single knot at the boundaries.

        References:
            .. [RS05] Ramsay, J., Silverman, B. W. (2005). *Functional Data
                Analysis*. Springler. 50-51.

        """
        # Places m knots at the boundaries
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
                                                           self.order - 1),
                                             der=derivative)
            c[i] = 0

        return mat

    def __repr__(self):
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"nbasis={self.nbasis}, "
                f"order={self.order}, "
                f"knots={self.knots})")


class Fourier(Basis):
    r"""Fourier basis.

    Defines a functional basis for representing functions on a fourier
    series expansion of period :math:`T`.

    .. math::
        \phi_0(t) = \frac{1}{\sqrt{2}}

    .. math::
        \phi_{2n -1}(t) = sin(\frac{2 \pi n}{T} t)

    .. math::
        \phi_{2n}(t) = cos(\frac{2 \pi n}{T} t)


    Actually this basis functions are not orthogonal but not orthonormal. To
    achive this they are divided by its norm: :math:`\sqrt\left(\frac{T}{2}
    \right)`.

    Attributes:
        domain_range (tuple): A tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        nbasis (int): Number of functions in the basis.
        period (int or float): Period (:math:`T`).

    Examples:
        Constructs specifying number of basis, definition interval and period.

        >>> fb = Fourier([0, numpy.pi], nbasis=3, period=1)
        >>> fb.evaluate([0, numpy.pi / 4, numpy.pi / 2, numpy.pi]).round(2)
        array([[ 1.  ,  1.  ,  1.  ,  1.  ],
               [ 1.41,  0.31, -1.28,  0.89],
               [ 0.  , -1.38, -0.61,  1.1 ]])

        And evaluate second derivative

        >>> fb.evaluate([0, numpy.pi / 4, numpy.pi / 2, numpy.pi],
        ...             derivative = 2).round(2)
        array([[  0.  ,   0.  ,   0.  ,   0.  ],
               [-55.83, -12.32,  50.4 , -35.16],
               [ -0.  ,  54.46,  24.02, -43.37]])



    """
    def __init__(self, domain_range=(0, 1), nbasis=3, period=1):

        self.period = period
        super().__init__(domain_range, nbasis)

    def _compute_matrix(self, eval_points, derivative=0):
        """Computes the basis or its derivatives given a list of values.

        Args:
            eval_points (array_like): List of points where the basis is
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (:obj:`numpy.darray`): Matrix whose rows are the values of the each
            basis function or its derivatives at the values specified in
            eval_points.

        """
        if derivative < 0:
            raise ValueError("derivative only takes non-negative values.")

        omega = 2 * numpy.pi / self.period
        omega_t = omega * eval_points
        nbasis = self.nbasis if self.nbasis % 2 != 0 else self.nbasis + 1

        # Initialise empty matrix
        mat = numpy.empty((self.nbasis, len(eval_points)))

        if derivative == 0:
            # First base function is a constant
            # The division by numpy.sqrt(2) is so that it has the same norm as the
            # sine and cosine: sqrt(period / 2)
            mat[0] = numpy.ones(len(eval_points)) / numpy.sqrt(2)
            if nbasis > 1:
                # 2*pi*n*x / period
                args = numpy.outer(range(1, nbasis // 2 + 1), omega_t)
                index = range(2, nbasis, 2)
                # even indexes are sine functions
                mat[index] = numpy.sin(args)
                index = range(1 , nbasis - 1, 2)
                # odd indexes are cosine functions
                mat[index] = numpy.cos(args)
        # evaluates the derivatives
        else:
            # First base function is a constant, so its derivative is 0.
            mat[0] = numpy.zeros(len(eval_points))
            if nbasis > 1:
                # (2*pi*n / period) ^ n_derivative
                factor = numpy.outer(
                    (-1) ** (derivative // 2)
                    * (numpy.array(range(1, nbasis // 2 + 1)) * omega)
                    ** derivative,
                    numpy.ones(len(eval_points)))
                # 2*pi*n*x / period
                args = numpy.outer(range(1, nbasis // 2 + 1), omega_t)
                # even indexes
                index_e = range(2, nbasis, 2)
                # odd indexes
                index_o = range(1, nbasis - 1, 2)
                if derivative % 2 == 0:
                    mat[index_e] = factor * numpy.sin(args)
                    mat[index_o] = factor * numpy.cos(args)
                else:
                    mat[index_e] = factor * numpy.cos(args)
                    mat[index_o] = -factor * numpy.sin(args)

        # normalise
        mat = mat / numpy.sqrt(self.period / 2)
        return mat

    def __repr__(self):
        return (f"{self.__class__.__name__}(domain_range={self.domain_range}, "
                f"nbasis={self.nbasis}, "
                f"period={self.period})")

class FDataBasis:
    r"""Basis representation of functional data.

    Class representation for functional data in the form of a set of basis
    functions multplied by a set of coefficients.

    .. math::
        f(x) = \sum_{k=1}{K}c_k\phi_k

    Where n is the number of basis functions, :math:`c = (c_1, c_2, ...,
    c_K)` the vector of coefficients and  :math:`\phi = (\phi_1, \phi_2,
    ..., \phi_K)` the basis function system.

    Attributes:
        basis (:obj:`Basis`): Basis function system.
        coefficients (numpy.darray): List or matrix of coefficients. Has to
            have the same length or number of columns as the number of basis
            function in the basis. If a matrix, each row contains the
            coefficients that multiplied by the basis functions produce each
            functional datum.

    Examples:
        >>> basis = Monomial(nbasis=4)
        >>> coefficients = [1, 1, 3, .5]
        >>> FDataBasis(basis, coefficients)
        FDataBasis(basis=Monomial(...), coefficients=[[ 1.   1.   3.   0.5]])

    """

    def __init__(self, basis, coefficients):
        """Constructor of FDataBasis.

        Args:
            basis (:obj:`Basis`): Basis function system.
            coefficients (array_like): List or matrix of coefficients. Has to
                have the same length or number of columns as the number of
                basis function in the basis.
        """
        coefficients = numpy.atleast_2d(coefficients)
        if coefficients.shape[1] != basis.nbasis:
            raise ValueError("The length or number of columns of coefficients "
                             "has to be the same equal to the number of "
                             "elements of the basis.")
        self.basis = basis
        self.coefficients = coefficients

    @staticmethod
    def from_data(data_matrix, sample_points, basis, method='cholesky'):
        """Raw data to functional form.

        Takes functional data in a discrete form and makes an approximates it
        to the closest function that can be generated by the basis.

        Args:
            data_matrix (array_like): List of matrix containing the
                observations. If a matrix each row represents a single
                functional datum and the columns the different observations.
            sample_points (array_like): Values of the domain where the previous
                data were taken.
            basis: (:obj:`Basis`): Basis used.
            method (str): Algorithm used for calculating the coefficients using
                the least squares method. The values admitted are 'cholesky'
                and 'qr' for Cholesky and QR factorisation methods
                respectively.

        Returns:
            :obj:`FDataBasis`: Represention of the data in a functional form as
                product of coefficients by basis functions.

        Examples:
            >>> import numpy as np
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
            >>> x
            array([ 1.,  1., -1., -1.,  1.])

            >>> basis = Fourier((0, 1), nbasis=3)
            >>> fd = FDataBasis.from_data(x, t, basis)
            >>> fd.coefficients.round(2)
            array([[ 0.  ,  0.71,  0.71]])

        """

        # n is the samples
        # m is the observations
        # k is the number of elements of the basis

        # Each sample in a column (m x n)
        data_matrix = numpy.atleast_2d(data_matrix).T

        # Each basis in a column
        basis_values = basis.evaluate(sample_points).T

        # We need to solve the equation
        # basis_values(B) @ C = data_matrix(D)

        method = method.lower()
        if method == 'cholesky':
            # Resolves the equation
            # B.T @ B @ C = B.T @ D
            # by means of the cholesky decomposition
            left_matrix = basis_values.T @ basis_values
            right_matrix = basis_values.T @ data_matrix
            coefficients = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                left_matrix, lower=True), right_matrix)

            # The ith column is the coefficients of the ith basis for each
            #  sample
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
            # The ith column is the coefficients of the ith basis for each
            # sample
            coefficients = coefficients.T

        else:
            raise ValueError("Unknown method.")

        return FDataBasis(basis, coefficients)

    @property
    def nsamples(self):
        """Number of samples"""
        return self.coefficients.shape[0]

    @property
    def nbasis(self):
        """Number of basis"""
        return self.basis.nbasis

    @property
    def domain_range(self):
        """Definition range"""
        return self.basis.domain_range

    def evaluate(self, eval_points, derivative=0):
        """Evaluates the object or its derivatives at a list of values.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        # each column is the values of one element of the basis
        basis_values = self.basis.evaluate(eval_points, derivative).T

        res_matrix = numpy.empty((self.nsamples, len(eval_points)))

        for i in range(self.nsamples):
            _matrix = basis_values * self.coefficients[i]
            res_matrix[i] = _matrix.sum(axis=1)

        return res_matrix

    def plot(self, derivative=0, **kwargs):
        """Plots the FDataBasis object or its derivatives.

        Args:
            derivative (int, optional): Order of the derivative. Defaults to 0.
            **kwargs: keyword arguments to be passed to the
                matplotlib.pyplot.plot function.

        Returns:
            List of lines that were added to the plot.

        """
        npoints = max(501, 10 * self.nbasis)
        # List of points where the basis are evaluated
        eval_points = numpy.linspace(self.domain_range[0],
                                     self.domain_range[1],
                                     npoints)
        # Basis evaluated in the previous list of points
        mat = self.evaluate(eval_points, derivative)
        # Plot
        return matplotlib.pyplot.plot(eval_points, mat.T, **kwargs)

    def mean(self):
        """ Computes the mean of all the samples in a FDataBasis object.

        Returns:
            :obj:`FDataBasis`: A FDataBais object with just one sample
            representing the mean of all the samples in the original
            FDataBasis object.

        Examples:
            >>> basis = Monomial(nbasis=4)
            >>> coefficients = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
            >>> FDataBasis(basis, coefficients).mean()
            FDataBasis(basis=..., coefficients=[[ 1.   1.   3.   0.5]])

        """
        return FDataBasis(self.basis, numpy.mean(self.coefficients, axis=0))

    def __repr__(self):
        return (f"FDataBasis(basis={self.basis}, coefficients="
                f"{self.coefficients})")

    def __call__(self, eval_points):
        """Evaluates the functions in the object at a list of values.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated.

        Returns:
            (numpy.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        return self.evaluate(eval_points)

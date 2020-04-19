import numpy as np
from ..._utils import _same_domain
from ._basis import Basis


class Monomial(Basis):
    """Monomial basis.

    Basis formed by powers of the argument :math:`t`:

    .. math::
        1, t, t^2, t^3...

    Attributes:
        domain_range (tuple): a tuple of length 2 containing the initial and
            end values of the interval over which the basis can be evaluated.
        n_basis (int): number of functions in the basis.

    Examples:
        Defines a monomial base over the interval :math:`[0, 5]` consisting
        on the first 3 powers of :math:`t`: :math:`1, t, t^2`.

        >>> bs_mon = Monomial((0,5), n_basis=3)

        And evaluates all the functions in the basis in a list of descrete
        values.

        >>> bs_mon.evaluate([0, 1, 2])
        array([[1, 1, 1],
               [0, 1, 2],
               [0, 1, 4]])

        And also evaluates its derivatives

        >>> bs_mon.evaluate([0, 1, 2], derivative=1)
        array([[0, 0, 0],
               [1, 1, 1],
               [0, 2, 4]])
        >>> bs_mon.evaluate([0, 1, 2], derivative=2)
        array([[0, 0, 0],
               [0, 0, 0],
               [2, 2, 2]])

    """

    def _coefs_exps_derivatives(self, derivative):
        """
        Return coefficients and exponents of the derivatives.

        This function is used for computing the basis functions and evaluate.

        When the exponent would be negative (the coefficient in that case
        is zero) returns 0 as the exponent (to prevent division by zero).
        """
        seq = np.arange(self.n_basis)
        coef_mat = np.linspace(seq, seq - derivative + 1,
                               derivative, dtype=int)
        coefs = np.prod(coef_mat, axis=0)

        exps = np.maximum(seq - derivative, 0)

        return coefs, exps

    def _evaluate(self, eval_points, derivative=0):

        coefs, exps = self._coefs_exps_derivatives(derivative)

        raised = np.power.outer(eval_points, exps)

        return (coefs * raised).T

    def _derivative(self, coefs, order=1):
        return (Monomial(self.domain_range, self.n_basis - order),
                np.array([np.polyder(x[::-1], order)[::-1]
                          for x in coefs]))

    def _evaluate_constant_lfd(self, weights):
        """
        Evaluate constant weights of a linear differential operator
        over the basis functions.
        """

        max_derivative = len(weights) - 1

        _, coef_mat = self._coef_mat(max_derivative)

        # Compute coefficients for each derivative
        coefs = np.cumprod(coef_mat, axis=0)

        # Add derivative 0 row
        coefs = np.concatenate((np.ones((1, self.n_basis)), coefs))

        # Now each row correspond to each basis and each column to
        # each derivative
        coefs_t = coefs.T

        # Multiply by the weights
        weighted_coefs = coefs_t * weights
        assert len(weighted_coefs) == self.n_basis

        # Now each row has the right weight, but the polynomials are in a
        # decreasing order and with different exponents

        # Resize the coefs so that there are as many rows as the number of
        # basis
        # The matrix is now triangular
        # refcheck is False to prevent exceptions while debugging
        weighted_coefs = np.copy(weighted_coefs.T)
        weighted_coefs.resize(self.n_basis,
                              self.n_basis, refcheck=False)
        weighted_coefs = weighted_coefs.T

        # Shift the coefficients so that they correspond to the right
        # exponent
        indexes = np.tril_indices(self.n_basis)
        polynomials = np.zeros_like(weighted_coefs)
        polynomials[indexes[0], indexes[1] -
                    indexes[0] - 1] = weighted_coefs[indexes]

        # At this point, each row of the matrix correspond to a polynomial
        # that is the result of applying the linear differential operator
        # to each element of the basis

        return polynomials

    def _penalty(self, lfd):

        weights = lfd.constant_weights()
        if weights is None:
            return NotImplemented

        polynomials = self._evaluate_constant_lfd(weights)

        # Expand the polinomials with 0, so that the multiplication fits
        # inside. It will need the double of the degree
        length_with_padding = polynomials.shape[1] * 2 - 1

        # Multiplication of polynomials is a convolution.
        # The convolution can be performed in parallel applying a Fourier
        # transform and then doing a normal multiplication in that
        # space, coverting back with the inverse Fourier transform
        fft = np.fft.rfft(polynomials, length_with_padding)

        # We compute only the upper matrix, as the penalty matrix is
        # symmetrical
        indices = np.triu_indices(self.n_basis)
        fft_mul = fft[indices[0]] * fft[indices[1]]

        integrand = np.fft.irfft(fft_mul, length_with_padding)

        integration_domain = self.domain_range[0]

        # To integrate, divide by the position and increase the exponent
        # in the evaluation
        denom = np.arange(integrand.shape[1], 0, -1)
        integrand /= denom

        # Add column of zeros at the right to increase exponent
        integrand = np.pad(integrand,
                           pad_width=((0, 0),
                                      (0, 1)),
                           mode='constant')

        # Now, apply Barrow's rule
        # polyval applies Horner method over the first dimension,
        # so we need to transpose
        x_right = np.polyval(integrand.T, integration_domain[1])
        x_left = np.polyval(integrand.T, integration_domain[0])

        integral = x_right - x_left

        penalty_matrix = np.empty((self.n_basis, self.n_basis))

        # Set upper matrix
        penalty_matrix[indices] = integral

        # Set lower matrix
        penalty_matrix[(indices[1], indices[0])] = integral

        return penalty_matrix

    def basis_of_product(self, other):
        """Multiplication of a Monomial Basis with other Basis"""
        if not _same_domain(self, other):
            raise ValueError("Ranges are not equal.")

        if isinstance(other, Monomial):
            return Monomial(self.domain_range, self.n_basis + other.n_basis)

        return other.rbasis_of_product(self)

    def rbasis_of_product(self, other):
        """Multiplication of a Monomial Basis with other Basis"""
        return Basis.default_basis_of_product(self, other)

    def _to_R(self):
        drange = self.domain_range[0]
        return "create.monomial.basis(rangeval = c(" + str(drange[0]) + "," +\
               str(drange[1]) + "), nbasis = " + str(self.n_basis) + ")"

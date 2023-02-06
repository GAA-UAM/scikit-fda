from __future__ import annotations

import numbers
from typing import Callable, Sequence, Union

import numpy as np
import scipy.integrate
from numpy import polyder, polyint, polymul, polyval
from scipy.interpolate import PPoly

from ...representation import FData, FDataGrid
from ...representation.basis import (
    Basis,
    BSplineBasis,
    ConstantBasis,
    FourierBasis,
    MonomialBasis,
)
from ...typing._base import DomainRangeLike
from ...typing._numpy import NDArrayFloat
from ._operators import Operator, gram_matrix_optimization

Order = int

WeightSequence = Sequence[
    Union[
        float,
        Callable[[NDArrayFloat], NDArrayFloat],
    ],
]


class LinearDifferentialOperator(
    Operator[Union[FData, Basis], Callable[[NDArrayFloat], NDArrayFloat]],
):
    r"""
    Defines the structure of a linear differential operator function system.

    .. math::
        Lx(t) = b_0(t) x(t) + b_1(t) x'(t) +
                \dots + b_{n-1}(t) D^{n-1}(x(t)) + b_n(t) D^n(x(t))

    Can only be applied to functional data, as multivariate data has no
    derivatives.

    You have to provide either order or weights.
    If both are provided, it will raise an error.
    If a positional argument is supplied it will be considered the
    order if it is an integral type and the weights otherwise.

    Parameters:
        order: the order of the operator. It's the highest derivative order
            of the operator.
        weights: A FDataBasis objects list of length order + 1 items.
        domain_range: Definition of the interval where the weight functions
            are defined. If the functional weights are specified and this is
            not, takes the domain range from them. Otherwise, defaults to
            (0,1).

    Attributes:
        weights (list):  A list of callables.

    Examples:
        Create a linear differential operator that penalizes the second
        derivative (acceleration)

        >>> from skfda.misc.operators import LinearDifferentialOperator
        >>> from skfda.representation.basis import (FDataBasis,
        ...                                         MonomialBasis,
        ...                                         ConstantBasis)
        >>>
        >>> LinearDifferentialOperator(2)
        LinearDifferentialOperator(
            weights=(0, 0, 1),
        )

        Create a linear differential operator that penalizes three times
        the second derivative (acceleration) and twice the first (velocity).

        >>> LinearDifferentialOperator(weights=[0, 2, 3])
        LinearDifferentialOperator(
            weights=(0, 2, 3),
        )

        Create a linear differential operator with non-constant weights.

        >>> constant = ConstantBasis()
        >>> monomial = MonomialBasis(domain_range=(0, 1), n_basis=3)
        >>> fdlist = [FDataBasis(constant, [0.]),
        ...           FDataBasis(constant, [0.]),
        ...           FDataBasis(monomial, [1., 2., 3.])]
        >>> LinearDifferentialOperator(weights=fdlist)
        LinearDifferentialOperator(
        weights=(FDataBasis(
                basis=ConstantBasis(domain_range=((0.0, 1.0),), n_basis=1),
                coefficients=[[ 0.]],
                ...),
            FDataBasis(
                basis=ConstantBasis(domain_range=((0.0, 1.0),), n_basis=1),
                coefficients=[[ 0.]],
                ...),
            FDataBasis(
                basis=MonomialBasis(domain_range=((0.0, 1.0),), n_basis=3),
                coefficients=[[ 1. 2. 3.]],
                ...)),
        )

    """

    def __init__(
        self,
        order_or_weights: Order | WeightSequence | None = None,
        *,
        order: int | None = None,
        weights: WeightSequence | None = None,
        domain_range: DomainRangeLike | None = None,
    ) -> None:

        num_args = sum(
            a is not None for a in (order_or_weights, order, weights)
        )

        if num_args > 1:
            raise ValueError(
                "You have to provide the order or the weights, not both.",
            )

        if order_or_weights is not None:
            if isinstance(order_or_weights, numbers.Integral):
                order = int(order_or_weights)
            else:
                assert isinstance(order_or_weights, Sequence)
                weights = order_or_weights

        if order is None and weights is None:
            weights = (0,)

        elif order is not None:
            if order < 0:
                raise ValueError("Order should be an non-negative integer.")

            weights = tuple(
                0 if (i < order) else 1
                for i in range(order + 1)
            )

        assert weights is not None
        if len(weights) == 0:
            raise ValueError("You have to provide one weight at least.")

        # Check domain ranges
        for w in weights:
            w_domain_range = getattr(w, "domain_range", None)

            if w_domain_range is not None:
                if domain_range is None:
                    domain_range = w_domain_range
                elif not np.array_equal(w_domain_range, domain_range):
                    raise ValueError(
                        "Weights with wrong domain range.",
                    )
        self.weights = tuple(weights)

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}(\n"
            f"\tweights={self.weights},\n"
            f")"
        ).replace('\n', '\n    ')

    def __eq__(self, other: object) -> bool:

        return (
            isinstance(other, LinearDifferentialOperator)
            and self.weights == other.weights
        )

    def constant_weights(self) -> NDArrayFloat | None:
        """
        Return constant weights.

        Return the scalar weights of the linear differential operator if they
        are constant basis.
        Otherwise, return None.

        This function is mostly useful for basis which want to override
        the _penalty method in order to use an analytical expression
        for constant weights.

        """
        weights = np.array(self.weights)

        return None if weights.dtype == np.object_ else weights

    def __call__(
        self,
        f: FData | Basis,
    ) -> Callable[[NDArrayFloat], NDArrayFloat]:
        """Return the function that results of applying the operator."""
        function_derivatives = [
            f.derivative(order=i) for i, _ in enumerate(self.weights)
        ]

        def applied_linear_diff_op(
            t: NDArrayFloat,
        ) -> NDArrayFloat:
            result = sum(
                (w(t) if callable(w) else np.asarray(w))
                * function_derivatives[i](t)
                for i, w in enumerate(self.weights)
            )
            assert not isinstance(result, int)
            return result

        return applied_linear_diff_op


#############################################################
#
# Optimized implementations of gram matrix for each basis.
#
#############################################################


@gram_matrix_optimization.register
def constant_penalty_matrix_optimized(
    linear_operator: LinearDifferentialOperator,
    basis: ConstantBasis,
) -> NDArrayFloat:
    """Optimized version for Constant basis."""
    coefs = linear_operator.constant_weights()
    if coefs is None:
        return NotImplemented

    return np.array([[
        coefs[0] ** 2
        * (basis.domain_range[0][1] - basis.domain_range[0][0]),
    ]])


def _monomial_evaluate_constant_linear_diff_op(
    basis: MonomialBasis,
    weights: NDArrayFloat,
) -> NDArrayFloat:
    """Evaluate constant weights over the monomial basis."""
    max_derivative = len(weights) - 1

    seq = np.arange(basis.n_basis)
    coef_mat = np.linspace(
        seq,
        seq - max_derivative + 1,
        max_derivative,
        dtype=int,
    )

    # Compute coefficients for each derivative
    coefs = np.cumprod(coef_mat, axis=0)

    # Add derivative 0 row
    coefs = np.concatenate((np.ones((1, basis.n_basis)), coefs))

    # Now each row correspond to each basis and each column to
    # each derivative
    coefs_t = coefs.T

    # Multiply by the weights
    weighted_coefs = coefs_t * weights
    assert len(weighted_coefs) == basis.n_basis

    # Now each row has the right weight, but the polynomials are in a
    # decreasing order and with different exponents

    # Resize the coefs so that there are as many rows as the number of
    # basis
    # The matrix is now triangular
    # refcheck is False to prevent exceptions while debugging
    weighted_coefs = np.copy(weighted_coefs.T)
    weighted_coefs.resize(
        basis.n_basis,
        basis.n_basis,
        refcheck=False,
    )
    weighted_coefs = weighted_coefs.T

    # Shift the coefficients so that they correspond to the right
    # exponent
    indexes = np.tril_indices(basis.n_basis)
    polynomials = np.zeros_like(weighted_coefs)
    polynomials[
        indexes[0],
        indexes[1] - indexes[0] - 1,
    ] = weighted_coefs[indexes]

    # At this point, each row of the matrix correspond to a polynomial
    # that is the result of applying the linear differential operator
    # to each element of the basis

    return polynomials  # type: ignore[no-any-return]


@gram_matrix_optimization.register
def monomial_penalty_matrix_optimized(
    linear_operator: LinearDifferentialOperator,
    basis: MonomialBasis,
) -> NDArrayFloat:
    """Optimized version for Monomial basis."""
    weights = linear_operator.constant_weights()
    if weights is None:
        return NotImplemented

    polynomials = _monomial_evaluate_constant_linear_diff_op(basis, weights)

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
    indices = np.triu_indices(basis.n_basis)
    fft_mul = fft[indices[0]] * fft[indices[1]]

    integrand = np.fft.irfft(fft_mul, length_with_padding)

    integration_domain = basis.domain_range[0]

    # To integrate, divide by the position and increase the exponent
    # in the evaluation
    denom = np.arange(integrand.shape[1], 0, -1)
    integrand /= denom

    # Add column of zeros at the right to increase exponent
    integrand = np.pad(
        integrand,
        pad_width=(
            (0, 0),
            (0, 1),
        ),
        mode='constant',
    )

    # Now, apply Barrow's rule
    # polyval applies Horner method over the first dimension,
    # so we need to transpose
    x_right = np.polyval(integrand.T, integration_domain[1])
    x_left = np.polyval(integrand.T, integration_domain[0])

    integral = x_right - x_left

    penalty_matrix = np.empty((basis.n_basis, basis.n_basis))

    # Set upper matrix
    penalty_matrix[indices] = integral

    # Set lower matrix
    penalty_matrix[(indices[1], indices[0])] = integral

    return penalty_matrix


def _fourier_penalty_matrix_optimized_orthonormal(
    basis: FourierBasis,
    weights: NDArrayFloat,
) -> NDArrayFloat:
    """Return the penalty when the basis is orthonormal."""
    signs = np.array([1, 1, -1, -1])
    signs_expanded = np.tile(signs, len(weights) // 4 + 1)

    signs_odd = signs_expanded[:len(weights)]
    signs_even = signs_expanded[1:len(weights) + 1]

    phases = (
        np.arange(1, (basis.n_basis - 1) // 2 + 1)
        * 2 * np.pi / basis.period
    )

    # Compute increasing powers
    coefs_no_sign = np.vander(phases, len(weights), increasing=True)

    coefs_no_sign *= weights

    coefs_odd = signs_odd * coefs_no_sign
    coefs_even = signs_even * coefs_no_sign

    # After applying the linear differential operator to a sinusoidal
    # element of the basis e, the result can be expressed as
    # A e + B e*, where e* is the other basis element in the pair
    # with the same phase

    odd_sin_coefs = np.sum(coefs_odd[:, ::2], axis=1)
    odd_cos_coefs = np.sum(coefs_odd[:, 1::2], axis=1)

    even_cos_coefs = np.sum(coefs_even[:, ::2], axis=1)
    even_sin_coefs = np.sum(coefs_even[:, 1::2], axis=1)

    # The diagonal is the inner product of A e + B e*
    # with itself. As the basis is orthonormal, the cross products e e*
    # are 0, and the products e e and e* e* are one.
    # Thus, the diagonal is A^2 + B^2
    # All elements outside the main diagonal are 0
    main_diag_odd = odd_sin_coefs**2 + odd_cos_coefs**2
    main_diag_even = even_sin_coefs**2 + even_cos_coefs**2

    # The main diagonal should intercalate both diagonals
    main_diag = np.array((main_diag_odd, main_diag_even)).T.ravel()

    penalty_matrix = np.diag(main_diag)

    # Add row and column for the constant
    penalty_matrix = np.pad(
        penalty_matrix,
        pad_width=(
            (1, 0),
            (1, 0),
        ),
        mode='constant',
    )

    penalty_matrix[0, 0] = weights[0]**2

    return penalty_matrix


@gram_matrix_optimization.register
def fourier_penalty_matrix_optimized(
    linear_operator: LinearDifferentialOperator,
    basis: FourierBasis,
) -> NDArrayFloat:
    """Optimized version for Fourier basis."""
    weights = linear_operator.constant_weights()
    if weights is None:
        return NotImplemented

    # If the period and domain range are not the same, the basis functions
    # are not orthogonal
    if basis.period != (basis.domain_range[0][1] - basis.domain_range[0][0]):
        return NotImplemented

    return _fourier_penalty_matrix_optimized_orthonormal(basis, weights)


@gram_matrix_optimization.register
def bspline_penalty_matrix_optimized(
    linear_operator: LinearDifferentialOperator,
    basis: BSplineBasis,
) -> NDArrayFloat:
    """Optimized version for BSpline basis."""
    coefs = linear_operator.constant_weights()
    if coefs is None:
        return NotImplemented

    nonzero = np.flatnonzero(coefs)

    # All derivatives above the order of the spline are effectively
    # zero
    nonzero = nonzero[nonzero < basis.order]

    if len(nonzero) == 0:
        return np.zeros((basis.n_basis, basis.n_basis))

    # We will only deal with one nonzero coefficient right now
    if len(nonzero) != 1:
        return NotImplemented

    derivative_degree = nonzero[0]

    if derivative_degree == basis.order - 1:
        # The derivative of the bsplines are constant in the intervals
        # defined between knots
        knots = np.array(basis.knots)
        mid_inter = (knots[1:] + knots[:-1]) / 2
        basis_deriv = basis.derivative(order=derivative_degree)
        constants = basis_deriv(mid_inter)[..., 0].T
        knots_intervals = np.diff(basis.knots)
        # Integration of product of constants
        return (  # type: ignore[no-any-return]
            constants.T @ np.diag(knots_intervals) @ constants
        )

    # We only deal with the case without zero length intervals
    # for now
    if np.any(np.diff(basis.knots) == 0):
        return NotImplemented

    # Compute exactly using the piecewise polynomial
    # representation of splines

    # Places m knots at the boundaries
    knots = np.array(basis._evaluation_knots())

    # c is used the select which spline the function
    # PPoly.from_spline below computes
    c = np.zeros(len(knots))

    # Initialise empty list to store the piecewise polynomials
    ppoly_lst = []

    no_0_intervals = np.where(np.diff(knots) > 0)[0]

    # For each basis gets its piecewise polynomial representation
    for i in range(basis.n_basis):

        # Write a 1 in c in the position of the spline
        # transformed in each iteration
        c[i] = 1

        # Gets the piecewise polynomial representation and gets
        # only the positions for no zero length intervals
        # This polynomial are defined relatively to the knots
        # meaning that the column i corresponds to the ith knot.
        # Let the ith knot be a
        # Then f(x) = pp(x - a)
        pp = PPoly.from_spline((knots, c, basis.order - 1))
        pp_coefs = pp.c[:, no_0_intervals]

        # We have the coefficients for each interval in coordinates
        # (x - a), so we will need to subtract a when computing the
        # definite integral
        ppoly_lst.append(pp_coefs)
        c[i] = 0

    # Now for each pair of basis computes the inner product after
    # applying the linear differential operator
    penalty_matrix = np.zeros((basis.n_basis, basis.n_basis))
    for interval, _ in enumerate(no_0_intervals):
        for i in range(basis.n_basis):
            poly_i = np.trim_zeros(ppoly_lst[i][:, interval], 'f')
            if len(poly_i) <= derivative_degree:
                # if the order of the polynomial is lesser or
                # equal to the derivative the result of the
                # integral will be 0
                continue
            # indefinite integral
            derivative = polyder(poly_i, derivative_degree)
            square = polymul(derivative, derivative)
            integral = polyint(square)

            # definite integral
            penalty_matrix[i, i] += np.diff(
                polyval(
                    integral,
                    np.asarray(basis.knots[interval: interval + 2])
                    - basis.knots[interval],
                ),
            )[0]

            for j in range(i + 1, basis.n_basis):
                poly_j = np.trim_zeros(ppoly_lst[j][:, interval], 'f')
                if len(poly_j) <= derivative_degree:
                    # if the order of the polynomial is lesser
                    # or equal to the derivative the result of
                    # the integral will be 0
                    continue
                    # indefinite integral
                integral = polyint(
                    polymul(
                        polyder(poly_i, derivative_degree),
                        polyder(poly_j, derivative_degree),
                    ),
                )
                # definite integral
                penalty_matrix[i, j] += np.diff(
                    polyval(
                        integral,
                        np.asarray(basis.knots[interval: interval + 2])
                        - basis.knots[interval],
                    ),
                )[0]
                penalty_matrix[j, i] = penalty_matrix[i, j]
    return penalty_matrix


@gram_matrix_optimization.register
def fdatagrid_penalty_matrix_optimized(
    linear_operator: LinearDifferentialOperator,
    basis: FDataGrid,
) -> NDArrayFloat:
    """Optimized version for FDatagrid."""
    evaluated_basis = sum(
        w(basis.grid_points[0]) if callable(w) else w
        * basis.derivative(order=i)(basis.grid_points[0])
        for i, w in enumerate(linear_operator.weights)
    )
    assert not isinstance(evaluated_basis, int)

    indices = np.triu_indices(basis.n_samples)
    product = evaluated_basis[indices[0]] * evaluated_basis[indices[1]]

    triang_vec = scipy.integrate.simps(product[..., 0], x=basis.grid_points)

    matrix = np.empty((basis.n_samples, basis.n_samples))

    # Set upper matrix
    matrix[indices] = triang_vec

    # Set lower matrix
    matrix[(indices[1], indices[0])] = triang_vec

    return matrix

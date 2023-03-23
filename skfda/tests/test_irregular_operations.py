"""Test the basic methods of the FDataIrregular structure"""
from ..typing._numpy import ArrayLike, Any
from typing import Tuple, Optional
import numpy as np
import pytest

from skfda.datasets._real_datasets import _fetch_loon_data
from skfda.representation import FDataIrregular, FDataGrid
from skfda.representation.interpolation import SplineInterpolation

############
# FIXTURES
############

NUM_CURVES = 10
MAX_VALUES_PER_CURVE = 99
DIMENSIONS = 2

random_state = np.random.RandomState(seed=14)


@pytest.fixture()
def input_arrays(
    num_curves: Optional[int] = NUM_CURVES,
    max_values_per_curve: Optional[int] = MAX_VALUES_PER_CURVE,
    dimensions: Optional[int] = 1
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Generate three unidimensional arrays describing a
    FDataIrregular structure with fixed sizes given by
    the parameters
    """
    num_values_per_curve = max_values_per_curve*np.ones(num_curves).astype(int)
    values_per_curve = [random_state.rand(num_values, dimensions)
                        for num_values in num_values_per_curve]
    args_per_curve = [random_state.rand(num_values, dimensions)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def input_arrays_2D(
    num_curves: Optional[int] = NUM_CURVES,
    max_values_per_curve: Optional[int] = MAX_VALUES_PER_CURVE,
    dimensions: Optional[int] = DIMENSIONS
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Generate three unidimensional arrays describing a
    FDataIrregular structure with fixed sizes given by
    the parameters
    """
    num_values_per_curve = max_values_per_curve*np.ones(num_curves).astype(int)
    values_per_curve = [random_state.rand(num_values, dimensions)
                        for num_values in num_values_per_curve]
    args_per_curve = [random_state.rand(num_values, dimensions)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def fdatairregular1D(
    input_arrays: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """
    Generate three multidimensional arrays
    describing a FDataIrregular structure
    """
    return FDataIrregular(*input_arrays)


@pytest.fixture()
def fdatairregular2D(
    input_arrays_2D: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """
    Generate three multidimensional arrays
    describing a FDataIrregular structure
    """
    return FDataIrregular(*input_arrays_2D)

############
# TESTS
############


@pytest.mark.parametrize(
    ("fdatairregular", "other"),
    [
        ("fdatairregular1D", 2),
        ("fdatairregular1D", 2*np.ones(NUM_CURVES)),
        ("fdatairregular1D", 2*np.ones((NUM_CURVES, 1))),
        ("fdatairregular1D", "fdatairregular1D"),
        ("fdatairregular2D", 2),
        ("fdatairregular2D", 2*np.ones(NUM_CURVES)),
        ("fdatairregular2D", 2*np.ones((NUM_CURVES, 2))),
        ("fdatairregular2D", "fdatairregular2D")
    ],
)

class TestArithmeticOperations:
    """Class which encapsulates the testing of basic arithmetic operations"""

    def _take_first(
        self,
        other,
    ) -> float:
        if isinstance(other, np.ndarray):
            return other[0]
        elif isinstance(other, FDataIrregular):
            return other.function_values
        return other

    def test_fdatairregular_arithmetic_sum(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular + other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = f_data_irreg + other

        assert np.all(
            f_data_sum.function_values ==
            f_data_irreg.function_values + self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rsum(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = other + f_data_irreg

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other) + f_data_irreg.function_values
            )

    def test_fdatairregular_arithmetic_sum_commutative(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        assert f_data_irreg + other == other + f_data_irreg

    def test_fdatairregular_arithmetic_sub(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular - other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = f_data_irreg - other

        assert np.all(
            f_data_sum.function_values ==
            f_data_irreg.function_values - self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rsub(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other - fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = other - f_data_irreg

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other) - f_data_irreg.function_values
            )

    def test_fdatairregular_arithmetic_mul(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular * other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_mul = f_data_irreg * other

        assert np.all(
            f_data_mul.function_values ==
            f_data_irreg.function_values * self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rmul(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_mul = other * f_data_irreg

        assert np.all(
            f_data_mul.function_values ==
            self._take_first(other) * f_data_irreg.function_values
            )

    def test_fdatairregular_arithmetic_mul_commutative(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        assert f_data_irreg * other == other * f_data_irreg

    def test_fdatairregular_arithmetic_div(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular / other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_div = f_data_irreg / other

        assert np.all(
            f_data_div.function_values ==
            f_data_irreg.function_values / self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rdiv(
        self,
        fdatairregular: str,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other / fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        f_data_irreg = request.getfixturevalue(fdatairregular)
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_div = other / f_data_irreg

        assert np.all(
            f_data_div.function_values ==
            self._take_first(other) / f_data_irreg.function_values
            )

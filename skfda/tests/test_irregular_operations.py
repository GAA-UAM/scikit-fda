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
    values_per_curve = [np.random.rand(num_values, dimensions)
                        for num_values in num_values_per_curve]
    args_per_curve = [np.random.rand(num_values, dimensions)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def input_arrays_multidimensional(
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Generate three multidimensional arrays
    describing a FDataIrregular structure
    """
    # TODO Make editable with pytest
    num_values_per_curve = np.random.randint(1,
                                             MAX_VALUES_PER_CURVE,
                                             size=(NUM_CURVES, )
                                             )

    values_per_curve = [np.random.rand(num_values, DIMENSIONS)
                        for num_values in num_values_per_curve]
    args_per_curve = [np.random.rand(num_values, DIMENSIONS)
                      for num_values in num_values_per_curve]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def fdatairregular(
    input_arrays: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """
    Generate three multidimensional arrays
    describing a FDataIrregular structure
    """
    return FDataIrregular(*input_arrays)

############
# TESTS
############


@pytest.mark.parametrize(
    ("other"),
    [
        (2),
        (2*np.ones(NUM_CURVES)),
        (2*np.ones((NUM_CURVES, 1))),
        ("fdatairregular")
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
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular + other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = fdatairregular + other

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular.function_values + self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rsum(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = other + fdatairregular

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other) + fdatairregular.function_values
            )

    def test_fdatairregular_arithmetic_sum_commutative(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        assert fdatairregular + other == other + fdatairregular

    def test_fdatairregular_arithmetic_sub(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular - other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = fdatairregular - other

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular.function_values - self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rsub(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other - fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = other - fdatairregular

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other) - fdatairregular.function_values
            )

    def test_fdatairregular_arithmetic_mul(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular * other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = fdatairregular * other

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular.function_values * self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rmul(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = other * fdatairregular

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other) * fdatairregular.function_values
            )

    def test_fdatairregular_arithmetic_mul_commutative(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        assert fdatairregular * other == other * fdatairregular

    def test_fdatairregular_arithmetic_div(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular / other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = fdatairregular / other

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular.function_values / self._take_first(other)
            )

    def test_fdatairregular_arithmetic_rdiv(
        self,
        fdatairregular: FDataIrregular,
        other: Any,
        request,
    ) -> None:
        """Tests the basic arithmetic operation other / fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        if isinstance(other, str):
            other = request.getfixturevalue(other)

        f_data_sum = other / fdatairregular

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other) / fdatairregular.function_values
            )

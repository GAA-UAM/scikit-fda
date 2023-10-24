"""Test the operations of the FDataIrregular structure."""
from typing import Optional, Tuple

import numpy as np
import pytest

from skfda.representation import FDataGrid, FDataIrregular
from skfda.representation.basis import (
    Basis,
    BSplineBasis,
    FDataBasis,
    FourierBasis,
    TensorBasis,
)

from ..typing._numpy import Any, ArrayLike

############
# MACROS
############
SEED = 2906198114

NUM_CURVES = 100
MAX_VALUES_PER_CURVE = 10
DIMENSIONS = 2
N_BASIS = 5
DECIMALS = 4

random_state = np.random.default_rng(seed=SEED)

############
# FIXTURES
############


@pytest.fixture()
def input_arrays(
    num_curves: Optional[int] = NUM_CURVES,
    max_values_per_curve: Optional[int] = MAX_VALUES_PER_CURVE,
    dimensions: Optional[int] = 1,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Create undiimensional arrays for FDataIrregular.

    Generate three unidimensional arrays describing a
    FDataIrregular structure with fixed sizes given by
    the parameters
    """
    num_values_per_curve = max_values_per_curve * np.ones(num_curves)
    num_values_per_curve = num_values_per_curve.astype(int)

    values_per_curve = [
        random_state.random((num_values, dimensions))
        for num_values in num_values_per_curve
    ]
    args_per_curve = [
        random_state.random((num_values, dimensions))
        for num_values in num_values_per_curve
    ]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def input_arrays_2d(
    num_curves: Optional[int] = NUM_CURVES,
    max_values_per_curve: Optional[int] = MAX_VALUES_PER_CURVE,
    dimensions: Optional[int] = DIMENSIONS,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Create multidimensional arrays for FDataIrregular.

    Generate three unidimensional arrays describing a
    FDataIrregular structure with fixed sizes given by
    the parameters
    """
    num_values_per_curve = max_values_per_curve * np.ones(num_curves)
    num_values_per_curve = num_values_per_curve.astype(int)

    values_per_curve = [
        random_state.random((num_values, dimensions))
        for num_values in num_values_per_curve
    ]
    args_per_curve = [
        random_state.random((num_values, dimensions))
        for num_values in num_values_per_curve
    ]

    indices = np.cumsum(num_values_per_curve) - num_values_per_curve
    values = np.concatenate(values_per_curve)
    arguments = np.concatenate(args_per_curve)

    return indices, values, arguments


@pytest.fixture()
def fdatagrid_1d(
) -> FDataGrid:
    """Generate FDataGrid."""
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.random((NUM_CURVES, num_values_per_curve, 1))
    # Grid points must be sorted
    grid_points = np.sort(random_state.random((num_values_per_curve,)))

    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )


@pytest.fixture()
def fdatagrid_2d(
) -> FDataGrid:
    """Generate multidimensional FDataGrid."""
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.random((
        NUM_CURVES,
        num_values_per_curve,
        DIMENSIONS,
    ))

    # Grid points must be sorted
    grid_points = np.sort(random_state.random((num_values_per_curve,)))

    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )


@pytest.fixture(
    params=[
        "fdatagrid_1d",
        "fdatagrid_2d",
    ],
)
def fdatagrid(
    request: Any,
    fdatagrid_1d: FDataGrid,
    fdatagrid_2d: FDataGrid,
) -> FDataIrregular:
    """Return 'fdatagrid_1d' or 'fdatagrid_2d'."""
    if request.param == "fdatagrid_1d":
        return fdatagrid_1d
    elif request.param == "fdatagrid_2d":
        return fdatagrid_2d


@pytest.fixture(params=["single_curve", "multiple_curves"])
def fdatairregular_1d(
    request: Any,
    input_arrays: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """Return FDataIrregular with only 1 curve or NUM_CURVES as requested."""
    indices, arguments, values = input_arrays
    f_data_irreg = FDataIrregular(
        start_indices=indices,
        points=arguments,
        values=values,
    )

    if request.param == "single_curve":
        return f_data_irreg[0]
    elif request.param == "multiple_curves":
        return f_data_irreg


@pytest.fixture(params=["single_curve", "multiple_curves"])
def fdatairregular_2d(
    request: Any,
    input_arrays_2d: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """Return FDataIrregular with only 1 curve or NUM_CURVES as requested."""
    indices, arguments, values = input_arrays_2d
    f_data_irreg = FDataIrregular(
        start_indices=indices,
        points=arguments,
        values=values,
    )

    if request.param == "single_curve":
        return f_data_irreg[0]
    elif request.param == "multiple_curves":
        return f_data_irreg


@pytest.fixture(params=["fdatairregular_1d", "fdatairregular_2d"])
def fdatairregular(
    request: Any,
    fdatairregular_1d: FDataIrregular,
    fdatairregular_2d: FDataIrregular,
) -> FDataIrregular:
    """Return 'fdatairregular_1d' or 'fdatairregular_2d'."""
    if request.param == "fdatairregular_1d":
        return fdatairregular_1d
    elif request.param == "fdatairregular_2d":
        return fdatairregular_2d


@pytest.fixture(
    params=[
        "unidimensional",
        "multidimensional",
    ],
)
def fdatairregular_and_sum(request: Any) -> FDataIrregular:
    if request.param == "unidimensional":
        return (
            FDataIrregular(
                start_indices=[0, 3, 7],
                points=[
                    -9, -3, 3, -3, 3, 9, 15, -15, -9, -3, 3, 9, 17, 22, 29,
                ],
                values=[
                    548, 893, 657, 752, 459, 181, 434, 846, 1102, 801, 824,
                    866, 704, 757, 726,
                ],
            ),
            FDataIrregular(
                start_indices=[0],
                points=[-3, 3],
                values=[2446, 1940],
            ),
        )
    if request.param == "multidimensional":
        return (
            FDataIrregular(
                start_indices=[0, 3, 5],
                points=[
                    [0, 0], [1, 2], [1, 1],
                    [0, 0], [1, 1],
                    [0, 0], [6, 2], [1, 1],
                ],
                values=[
                    [0, 0, -1], [657, 752, 5], [10, 20, 30],
                    [-1, 0, 0], [1102, 801, 2],
                    [0, 1, 0], [704, 0, 757], [-11, -21, 31],
                ],
            ),
            FDataIrregular(
                start_indices=[0],
                points=[[0, 0], [1, 1]],
                values=[[-1, 1, -1], [1101, 800, 63]],
            ),
        )


@pytest.fixture(
    params=[
        "unidimensional",
        "multidimensional",
    ],
)
def fdatairregular_common_points(request: Any) -> FDataIrregular:
    if request.param == "unidimensional":
        return FDataIrregular(
            start_indices=[0, 3, 7],
            points=[
                -9, -3, 3, -3, 3, 9, 15, -15, -9, -3, 3, 9, 17, 22, 29,
            ],
            values=[
                548, 893, 657, 752, 459, 181, 434, 846, 1102, 801, 824,
                866, 704, 757, 726,
            ],
        )
    if request.param == "multidimensional":
        return FDataIrregular(
            start_indices=[0, 3, 5],
            points=[
                [0, 0], [1, 2], [1, 1],
                [0, 0], [1, 1],
                [0, 0], [6, 2], [1, 1],
            ],
            values=[
                [0, 0, -1], [657, 752, 5], [10, 20, 30],
                [-1, 0, 0], [1102, 801, 2],
                [0, 1, 0], [704, 0, 757], [-11, -21, 31],
            ],
        )


@pytest.fixture()
def fdatairregular_no_common_points() -> FDataIrregular:
    return FDataIrregular(
        start_indices=[0, 3, 5],
        points=[
            [0, 1], [1, 2], [1, 1],
            [0, -1], [1, 10],
            [0, -2], [6, 2], [10, 1],
        ],
        values=[
            [0, 0, -1], [657, 752, 5], [10, 20, 30],
            [-1, 0, 0], [1102, 801, 2],
            [0, 1, 0], [704, 0, 757], [-11, -21, 31],
        ],
    )


@pytest.fixture(params=["scalar", "vector", "matrix", "fdatairregular"])
def other_1d(
    request: Any,
    fdatairregular_1d: FDataIrregular,
) -> FDataIrregular:
    """Return an operator for testing FDataIrregular operations."""
    if request.param == "scalar":
        return 2
    elif request.param == "vector":
        return 2 * np.ones(NUM_CURVES)
    elif request.param == "matrix":
        return 2 * np.ones((NUM_CURVES, 1))
    elif request.param == "fdatairregular":
        return fdatairregular_1d


@pytest.fixture(params=["scalar", "vector", "matrix", "fdatairregular"])
def other_2d(
    request: Any,
    fdatairregular_2d: FDataIrregular,
) -> FDataIrregular:
    """Return an operator for testing FDataIrregular operations."""
    if request.param == "scalar":
        return 2
    elif request.param == "vector":
        return 2 * np.ones(NUM_CURVES)
    elif request.param == "matrix":
        return 2 * np.ones((NUM_CURVES, DIMENSIONS))
    elif request.param == "fdatairregular":
        return fdatairregular_2d


_all_numeric_reductions = [
    "sum",
    "var",
    "mean",
    # "cov",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request: Any) -> Any:
    """Fixture for numeric reduction names."""
    return request.param


_all_basis_operations = [
    "to_basis",
]


@pytest.fixture(params=_all_basis_operations)
def all_basis_operations(request: Any) -> Any:
    """Fixture for basis operation names."""
    return request.param


_all_basis = [
    FourierBasis,
    BSplineBasis,
]


@pytest.fixture(params=_all_basis)
def all_basis(request: Any) -> Any:
    """Fixture for basis names."""
    return request.param

##################
# TEST OPERATIONS
##################


class TestArithmeticOperations1D:
    """Class for testing basic operations for unidimensional FDataIrregular."""

    def _take_first(
        self,
        other,
    ) -> float:
        if isinstance(other, np.ndarray):
            return other[0]
        elif isinstance(other, FDataIrregular):
            return other.values
        return other

    def _single_curve(
        self,
        fdatairregular_1d,
        other_1d,
    ) -> np.ndarray:
        if isinstance(other_1d, (np.ndarray, FDataIrregular)):
            if len(fdatairregular_1d) == 1:
                return other_1d[:1]
        return other_1d

    def test_fdatairregular_arithmetic_sum(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular + other.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_sum = fdatairregular_1d + other_1d

        result = fdatairregular_1d.values + self._take_first(other_1d)

        assert np.all(f_data_sum.values == result)

    def test_fdatairregular_arithmetic_rsum(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_sum = other_1d + fdatairregular_1d

        result = self._take_first(other_1d) + fdatairregular_1d.values

        assert np.all(f_data_sum.values == result)

    def test_fdatairregular_arithmetic_sum_commutative(  # noqa: WPS118
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        assert np.all(
            (fdatairregular_1d + other_1d) == (other_1d + fdatairregular_1d),
        )

    def test_fdatairregular_arithmetic_sub(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular - other.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_sub = fdatairregular_1d - other_1d

        result = fdatairregular_1d.values - self._take_first(other_1d)

        assert np.all(f_data_sub.values == result)

    def test_fdatairregular_arithmetic_rsub(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other - fdatairregular.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_sub = other_1d - fdatairregular_1d

        result = self._take_first(other_1d) - fdatairregular_1d.values

        assert np.all(f_data_sub.values == result)

    def test_fdatairregular_arithmetic_mul(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular * other.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_mul = fdatairregular_1d * other_1d

        result = fdatairregular_1d.values * self._take_first(other_1d)

        assert np.all(f_data_mul.values == result)

    def test_fdatairregular_arithmetic_rmul(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_mul = other_1d * fdatairregular_1d

        result = self._take_first(other_1d) * fdatairregular_1d.values

        assert np.all(f_data_mul.values == result)

    def test_fdatairregular_arithmetic_mul_commutative(  # noqa: WPS118
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        assert np.all(
            (fdatairregular_1d * other_1d) == (other_1d * fdatairregular_1d),
        )

    def test_fdatairregular_arithmetic_div(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular / other.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_div = fdatairregular_1d / other_1d

        result = fdatairregular_1d.values / self._take_first(other_1d)

        assert np.all(f_data_div.values == result)

    def test_fdatairregular_arithmetic_rdiv(
        self,
        fdatairregular_1d: FDataIrregular,
        other_1d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other / fdatairregular.

        Args:
            fdatairregular_1d (FDataIrregular): FDataIrregular object to test.
            other_1d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_1d = self._single_curve(fdatairregular_1d, other_1d)

        f_data_div = other_1d / fdatairregular_1d

        result = self._take_first(other_1d) / fdatairregular_1d.values

        assert np.all(f_data_div.values == result)


class TestArithmeticOperations2D:
    """Test basic operations for multidimensional FDataIrregular."""

    def _take_first(
        self,
        other,
    ) -> float:
        if isinstance(other, np.ndarray):
            return other[0]
        elif isinstance(other, FDataIrregular):
            return other.values
        return other

    def _single_curve(
        self,
        fdatairregular_2d,
        other_2d,
    ) -> np.ndarray:
        if isinstance(other_2d, (np.ndarray, FDataIrregular)):
            if len(fdatairregular_2d) == 1:
                return other_2d[:1]
        return other_2d

    def test_fdatairregular_arithmetic_sum(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular + other.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_sum = fdatairregular_2d + other_2d

        result = fdatairregular_2d.values + self._take_first(other_2d)

        assert np.all(f_data_sum.values == result)

    def test_fdatairregular_arithmetic_rsum(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_sum = other_2d + fdatairregular_2d

        result = self._take_first(other_2d) + fdatairregular_2d.values

        assert np.all(f_data_sum.values == result)

    def test_fdatairregular_arithmetic_sum_commutative(  # noqa: WPS118
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        assert np.all(
            (fdatairregular_2d + other_2d) == (other_2d + fdatairregular_2d),
        )

    def test_fdatairregular_arithmetic_sub(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular - other.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_sub = fdatairregular_2d - other_2d

        result = fdatairregular_2d.values - self._take_first(other_2d)

        assert np.all(f_data_sub.values == result)

    def test_fdatairregular_arithmetic_rsub(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other - fdatairregular.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_sub = other_2d - fdatairregular_2d

        result = self._take_first(other_2d) - fdatairregular_2d.values

        assert np.all(f_data_sub.values == result)

    def test_fdatairregular_arithmetic_mul(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular * other.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_mul = fdatairregular_2d * other_2d

        result = fdatairregular_2d.values * self._take_first(other_2d)

        assert np.all(f_data_mul.values == result)

    def test_fdatairregular_arithmetic_rmul(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_mul = other_2d * fdatairregular_2d

        result = self._take_first(other_2d) * fdatairregular_2d.values

        assert np.all(f_data_mul.values == result)

    def test_fdatairregular_arithmetic_mul_commutative(  # noqa: WPS118
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        assert np.all(
            (fdatairregular_2d * other_2d) == (other_2d * fdatairregular_2d),
        )

    def test_fdatairregular_arithmetic_div(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular / other.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_div = fdatairregular_2d / other_2d

        result = fdatairregular_2d.values / self._take_first(other_2d)

        assert np.all(f_data_div.values == result)

    def test_fdatairregular_arithmetic_rdiv(
        self,
        fdatairregular_2d: FDataIrregular,
        other_2d: Any,
    ) -> None:
        """Tests the basic arithmetic operation other / fdatairregular.

        Args:
            fdatairregular_2d (FDataIrregular): FDataIrregular object to test.
            other_2d (Any): Scalar, vector, matrix or FDataIrregular.
        """
        # Account for single curve test
        other_2d = self._single_curve(fdatairregular_2d, other_2d)

        f_data_div = other_2d / fdatairregular_2d

        result = self._take_first(other_2d) / fdatairregular_2d.values

        assert np.all(f_data_div.values == result)


##########################
# TEST NUMERIC REDUCTIONS
##########################

class TestNumericReductions:
    """Class for testing numeric reductions (mean, std) for FDataIrregular."""

    def test_fdatairregular_numeric_reduction(
        self,
        fdatairregular_common_points: FDataIrregular,
        all_numeric_reductions: str,
    ) -> None:
        """Test FDataIrregular numeric statistichal operations.

        All conversion methods will be tested with multiple
        dimensions of codomain and domain.

        Args:
            fdatairregular_common_points (FDataIrregular): FDataIrregular
                object with points common to all samples.
            all_numeric_reductions (str): Method of the class
                FDataIrregular to be tested.
        """
        reduction = getattr(
            fdatairregular_common_points, all_numeric_reductions,
        )()
        assert isinstance(reduction, FDataIrregular)

    def test_fdatairregular_sum(
        self,
        fdatairregular_and_sum: Tuple[FDataIrregular, FDataIrregular],
    ) -> None:
        """Test the sum function for FDataIrregular.

        Test both unidimensional and multidimensional.

        Args:
            fdatairregular_and_sum: FDataIrregular object and expected sum.
        """
        fdatairregular, expected_sum = fdatairregular_and_sum
        actual_sum = fdatairregular.sum()
        assert actual_sum.equals(expected_sum)

    def test_fdatairregular_mean(
        self,
        fdatairregular_and_sum: Tuple[FDataIrregular, FDataIrregular],
    ) -> None:
        """Test the mean function for FDataIrregular.

        Test both unidimensional and multidimensional.

        Args:
            fdatairregular_and_sum: FDataIrregular object and expected sum.
        """
        fdatairregular, expected_sum = fdatairregular_and_sum
        actual_mean = fdatairregular.mean()
        assert actual_mean.equals(expected_sum / fdatairregular.n_samples)

    def test_fdatairregular_sum_invalid(
        self,
        fdatairregular_no_common_points: FDataIrregular,
    ) -> None:
        """Test the sum function for FDataIrregular.

        Args:
            fdatairregular_no_common_points: FDataIrregular object with no
                common points.
        """
        with pytest.raises(ValueError):
            fdatairregular_no_common_points.sum()


########################
# TEST BASIS OPERATIONS
########################


class TestBasisOperations:
    """Class for testing the basis operations or FDataIrregular objects."""

    def test_fdatairregular_basis_operation(
        self,
        fdatairregular: FDataIrregular,
        all_basis: Basis,
        all_basis_operations: str,
    ) -> None:
        """Test FDataIrregular conversion to FDataBasis.

        All conversion methods will be tested with multiple
        dimensions of codomain and domain, as well as with
        different types of Basis.

        Args:
            fdatairregular (FDataIrregular): FDataIrregular
                object to be transformed to basis.
            all_basis (Basis): Basis to use (Spline, Fourier, ..).
            all_basis_operations (str): Method of the class
                FDataIrregular to be tested.
        """
        # Create Tensor basis for higher dimensions
        if fdatairregular.dim_domain == 1:
            basis = all_basis(
                domain_range=fdatairregular.domain_range,
                n_basis=N_BASIS,
            )
        else:
            basis_by_dim = [
                all_basis(
                    domain_range=fdatairregular.domain_range[dim: dim + 1],
                    n_basis=N_BASIS,
                )
                for dim in range(fdatairregular.dim_domain)
            ]
            basis = TensorBasis(basis_by_dim)

        fd_basis_coords = [
            getattr(coordinate, all_basis_operations)(basis)
            for coordinate in fdatairregular.coordinates
        ]

        assert all(
            isinstance(fd_basis, FDataBasis) for fd_basis in fd_basis_coords
        )


def test_fdatairregular_to_basis_consistency(
    fdatagrid: FDataGrid,
    all_basis: Basis,
) -> None:
    """Test that irregular to_basis is consistent with FDataGrid.

    FDataGrid is used as source because FDataIrregular can support
    regular data, but the reverse is not necessarily true. The
    to_basis method specifically does not allow NaN values.

    Args:
        fdatagrid (FDataGrid): FDataGrid object
        all_basis (Basis): FDataBasis object.
    """
    fd_irregular = FDataIrregular.from_fdatagrid(fdatagrid)

    if fd_irregular.dim_domain == 1:
        basis = all_basis(
            domain_range=fd_irregular.domain_range,
            n_basis=N_BASIS,
        )
    else:
        basis_by_dim = [
            all_basis(
                domain_range=fd_irregular.domain_range[dim: dim + 1],
                n_basis=N_BASIS,
            )
            for dim in range(fd_irregular.dim_domain)
        ]
        basis = TensorBasis(basis_by_dim)

    irregular_basis = [
        coord.to_basis(basis)
        for coord in fd_irregular.coordinates
    ]

    grid_basis = [
        coord.to_basis(basis)
        for coord in fdatagrid.coordinates
    ]

    irregular_coefs = [
        b.coefficients.round(DECIMALS)
        for b in irregular_basis
    ]

    grid_coefs = [
        b.coefficients.round(DECIMALS)
        for b in grid_basis
    ]

    assert all(
        np.all(irregular_coefs[i] == g_coef)
        for i, g_coef in enumerate(grid_coefs)
    )

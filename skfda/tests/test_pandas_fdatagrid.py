"""Tests for Pandas ExtensionArray for FDataGrid."""
from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension import base
from typing_extensions import override

import skfda
from skfda.representation.grid import FDataGrid

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas.api.extensions import ExtensionArray, ExtensionDtype


##############################################################################
# Fixtures
##############################################################################
@pytest.fixture
def dtype() -> ExtensionDtype:
    """Return the ExtensionDtype to validate."""
    return skfda.representation.grid.FDataGridDType(
        grid_points=[
            np.arange(9),
            np.arange(8) / 7,
        ],
        dim_codomain=3,
    )


@pytest.fixture
def data() -> ExtensionArray:
    """
    Return data.

    Length-10 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal

    """
    data_matrix = np.arange(1, 10 * 9 * 8 * 3 + 1).reshape(10, 9, 8, 3)
    grid_points: list[NDArray[np.number]] = [
        np.arange(9),
        np.arange(8) / 7,
    ]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture
def data_for_twos() -> ExtensionArray:
    """Return a length-10 array in which all the elements are two."""
    data_matrix = np.full(
        10 * 9 * 8 * 3, fill_value=2,
    ).reshape(10, 9, 8, 3)
    grid_points: list[NDArray[np.number]] = [
        np.arange(9),
        np.arange(8) / 7,
    ]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture
def data_missing() -> ExtensionArray:
    """Return a length-2 array with [NA, Valid]."""
    data_matrix = np.arange(
        2 * 9 * 8 * 3,
        dtype=np.float64,
    ).reshape(2, 9, 8, 3)
    data_matrix[0, ...] = np.nan
    grid_points: list[NDArray[np.number]] = [
        np.arange(9),
        np.arange(8) / 7,
    ]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture(params=["data", "data_missing"])
def all_data(
    request: pytest.FixtureRequest,
    data: ExtensionArray,
    data_missing: ExtensionArray,
) -> ExtensionArray:
    """Return 'data' or 'data_missing'."""
    if request.param == "data":
        return data

    if request.param == "data_missing":
        return data_missing

    msg = "Unreachable"
    raise ValueError(msg)


@pytest.fixture
def data_repeated(
    data: ExtensionArray,
) -> Callable[[int], Generator[ExtensionArray, None, None]]:
    """
    Generate many datasets.

    Args:
        data : Fixture implementing `data`

    Returns:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.

    """

    def gen(count: int) -> Generator[ExtensionArray, None, None]:
        yield from (
            data for _ in range(count)
        )

    return gen


@pytest.fixture
def data_for_sorting() -> NoReturn:
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C

    For boolean dtypes (for which there are only 2 values available),
    set B=C=True
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting() -> NoReturn:
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


@pytest.fixture
def na_cmp() -> Callable[..., bool]:
    """
    Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    def isna(
        x: pd.api.typing.NAType | FDataGrid,
        y: pd.api.typing.NAType | FDataGrid,
    ) -> bool:
        return (
            (x is pd.NA or all(x.isna()))
            and (y is pd.NA or all(y.isna()))
        )

    return isna


@pytest.fixture
def na_value() -> pd.api.typing.NAType:
    """Return the scalar missing value for this type. Default 'None'."""
    return pd.NA


@pytest.fixture
def data_for_grouping() -> NoReturn:
    """
    Return data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request: pytest.FixtureRequest) -> bool:
    """Whether to box the data in a Series."""
    return request.param


@pytest.fixture(
    params=list[Callable[[FDataGrid], Any]]([
        lambda x: 1,  # noqa: ARG005
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ]),
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(
    request: pytest.FixtureRequest,
) -> Callable[[FDataGrid], Any]:
    """Functions to test groupby.apply()."""
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request: pytest.FixtureRequest) -> bool:
    """Whether to support Series and Series.to_frame() comparison testing."""
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request: pytest.FixtureRequest) -> bool:
    """Boolean fixture to support arr and Series(arr) comparison testing."""
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request: pytest.FixtureRequest) -> bool:
    """
    Compare ExtensionDtype and numpy.

    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request: pytest.FixtureRequest) -> str:
    """
    Series.fillna parameter fixture.

    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request: pytest.FixtureRequest) -> bool:
    """Whether to support ExtensionDtype _from_sequence method testing."""
    return request.param


_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    #     "__floordiv__",
    #     "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    #     "__pow__",
    #     "__rpow__",
    #     "__mod__",
    #     "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request: pytest.FixtureRequest) -> str:
    """Fixture for dunder names for common arithmetic operations."""
    return request.param


@pytest.fixture(params=["__eq__", "__ne__",
                        # "__le__", "__lt__", "__ge__", "__gt__"
                        ])
def all_compare_operators(request: pytest.FixtureRequest) -> str:
    """Fixture for dunder names for common compare operations."""
    return request.param


_all_numeric_reductions = [
    "sum",
    #     "max",
    #     "min",
    "mean",
    #     "prod",
    #     "std",
    #     "var",
    #     "median",
    #     "kurt",
    #     "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request: pytest.FixtureRequest) -> str:
    """Fixture for numeric reduction names."""
    return request.param


_all_boolean_reductions: Sequence[str] = [
    # "all",
    # "any",
]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request: pytest.FixtureRequest) -> str:
    """Fixture for boolean reduction names."""
    return request.param

##############################################################################
# Tests
##############################################################################


class TestCasting(base.BaseCastingTests):  # type: ignore[misc]
    """Casting to and from the ExtensionDtype."""

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_astype_str(self, data: ExtensionArray) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_astype_string(
        self,
        data: ExtensionArray,
        nullable_string_dtype: ExtensionDtype,
    ) -> None:
        pass


class TestConstructors(base.BaseConstructorsTests):  # type: ignore[misc]
    """Tests of constructors."""

    # Does not support scalars which are also ExtensionArrays
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_series_constructor_scalar_with_index(
        self,
        data: ExtensionArray,
        dtype: ExtensionDtype,
    ) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_from_dtype(
        self,
        data: ExtensionArray,
    ) -> None:
        pass


class TestDtype(base.BaseDtypeTests):  # type: ignore[misc]
    """Tests of the ExtensionDtype."""

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_construct_from_string_own_name(
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_is_dtype_from_name(
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_eq_with_str(
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass


class TestGetitem(base.BaseGetitemTests):  # type: ignore[misc]
    """Tests for ExtensionArray.__getitem__."""

    # We do not support readonly property
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_getitem_propagates_readonly_property(
        self,
        data: ExtensionArray,
    ) -> None:
        pass


class TestInterface(base.BaseInterfaceTests):  # type: ignore[misc]
    """Tests that the basic interface is satisfied."""

    # Does not support scalars which are also array_like
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_array_interface(
        self,
        data: ExtensionArray,
    ) -> None:
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_copy(
        self,
        data: ExtensionArray,
    ) -> None:
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_view(
        self,
        data: ExtensionArray,
    ) -> None:
        pass

    # Pending https://github.com/pandas-dev/pandas/issues/38812 resolution
    @pytest.mark.skip(reason="Bugged")
    @override
    def test_contains(
        self,
        data: ExtensionArray,
        data_missing: ExtensionArray,
        using_nan_is_na: bool,
    ) -> None:
        pass

    # We do not attempt to implement copy right now.
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_array_interface_copy(
        self,
        data: ExtensionArray,
    ) -> None:
        pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):  # type: ignore[misc]
    """Various Series and DataFrame arithmetic ops methods."""

    series_scalar_exc = None

    # Bug introduced by https://github.com/pandas-dev/pandas/pull/37132
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_arith_frame_with_scalar(
        self,
        data: ExtensionArray,
        all_arithmetic_operators: Callable[..., Any],
    ) -> None:
        pass

    # Does not convert properly a list of FData to a FData
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_arith_series_with_array(
        self,
        data: ExtensionArray,
        all_arithmetic_operators: str,
    ) -> None:
        pass


class TestComparisonOps(base.BaseComparisonOpsTests):  # type: ignore[misc]
    """Various Series and DataFrame comparison ops methods."""

    # Cannot be compared with 0
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_compare_scalar(
        self,
        data: ExtensionArray,
        comparison_op: Callable[..., Any],
    ) -> None:
        pass

    # Not sure how to pass it. Should it be reimplemented?
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_compare_array(
        self,
        data: ExtensionArray,
        comparison_op: Callable[..., Any],
    ) -> None:
        pass


class TestReduce(base.BaseReduceTests ):  # type: ignore[misc]
    """Reduction specific tests."""

    @override
    def _supports_reduction(
        self,
        ser: pd.Series,
        op_name: str,
    ) -> bool:
        return op_name in ["sum", "mean"]

    # Unsupported for now
    @pytest.mark.skip(reason="Unsupported")
    @override
    def test_reduce_frame(
        self,
        data: ExtensionArray,
        all_numeric_reductions: str,
        skipna: bool,
    ) -> None:
        pass

    @override
    def check_reduce(
        self,
        ser: pd.Series,
        op_name: str,
        skipna: bool,
    ) -> None:
        result = getattr(ser, op_name)(skipna=skipna)
        assert result.n_samples == 1

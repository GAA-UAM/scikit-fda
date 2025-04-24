from __future__ import annotations  # noqa: D100, I001

from typing import Any, Callable, Generator, NoReturn, Sequence, Union  # noqa: UP035

import numpy as np
import pandas  # noqa: ICN001
import pytest
from pandas import Series
from pandas.api.extensions import ExtensionArray, ExtensionDtype  # noqa: TC002
from pandas.tests.extension import base

import skfda
from skfda.representation.grid import FDataGrid  # noqa: TC001


##############################################################################
# Fixtures
##############################################################################
@pytest.fixture
def dtype() -> ExtensionDtype:
    """Return the ExtensionDtype to validate."""
    return skfda.representation.grid.FDataGridDType(
        grid_points=[
            np.arange(10),
            np.arange(10) / 10,
        ],
        dim_codomain=3,
    )


@pytest.fixture
def data() -> ExtensionArray:
    """
    Return data.

    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal

    """
    data_matrix = np.arange(1, 100 * 10 * 10 * 3 + 1).reshape(100, 10, 10, 3)
    grid_points = [
        np.arange(10),
        np.arange(10) / 10,
    ]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture
def data_for_twos() -> ExtensionArray:
    """Return a length-100 array in which all the elements are two."""
    data_matrix = np.full(
        100 * 10 * 10 * 3, fill_value=2,
    ).reshape(100, 10, 10, 3)
    grid_points = [
        np.arange(10),
        np.arange(10) / 10,
    ]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture
def data_missing() -> ExtensionArray:
    """Return a length-2 array with [NA, Valid]."""
    data_matrix = np.arange(
        2 * 10 * 10 * 3,
        dtype=np.float64,
    ).reshape(2, 10, 10, 3)
    data_matrix[0, ...] = np.nan
    grid_points = [
        np.arange(10),
        np.arange(10) / 10,
    ]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture(params=["data", "data_missing"])
def all_data(
    request: Any,  # noqa: ANN401
    data: ExtensionArray,
    data_missing: ExtensionArray,
) -> ExtensionArray:
    """Return 'data' or 'data_missing'."""
    if request.param == "data":  # noqa: RET503
        return data
    elif request.param == "data_missing":  # noqa: RET505
        return data_missing


@pytest.fixture
def data_repeated(
    data: ExtensionArray,
) -> Callable[[int], Generator[ExtensionArray, None, None]]:
    """
    Generate many datasets.

    Args:
        data : Fixture implementing `data`

    Returns:
        Callable[[int], Generator]:
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
    Return ength-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting() -> NoReturn:
    """
    Return length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """  # noqa: D205
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
        x: Union[pandas.NA, FDataGrid],  # noqa: UP007
        y: Union[pandas.NA, FDataGrid],  # noqa: UP007
    ) -> bool:
        return (
            (x is pandas.NA or all(x.isna()))
            and (y is pandas.NA or all(y.isna()))
        )

    return isna


@pytest.fixture
def na_value() -> pandas.NA:
    """Return the scalar missing value for this type. Default 'None'."""
    return pandas.NA


@pytest.fixture
def data_for_grouping() -> NoReturn:
    """
    Return data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request: Any) -> Any:  # noqa: ANN401
    """Whether to box the data in a Series."""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,  # noqa: ARG005
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request: Any) -> Any:  # noqa: ANN401
    """Functions to test groupby.apply()."""
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request: Any) -> Any:  # noqa: ANN401
    """Whether to support Series and Series.to_frame() comparison testing."""
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request: Any) -> Any:  # noqa: ANN401
    """Boolean fixture to support arr and Series(arr) comparison testing."""
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request: Any) -> Any:  # noqa: ANN401
    """
    Compare ExtensionDtype and numpy.

    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request: Any) -> Any:  # noqa: ANN401
    """
    Series.fillna parameter fixture.

    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request: Any) -> Any:  # noqa: ANN401
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
def all_arithmetic_operators(request: Any) -> Any:  # noqa: ANN401
    """
    Fixture for dunder names for common arithmetic operations.
    """  # noqa: D200
    return request.param


@pytest.fixture(params=["__eq__", "__ne__",
                        # "__le__", "__lt__", "__ge__", "__gt__"
                        ])
def all_compare_operators(request: Any) -> Any:  # noqa: ANN401
    """
    Fixture for dunder names for common compare operations
    """  # noqa: D200, D415
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
def all_numeric_reductions(request: Any) -> Any:  # noqa: ANN401
    """
    Fixture for numeric reduction names.
    """  # noqa: D200
    return request.param


_all_boolean_reductions: Sequence[str] = [
    # "all",
    # "any",
]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request: Any) -> Any:  # noqa: ANN401
    """
    Fixture for boolean reduction names.
    """  # noqa: D200
    return request.param

##############################################################################
# Tests
##############################################################################


class TestCasting(base.BaseCastingTests):  # type: ignore[misc]  # noqa: D101

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_astype_str(self) -> None:  # noqa: D102
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_astype_string(self) -> None:  # noqa: D102
        pass


class TestConstructors(base.BaseConstructorsTests):  # type: ignore[misc]  # noqa: D101

    # Does not support scalars which are also ExtensionArrays
    @pytest.mark.skip(reason="Unsupported")
    def test_series_constructor_scalar_with_index(self) -> None:  # noqa: D102
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_from_dtype(self) -> None:  # noqa: D102
        pass


class TestDtype(base.BaseDtypeTests):  # type: ignore[misc]  # noqa: D101

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_construct_from_string_own_name(self) -> None:  # noqa: D102
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_is_dtype_from_name(self) -> None:  # noqa: D102
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_eq_with_str(self) -> None:  # noqa: D102
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_construct_from_string(  # noqa: D102
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass


class TestGetitem(base.BaseGetitemTests):  # type: ignore[misc]  # noqa: D101
    pass


class TestInterface(base.BaseInterfaceTests):  # type: ignore[misc]  # noqa: D101

    # Does not support scalars which are also array_like
    @pytest.mark.skip(reason="Unsupported")
    def test_array_interface(self) -> None:  # noqa: D102
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    def test_copy(  # noqa: D102
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    def test_view(  # noqa: D102
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass

    # Pending https://github.com/pandas-dev/pandas/issues/38812 resolution
    @pytest.mark.skip(reason="Bugged")
    def test_contains(  # noqa: D102
        self,
        data: ExtensionArray,
        data_missing: ExtensionArray,
    ) -> None:
        pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):  # type: ignore[misc]  # noqa: D101

    series_scalar_exc = None

    # Bug introduced by https://github.com/pandas-dev/pandas/pull/37132
    @pytest.mark.skip(reason="Unsupported")
    def test_arith_frame_with_scalar(  # noqa: D102
        self,
        data: ExtensionArray,
        all_arithmetic_operators: Callable[..., Any],
    ) -> None:
        pass

    # Does not convert properly a list of FData to a FData
    @pytest.mark.skip(reason="Unsupported")
    def test_arith_series_with_array(  # noqa: D102
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass

    # Does not error on operations
    @pytest.mark.skip(reason="Unsupported")
    def test_error(  # noqa: D102
        self,
        dtype: ExtensionDtype,
    ) -> None:
        pass


class TestComparisonOps(base.BaseComparisonOpsTests):  # type: ignore[misc]  # noqa: D101

    # Cannot be compared with 0
    @pytest.mark.skip(reason="Unsupported")
    def test_compare_scalar(  # noqa: D102
        self,
        data: ExtensionArray,
        all_compare_operators: Callable[..., Any],
    ) -> None:
        pass

    # Not sure how to pass it. Should it be reimplemented?
    @pytest.mark.skip(reason="Unsupported")
    def test_compare_array(  # noqa: D102
        self,
        data: ExtensionArray,
        all_compare_operators: Callable[..., Any],
    ) -> None:
        pass


class TestNumericReduce(base.BaseNumericReduceTests):  # type: ignore[misc]  # noqa: D101

    def check_reduce(  # noqa: D102
        self,
        s: FDataGrid,
        op_name: str,
        skipna: bool,  # noqa: FBT001
    ) -> None:
        result = getattr(s, op_name)(skipna=skipna)
        assert result.n_samples == 1

from __future__ import annotations

from typing import Any, Callable, Iterable, NoReturn

import numpy as np
import pandas
import pytest
from pandas import Series
from pandas.tests.extension import base

from skfda.representation.basis import (
    Basis,
    BSplineBasis,
    FDataBasis,
    FDataBasisDType,
    FourierBasis,
    MonomialBasis,
)


##############################################################################
# Fixtures
##############################################################################
@pytest.fixture(params=[
    MonomialBasis(n_basis=5),
    FourierBasis(n_basis=5),
    BSplineBasis(n_basis=5),
])
def basis(request: Any) -> Any:
    """A fixture providing the ExtensionDtype to validate."""
    return request.param


@pytest.fixture
def dtype(basis: Basis) -> FDataBasisDType:
    """A fixture providing the ExtensionDtype to validate."""
    return FDataBasisDType(basis=basis)


@pytest.fixture
def data(basis: Basis) -> FDataBasis:
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    coef_matrix = np.arange(100 * basis.n_basis).reshape(100, basis.n_basis)

    return FDataBasis(basis=basis, coefficients=coef_matrix)


@pytest.fixture
def data_for_twos() -> NoReturn:
    """Length-100 array in which all the elements are two."""
    raise NotImplementedError


@pytest.fixture
def data_missing(basis: Basis) -> FDataBasis:
    """Length-2 array with [NA, Valid]."""
    coef_matrix = np.arange(
        2 * basis.n_basis,
        dtype=np.float_,
    ).reshape(2, basis.n_basis)
    coef_matrix[0, :] = np.NaN

    return FDataBasis(basis=basis, coefficients=coef_matrix)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request: Any, data: Any, data_missing: Any) -> Any:
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data: FDataBasis) -> Callable[[int], Iterable[FDataBasis]]:
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count: int) -> Iterable[FDataBasis]:
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting() -> NoReturn:
    """
    Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
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
def na_cmp() -> Callable[[Any, Any], bool]:
    """
    Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    def isna(x: Any, y: Any) -> bool:
        return (
            (x is pandas.NA or all(x.isna()))
            and (y is pandas.NA or all(y.isna()))
        )

    return isna


@pytest.fixture
def na_value() -> pandas.NA:
    """The scalar missing value for this type. Default 'None'"""
    return pandas.NA


@pytest.fixture
def data_for_grouping() -> NoReturn:
    """
    Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request: Any) -> Any:
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request: Any) -> Any:
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request: Any) -> Any:
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request: Any) -> Any:
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request: Any) -> Any:
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request: Any) -> Any:
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request: Any) -> Any:
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    #     "__mul__",
    #     "__rmul__",
    #     "__floordiv__",
    #     "__rfloordiv__",
    #     "__truediv__",
    #     "__rtruediv__",
    #     "__pow__",
    #     "__rpow__",
    #     "__mod__",
    #     "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request: Any) -> Any:
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__",
                        # "__le__", "__lt__", "__ge__", "__gt__"
                        ])
def all_compare_operators(request: Any) -> Any:
    """
    Fixture for dunder names for common compare operations
    """
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
def all_numeric_reductions(request: Any) -> Any:
    """
    Fixture for numeric reduction names.
    """
    return request.param

##############################################################################
# Tests
##############################################################################


class TestCasting(base.BaseCastingTests):  # type: ignore[misc]

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_astype_str(self) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_astype_string(self) -> None:
        pass


class TestConstructors(base.BaseConstructorsTests):  # type: ignore[misc]

    # Does not support scalars which are also ExtensionArrays
    @pytest.mark.skip(reason="Unsupported")
    def test_series_constructor_scalar_with_index(self) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_from_dtype(self) -> None:
        pass


class TestDtype(base.BaseDtypeTests):  # type: ignore[misc]

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_construct_from_string_own_name(self) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_is_dtype_from_name(self) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_eq_with_str(self) -> None:
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_construct_from_string(self, dtype: Any) -> None:
        pass


class TestGetitem(base.BaseGetitemTests):  # type: ignore[misc]
    pass


class TestInterface(base.BaseInterfaceTests):  # type: ignore[misc]

    # Does not support scalars which are also array_like
    @pytest.mark.skip(reason="Unsupported")
    def test_array_interface(self) -> None:
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    def test_copy(self, dtype: Any) -> None:
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    def test_view(self, dtype: Any) -> None:
        pass

    # Pending https://github.com/pandas-dev/pandas/issues/38812 resolution
    @pytest.mark.skip(reason="Bugged")
    def test_contains(self, data: Any, data_missing: Any) -> None:
        pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):  # type: ignore[misc]

    series_scalar_exc = None

    # Bug introduced by https://github.com/pandas-dev/pandas/pull/37132
    @pytest.mark.skip(reason="Unsupported")
    def test_arith_frame_with_scalar(
        self,
        data: Any,
        all_arithmetic_operators: Any,
    ) -> None:
        pass

    # FDatabasis does not implement division by non constant
    @pytest.mark.skip(reason="Unsupported")
    def test_divmod_series_array(self, dtype: Any) -> None:
        pass

    # Does not convert properly a list of FData to a FData
    @pytest.mark.skip(reason="Unsupported")
    def test_arith_series_with_array(self, dtype: Any) -> None:
        pass

    # Does not error on operations
    @pytest.mark.skip(reason="Unsupported")
    def test_error(self, dtype: Any) -> None:
        pass


class TestComparisonOps(base.BaseComparisonOpsTests):  # type: ignore[misc]

    # Cannot be compared with 0
    @pytest.mark.skip(reason="Unsupported")
    def test_compare_scalar(
        self,
        data: Any,
        all_compare_operators: Any,
    ) -> None:
        pass

    # Not sure how to pass it. Should it be reimplemented?
    @pytest.mark.skip(reason="Unsupported")
    def test_compare_array(
        self,
        data: Any,
        all_compare_operators: Any,
    ) -> None:
        pass


class TestNumericReduce(base.BaseNumericReduceTests):  # type: ignore[misc]

    def check_reduce(
        self,
        s: Any,
        op_name: str,
        skipna: bool,
    ) -> None:
        result = getattr(s, op_name)(skipna=skipna)
        assert result.n_samples == 1

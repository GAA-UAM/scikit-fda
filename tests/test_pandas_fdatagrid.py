import operator

import numpy as np
import pandas
import pytest
from pandas import Series
from pandas.tests.extension import base

import skfda


##############################################################################
# Fixtures
##############################################################################
@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return skfda.representation.grid.FDataGridDType(
        grid_points=[
            np.arange(10),
            np.arange(10) / 10],
        dim_codomain=3
    )


@pytest.fixture
def data():
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """

    data_matrix = np.arange(1, 100 * 10 * 10 * 3 + 1).reshape(100, 10, 10, 3)
    grid_points = [
        np.arange(10),
        np.arange(10) / 10]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""

    data_matrix = np.full(
        100 * 10 * 10 * 3, fill_value=2).reshape(100, 10, 10, 3)
    grid_points = [
        np.arange(10),
        np.arange(10) / 10]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""

    data_matrix = np.arange(
        2 * 10 * 10 * 3, dtype=np.float_).reshape(2, 10, 10, 3)
    data_matrix[0, ...] = np.NaN
    grid_points = [
        np.arange(10),
        np.arange(10) / 10]

    return skfda.FDataGrid(data_matrix, grid_points=grid_points)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
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

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """
    Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting():
    """
    Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


@pytest.fixture
def na_cmp():
    """
    Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    def isna(x, y):
        return ((x is pandas.NA or all(x.isna()))
                and (y is pandas.NA or all(y.isna())))

    return isna


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return pandas.NA


@pytest.fixture
def data_for_grouping():
    """
    Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request):
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
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
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
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(params=["__eq__", "__ne__",
                        # "__le__", "__lt__", "__ge__", "__gt__"
                        ])
def all_compare_operators(request):
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
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param

##############################################################################
# Tests
##############################################################################


class TestCasting(base.BaseCastingTests):

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_astype_str(self):
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_astype_string(self):
        pass


class TestConstructors(base.BaseConstructorsTests):

    # Does not support scalars which are also ExtensionArrays
    @pytest.mark.skip(reason="Unsupported")
    def test_series_constructor_scalar_with_index(self):
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_from_dtype(self):
        pass


class TestDtype(base.BaseDtypeTests):

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_construct_from_string_own_name(self):
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_is_dtype_from_name(self):
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_eq_with_str(self):
        pass

    # Tries to construct dtype from string
    @pytest.mark.skip(reason="Unsupported")
    def test_construct_from_string(self, dtype):
        pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestInterface(base.BaseInterfaceTests):

    # Does not support scalars which are also array_like
    @pytest.mark.skip(reason="Unsupported")
    def test_array_interface(self):
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    def test_copy(self, dtype):
        pass

    # We do not implement setitem
    @pytest.mark.skip(reason="Unsupported")
    def test_view(self, dtype):
        pass

    # Pending https://github.com/pandas-dev/pandas/issues/38812 resolution
    @pytest.mark.skip(reason="Bugged")
    def test_contains(self, data, data_missing):
        pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):

    series_scalar_exc = None

    # Does not convert properly a list of FData to a FData
    @pytest.mark.skip(reason="Unsupported")
    def test_arith_series_with_array(self, dtype):
        pass

    # Does not error on operations
    @pytest.mark.skip(reason="Unsupported")
    def test_error(self, dtype):
        pass


class TestComparisonOps(base.BaseComparisonOpsTests):

    # Cannot be compared with 0
    @pytest.mark.skip(reason="Unsupported")
    def test_compare_scalar(self, data, all_compare_operators):
        pass

    # Not sure how to pass it. Should it be reimplemented?
    @pytest.mark.skip(reason="Unsupported")
    def test_compare_array(self, data, all_compare_operators):
        pass


class TestNumericReduce(base.BaseNumericReduceTests):

    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        assert result.n_samples == 1

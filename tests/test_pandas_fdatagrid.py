import operator
import skfda

from pandas import Series
import pandas
from pandas.tests.extension import base
import pytest

import numpy as np


##############################################################################
# Fixtures
##############################################################################
@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return skfda.representation.grid.FDataGridDType(
        sample_points=[
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

    data_matrix = np.arange(100 * 10 * 10 * 3).reshape(100, 10, 10, 3)
    sample_points = [
        np.arange(10),
        np.arange(10) / 10]

    return skfda.FDataGrid(data_matrix, sample_points=sample_points)


@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""
    raise NotImplementedError


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    raise NotImplementedError


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
    return operator.is_


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

##############################################################################
# Tests
##############################################################################


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

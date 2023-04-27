"""Test the basic methods of the FDataIrregular structure"""
from ..typing._numpy import ArrayLike, Any
from typing import Tuple, Optional
import numpy as np
import pytest

from skfda.datasets._real_datasets import _fetch_loon_data
from skfda.representation import FDataIrregular, FDataGrid
from skfda.representation.interpolation import SplineInterpolation
from skfda.representation.basis import Basis, FDataBasis, FourierBasis, BSplineBasis, TensorBasis

############
# MACROS
############
SEED = 2906198114

NUM_CURVES = 100
MAX_VALUES_PER_CURVE = 10
DIMENSIONS = 2
N_BASIS = 5
DECIMALS = 4

random_state = np.random.RandomState(seed=SEED)

############
# FIXTURES
############

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
def fdatagrid1D(
) -> FDataGrid:
    """Generate FDataGrid"""
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.rand(NUM_CURVES, num_values_per_curve, 1)
    # Grid points must be sorted
    grid_points = np.sort(random_state.rand(num_values_per_curve))

    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )

@pytest.fixture()
def fdatagrid2D(
) -> FDataGrid:
    """Generate multidimensional FDataGrid."""
    num_values_per_curve = NUM_CURVES

    data_matrix = random_state.rand(
        NUM_CURVES,
        num_values_per_curve,
        DIMENSIONS,
    )

    # Grid points must be sorted
    grid_points = np.sort(random_state.rand(num_values_per_curve))

    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
    )

@pytest.fixture(
    params=[
        "fdatagrid1D",
        "fdatagrid2D",
    ],
)
def fdatagrid(
    request: Any,
    fdatagrid1D: FDataGrid,
    fdatagrid2D: FDataGrid,
) -> FDataIrregular:
    """Return 'fdatagrid1D' or 'fdatagrid2D'."""
    if request.param == "fdatagrid1D":
        return fdatagrid1D
    elif request.param == "fdatagrid2D":
        return fdatagrid2D

@pytest.fixture(params=["single_curve", "multiple_curves"])
def fdatairregular1D(
    request: Any,
    input_arrays: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """Return FDataIrregular with only 1 curve or NUM_CURVES as requested."""
    indices, arguments, values = input_arrays
    f_data_irreg = FDataIrregular(
        function_indices=indices,
        function_arguments=arguments,
        function_values=values,
    )
    
    if request.param == "single_curve":
        return f_data_irreg[0]
    elif request.param == "multiple_curves":
        return f_data_irreg
    
@pytest.fixture(params=["single_curve", "multiple_curves"])
def fdatairregular2D(
    request: Any,
    input_arrays_2D: Tuple[ArrayLike, ArrayLike, ArrayLike],
) -> FDataIrregular:
    """Return FDataIrregular with only 1 curve or NUM_CURVES as requested."""
    indices, arguments, values = input_arrays_2D
    f_data_irreg = FDataIrregular(
        function_indices=indices,
        function_arguments=arguments,
        function_values=values,
    )
    
    if request.param == "single_curve":
        return f_data_irreg[0]
    elif request.param == "multiple_curves":
        return f_data_irreg

@pytest.fixture(params=["fdatairregular1D", "fdatairregular2D"])
def fdatairregular(
    request: Any,
    fdatairregular1D: FDataIrregular,
    fdatairregular2D: FDataIrregular,
) -> FDataIrregular:
    """Return 'fdatairregular1D' or 'fdatairregular2D'."""
    if request.param == "fdatairregular1D":
        return fdatairregular1D
    elif request.param == "fdatairregular2D":
        return fdatairregular2D

@pytest.fixture(params=["scalar", "vector", "matrix", "fdatairregular"])
def other_1D(
    request: Any,
    fdatairregular1D: FDataIrregular,
) -> FDataIrregular:
    """Return an operator for testing FDataIrregular operations."""
    if request.param == "scalar":
        return 2
    elif request.param == "vector":
        return 2*np.ones(NUM_CURVES)
    elif request.param == "matrix":
        return 2*np.ones((NUM_CURVES, 1))
    elif request.param == "fdatairregular":
        return fdatairregular1D
    
@pytest.fixture(params=["scalar", "vector", "matrix", "fdatairregular"])
def other_2D(
    request: Any,
    fdatairregular2D: FDataIrregular,
) -> FDataIrregular:
    """Return an operator for testing FDataIrregular operations."""
    if request.param == "scalar":
        return 2
    elif request.param == "vector":
        return 2*np.ones(NUM_CURVES)
    elif request.param == "matrix":
        return 2*np.ones((NUM_CURVES, DIMENSIONS))
    elif request.param == "fdatairregular":
        return fdatairregular2D
    
_all_numeric_reductions = [
    "sum",
    "var",
    "mean",
    #"cov",
]

@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request: Any) -> Any:
    """
    Fixture for numeric reduction names.
    """
    return request.param

_all_basis_operations = [
    "to_basis",
]

@pytest.fixture(params=_all_basis_operations)
def all_basis_operations(request: Any) -> Any:
    """
    Fixture for basis operation names.
    """
    return request.param

_all_basis = [
    FourierBasis,
    BSplineBasis,
]

@pytest.fixture(params=_all_basis)
def all_basis(request: Any) -> Any:
    """
    Fixture for basis names.
    """
    return request.param

##################
# TEST OPERATIONS
##################
class TestArithmeticOperations1D:
    """
    Class which encapsulates the testing of basic arithmetic operations 
    for unidimensional FDataIrregular
    """

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
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular + other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_sum = fdatairregular1D + other_1D

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular1D.function_values + self._take_first(other_1D)
            )

    def test_fdatairregular_arithmetic_rsum(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_sum = other_1D + fdatairregular1D

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other_1D) + fdatairregular1D.function_values
            )

    def test_fdatairregular_arithmetic_sum_commutative(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        assert fdatairregular1D + other_1D == other_1D + fdatairregular1D

    def test_fdatairregular_arithmetic_sub(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular - other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_sum = fdatairregular1D - other_1D

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular1D.function_values - self._take_first(other_1D)
            )

    def test_fdatairregular_arithmetic_rsub(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other - fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_sum = other_1D - fdatairregular1D

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other_1D) - fdatairregular1D.function_values
            )

    def test_fdatairregular_arithmetic_mul(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular * other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_mul = fdatairregular1D * other_1D

        assert np.all(
            f_data_mul.function_values ==
            fdatairregular1D.function_values * self._take_first(other_1D)
            )

    def test_fdatairregular_arithmetic_rmul(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_mul = other_1D * fdatairregular1D

        assert np.all(
            f_data_mul.function_values ==
            self._take_first(other_1D) * fdatairregular1D.function_values
            )

    def test_fdatairregular_arithmetic_mul_commutative(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        assert fdatairregular1D * other_1D == other_1D * fdatairregular1D

    def test_fdatairregular_arithmetic_div(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular / other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_div = fdatairregular1D / other_1D

        assert np.all(
            f_data_div.function_values ==
            fdatairregular1D.function_values / self._take_first(other_1D)
            )

    def test_fdatairregular_arithmetic_rdiv(
        self,
        fdatairregular1D: FDataIrregular,
        other_1D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other / fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_1D, np.ndarray) or isinstance(other_1D, FDataIrregular):
            if len(fdatairregular1D) == 1:
                other_1D = other_1D[0]

        f_data_div = other_1D / fdatairregular1D

        assert np.all(
            f_data_div.function_values ==
            self._take_first(other_1D) / fdatairregular1D.function_values
            )

class TestArithmeticOperations2D:
    """
    Class which encapsulates the testing of basic arithmetic operations 
    for multidimensional FDataIrregular
    """

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
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular + other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_sum = fdatairregular2D + other_2D

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular2D.function_values + self._take_first(other_2D)
            )

    def test_fdatairregular_arithmetic_rsum(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_sum = other_2D + fdatairregular2D

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other_2D) + fdatairregular2D.function_values
            )

    def test_fdatairregular_arithmetic_sum_commutative(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other + fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        assert fdatairregular2D + other_2D == other_2D + fdatairregular2D

    def test_fdatairregular_arithmetic_sub(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular - other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_sum = fdatairregular2D - other_2D

        assert np.all(
            f_data_sum.function_values ==
            fdatairregular2D.function_values - self._take_first(other_2D)
            )

    def test_fdatairregular_arithmetic_rsub(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other - fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_sum = other_2D - fdatairregular2D

        assert np.all(
            f_data_sum.function_values ==
            self._take_first(other_2D) - fdatairregular2D.function_values
            )

    def test_fdatairregular_arithmetic_mul(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular * other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_mul = fdatairregular2D * other_2D

        assert np.all(
            f_data_mul.function_values ==
            fdatairregular2D.function_values * self._take_first(other_2D)
            )

    def test_fdatairregular_arithmetic_rmul(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_mul = other_2D * fdatairregular2D

        assert np.all(
            f_data_mul.function_values ==
            self._take_first(other_2D) * fdatairregular2D.function_values
            )

    def test_fdatairregular_arithmetic_mul_commutative(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other * fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        assert fdatairregular2D * other_2D == other_2D * fdatairregular2D

    def test_fdatairregular_arithmetic_div(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation fdatairregular / other

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_div = fdatairregular2D / other_2D

        assert np.all(
            f_data_div.function_values ==
            fdatairregular2D.function_values / self._take_first(other_2D)
            )

    def test_fdatairregular_arithmetic_rdiv(
        self,
        fdatairregular2D: FDataIrregular,
        other_2D: Any,
    ) -> None:
        """Tests the basic arithmetic operation other / fdatairregular

        Args:
            fdatairregular (FDataIrregular): FDataIrregular object to test
            other (Any): Scalar, vector, matrix or FDataIrregular
        """
        # Account for single curve test
        if isinstance(other_2D, np.ndarray) or isinstance(other_2D, FDataIrregular):
            if len(fdatairregular2D) == 1:
                other_2D = other_2D[:1]

        f_data_div = other_2D / fdatairregular2D

        assert np.all(
            f_data_div.function_values ==
            self._take_first(other_2D) / fdatairregular2D.function_values
            )


##########################
# TEST NUMERIC REDUCTIONS
##########################

class TestNumericReductions:
    """
    Class which encapsulates the testing of numeric reductions
    (such as mean, std) for FDataIrregular objects
    """
    def test_fdatairregular_numeric_reduction(
        self,
        fdatairregular: FDataIrregular,
        all_numeric_reductions: str,
    ) -> None:
    
        reduction = getattr(fdatairregular, all_numeric_reductions)()
        assert isinstance(reduction, FDataIrregular)

########################
# TEST BASIS OPERATIONS
########################

class TestBasisOperations:
    """
    Class which encapsulates the testing of basis operations
    (such as to_basis) for FDataIrregular objects
    """
    def test_fdatairregular_basis_operation(
        self,
        fdatairregular: FDataIrregular,
        all_basis: Basis,
        all_basis_operations: str,
    ) -> None:
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
        
        assert all([isinstance(fd_basis, FDataBasis) for fd_basis in fd_basis_coords])


def test_fdatairregular_to_basis_consistency(
    fdatagrid: FDataIrregular,
    all_basis : Basis,
) -> None:
    """Test that irregular to_basis is consistent with
    analogous FDataGrid to_basis in 1D.

    Args:
        fdatairregular1D (FDataIrregular): FDataIrregular
            object with dimensions (1,1).
        all_basis (Basis): FDataBasis object.
    """
    fd_irregular = FDataIrregular.from_datagrid(fdatagrid)
    
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
        [
            np.all(irregular_coefs[i] == g_coef)
            for i, g_coef in enumerate(grid_coefs)
        ],
    )
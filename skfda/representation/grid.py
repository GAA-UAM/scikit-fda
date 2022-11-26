"""Discretised functional data module.

This module defines a class for representing functional data as a series of
lists of values, each representing the observation of a function measured in a
list of discretisation points.

"""
from __future__ import annotations

import copy
import numbers
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import findiff
import numpy as np
import pandas.api.extensions
import scipy.integrate
import scipy.stats.mstats
from matplotlib.figure import Figure

from .._utils import _check_array_key, _int_to_real, _to_grid_points, constants
from ..typing._base import (
    DomainRange,
    DomainRangeLike,
    GridPoints,
    GridPointsLike,
    LabelTupleLike,
)
from ..typing._numpy import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt
from ._functional_data import FData
from .evaluator import Evaluator
from .extrapolation import ExtrapolationLike
from .interpolation import SplineInterpolation

if TYPE_CHECKING:
    from .basis import Basis, FDataBasis

T = TypeVar("T", bound='FDataGrid')


class FDataGrid(FData):  # noqa: WPS214
    r"""Represent discretised functional data.

    Class for representing functional data as a set of curves discretised
    in a grid of points.

    Attributes:
        data_matrix: a matrix where each entry of the first
            axis contains the values of a functional datum evaluated at the
            points of discretisation.
        grid_points: 2 dimension matrix where each row
            contains the points of dicretisation for each axis of data_matrix.
        domain_range: 2 dimension matrix where each row
            contains the bounds of the interval in which the functional data
            is considered to exist for each one of the axies.
        dataset_name: name of the dataset.
        argument_names: tuple containing the names of the different
            arguments.
        coordinate_names: tuple containing the names of the different
            coordinate functions.
        extrapolation: defines the default type of
            extrapolation. By default None, which does not apply any type of
            extrapolation. See `Extrapolation` for detailled information of the
            types of extrapolation.
        interpolation: Defines the type of interpolation
            applied in `evaluate`.

    Examples:
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`,
        with 3 discretization points.

        >>> data_matrix = [[1, 2, 3], [4, 5, 6]]
        >>> grid_points = [2, 4, 5]
        >>> FDataGrid(data_matrix, grid_points)
        FDataGrid(
            array([[[ 1.],
                    [ 2.],
                    [ 3.]],
        <BLANKLINE>
                   [[ 4.],
                    [ 5.],
                    [ 6.]]]),
            grid_points=(array([ 2., 4., 5.]),),
            domain_range=((2.0, 5.0),),
            ...)

        The number of columns of data_matrix have to be the length of
        grid_points.

        >>> FDataGrid(np.array([1,2,4,5,8]), range(6))
        Traceback (most recent call last):
            ....
        ValueError: Incorrect dimension in data_matrix and grid_points...


        FDataGrid support higher dimensional data both in the domain and image.
        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [2, 4]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fd.dim_domain, fd.dim_codomain
        (1, 2)

        Representation of a functional data object with 2 samples
        representing a function :math:`f : \mathbb{R}^2\longmapsto\mathbb{R}`.

        >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
        >>> grid_points = [[2, 4], [3,6]]
        >>> fd = FDataGrid(data_matrix, grid_points)
        >>> fd.dim_domain, fd.dim_codomain
        (2, 1)

    """

    def __init__(  # noqa: WPS211
        self,
        data_matrix: ArrayLike,
        grid_points: Optional[GridPointsLike] = None,
        *,
        sample_points: Optional[GridPointsLike] = None,
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
        interpolation: Optional[Evaluator] = None,
    ):
        """Construct a FDataGrid object."""
        from ..misc.validation import validate_domain_range

        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        self.data_matrix = _int_to_real(np.atleast_2d(data_matrix))

        if grid_points is None:
            self.grid_points = _to_grid_points([
                np.linspace(0, 1, self.data_matrix.shape[i])
                for i in range(1, self.data_matrix.ndim)
            ])

        else:
            # Check that the dimension of the data matches the grid_points
            # list

            self.grid_points = _to_grid_points(grid_points)

            data_shape = self.data_matrix.shape[1: 1 + self.dim_domain]
            grid_points_shape = [len(i) for i in self.grid_points]

            if not np.array_equal(data_shape, grid_points_shape):
                raise ValueError(
                    f"Incorrect dimension in data_matrix and "
                    f"grid_points. Data has shape {data_shape} and grid "
                    f"points have shape {grid_points_shape}",
                )

        self._sample_range = tuple(
            (s[0], s[-1]) for s in self.grid_points
        )

        if domain_range is None:
            domain_range = self.sample_range
            # Default value for domain_range is a list of tuples with
            # the first and last element of each list of the grid_points.

        self._domain_range = validate_domain_range(domain_range)

        if len(self._domain_range) != self.dim_domain:
            raise ValueError("Incorrect shape of domain_range.")

        for domain_range, grid_points in zip(
            self._domain_range,
            self.grid_points,
        ):
            if (
                domain_range[0] > grid_points[0]
                or domain_range[-1] < grid_points[-1]
            ):
                raise ValueError(
                    "Grid points must be within the domain range.",
                )

        # Adjust the data matrix if the dimension of the image is one
        if self.data_matrix.ndim == 1 + self.dim_domain:
            self.data_matrix = self.data_matrix[..., np.newaxis]

        self.interpolation = interpolation  # type: ignore[assignment]

        super().__init__(
            extrapolation=extrapolation,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
        )

    def round(  # noqa: WPS125
        self,
        decimals: int = 0,
        out: Optional[FDataGrid] = None,
    ) -> FDataGrid:
        """Evenly round to the given number of decimals.

        .. deprecated:: 0.6
            Use :func:`numpy.round` function instead.

        Args:
            decimals: Number of decimal places to round to.
                If decimals is negative, it specifies the number of
                positions to the left of the decimal point. Defaults to 0.
            out: FDataGrid where to place the result, if any.

        Returns:
            Returns a FDataGrid object where all elements
            in its data_matrix are rounded.

        """
        out_matrix = None if out is None else out.data_matrix

        if (
            out is not None
            and (
                self.domain_range != out.domain_range
                or not all(
                    np.array_equal(a, b)
                    for a, b in zip(self.grid_points, out.grid_points)
                )
                or self.data_matrix.shape != out.data_matrix.shape
            )
        ):
            raise ValueError("out parameter is not valid")

        data_matrix = np.round(
            self.data_matrix,
            decimals=decimals,
            out=out_matrix,
        )

        return self.copy(data_matrix=data_matrix) if out is None else out

    @property
    def sample_points(self) -> GridPoints:
        warnings.warn(
            "Parameter sample_points is deprecated. Use the "
            "parameter grid_points instead.",
            DeprecationWarning,
        )
        return self.grid_points

    @property
    def dim_domain(self) -> int:
        return len(self.grid_points)

    @property
    def dim_codomain(self) -> int:
        try:
            # The dimension of the image is the length of the array that can
            #  be extracted from the data_matrix using all the dimensions of
            #  the domain.
            return self.data_matrix.shape[1 + self.dim_domain]
        # If there is no array that means the dimension of the image is 1.
        except IndexError:
            return 1

    @property
    def coordinates(self: T) -> _CoordinateIterator[T]:
        r"""Returns an object to access to the image coordinates.

        If the functional object contains multivariate samples
        :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}^d`, this class allows
        iterate and get coordinates of the vector
        :math:`f = (f_0, ..., f_{d-1})`.

        Examples:

            We will construct a dataset of curves in :math:`\mathbb{R}^3`

            >>> from skfda.datasets import make_multimodal_samples
            >>> fd = make_multimodal_samples(dim_codomain=3, random_state=0)
            >>> fd.dim_codomain
            3

            The functions of this dataset are vectorial functions
            :math:`f(t) = (f_0(t), f_1(t), f_2(t))`. We can obtain a specific
            component of the vector, for example, the first one.

            >>> fd_0 = fd.coordinates[0]
            >>> fd_0
            FDataGrid(...)

            The object returned has image dimension equal to 1

            >>> fd_0.dim_codomain
            1

            Or we can get multiple components, it can be accesed as a 1-d
            numpy array of coordinates, for example, :math:`(f_0(t), f_1(t))`.

            >>> fd_01 = fd.coordinates[0:2]
            >>> fd_01.dim_codomain
            2

            We can use this method to iterate throught all the coordinates.

            >>> for fd_i in fd.coordinates:
            ...     fd_i.dim_codomain
            1
            1
            1

            This object can be used to split a FDataGrid in a list with
            their components.

            >>> fd_list = list(fd.coordinates)
            >>> len(fd_list)
            3

        """

        return _CoordinateIterator(self)

    @property
    def n_samples(self) -> int:
        """
        Return the number of samples.

        This is also the number of rows of the data_matrix.

        Returns:
            Number of samples of the FDataGrid object.

        """
        return self.data_matrix.shape[0]

    @property
    def sample_range(self) -> DomainRange:
        """
        Return the sample range of the function.

        This contains the minimum and maximum values of the grid points in
        each dimension.

        It does not have to be equal to the `domain_range`.
        """
        return self._sample_range

    @property
    def domain_range(self) -> DomainRange:
        """
        Return the :term:`domain range` of the function.

        It does not have to be equal to the `sample_range`.

        """
        return self._domain_range

    @property
    def interpolation(self) -> Evaluator:
        """Define the type of interpolation applied in `evaluate`."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, new_interpolation: Optional[Evaluator]) -> None:

        if new_interpolation is None:
            new_interpolation = SplineInterpolation()

        self._interpolation = new_interpolation

    def _evaluate(
        self,
        eval_points: NDArrayFloat,
        *,
        aligned: bool = True,
    ) -> NDArrayFloat:

        return self.interpolation(
            self,
            eval_points,
            aligned=aligned,
        )

    def derivative(
        self: T,
        *,
        order: int = 1,
        method: Optional[Basis] = None,
    ) -> T:
        """
        Differentiate a FDataGrid object.

        By default, it is calculated using central finite differences when
        possible. In the extremes, forward and backward finite differences
        with accuracy 2 are used.

        Args:
            order: Order of the derivative. Defaults to one.
            method: Method to use to compute the derivative. If ``None``
                (the default), finite differences are used. In a basis
                object is passed the grid is converted to a basis
                representation and the derivative is evaluated using that
                representation.

        Returns:
            Derivative function.

        Examples:
            First order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative()
            FDataGrid(
                array([[[ 0.5],
                        [ 1.5],
                        [ 1.5],
                        [ 2. ],
                        [ 4. ]]]),
                grid_points=(array([ 0., 1., 2., 3., 4.]),),
                domain_range=((0.0, 4.0),),
                ...)

            Second order derivative

            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.derivative(order=2)
            FDataGrid(
                array([[[ 3.],
                        [ 1.],
                        [-1.],
                        [ 2.],
                        [ 5.]]]),
                grid_points=(array([ 0., 1., 2., 3., 4.]),),
                domain_range=((0.0, 4.0),),
                ...)

        """
        order_list = np.atleast_1d(order)
        if order_list.ndim != 1 or len(order_list) != self.dim_domain:
            raise ValueError("The order for each partial should be specified.")

        if method is None:
            operator = findiff.FinDiff(*[
                (1 + i, *z)
                for i, z in enumerate(
                    zip(self.grid_points, order_list),
                )
            ])
            data_matrix = operator(self.data_matrix.astype(float))
        else:
            data_matrix = self.to_basis(method).derivative(
                order=order,
            )(
                self.grid_points,
                grid=True,
            )

        return self.copy(
            data_matrix=data_matrix,
        )

    def integrate(
        self: T,
        *,
        domain: Optional[DomainRange] = None,
    ) -> NDArrayFloat:
        """
        Integration of the FData object.

        The integration is performed over the whole domain. Thus, for a
        function of several variables this will be a multiple integral.

        For a vector valued function the vector of integrals will be
        returned.

        Args:
            domain: Domain range where we want to integrate.
                By default is None as we integrate on the whole domain.

        Returns:
            NumPy array of size (``n_samples``, ``dim_codomain``)
            with the integrated data.

        Examples:
            >>> fdata = FDataGrid([1,2,4,5,8], range(5))
            >>> fdata.integrate()
            array([[ 15.]])
        """
        if domain is not None:
            data = self.restrict(domain)
        else:
            data = self

        integrand = data.data_matrix

        for g in data.grid_points[::-1]:
            integrand = scipy.integrate.simps(
                integrand,
                x=g,
                axis=-2,
            )

        return integrand

    def _check_same_dimensions(self: T, other: T) -> None:
        if self.data_matrix.shape[1:-1] != other.data_matrix.shape[1:-1]:
            raise ValueError("Error in columns dimensions")
        if not np.array_equal(self.grid_points, other.grid_points):
            raise ValueError("Grid points for both objects must be equal")

    def sum(  # noqa: WPS125
        self: T,
        *,
        axis: Optional[int] = None,
        out: None = None,
        keepdims: bool = False,
        skipna: bool = False,
        min_count: int = 0,
    ) -> T:
        """Compute the sum of all the samples.

        Args:
            axis: Used for compatibility with numpy. Must be None or 0.
            out: Used for compatibility with numpy. Must be None.
            keepdims: Used for compatibility with numpy. Must be False.
            skipna: Wether the NaNs are ignored or not.
            min_count: Number of valid (non NaN) data to have in order
                for the a variable to not be NaN when `skipna` is
                `True`.

        Returns:
            A FDataGrid object with just one sample representing
            the sum of all the samples in the original object.

        Examples:
            >>> from skfda import FDataGrid
            >>> data_matrix = [[0.5, 1, 2, .5], [1.5, 1, 4, .5]]
            >>> FDataGrid(data_matrix).sum()
            FDataGrid(
                array([[[ 2.],
                        [ 2.],
                        [ 6.],
                        [ 1.]]]),
                ...)

        """
        super().sum(axis=axis, out=out, keepdims=keepdims, skipna=skipna)

        data = (
            np.nansum(self.data_matrix, axis=0, keepdims=True) if skipna
            else np.sum(self.data_matrix, axis=0, keepdims=True)
        )

        if min_count > 0:
            valid = ~np.isnan(self.data_matrix)
            n_valid = np.sum(valid, axis=0)
            data[n_valid < min_count] = np.NaN

        return self.copy(
            data_matrix=data,
            sample_names=(None,),
        )

    def var(self: T) -> T:
        """Compute the variance of a set of samples in a FDataGrid object.

        Returns:
            A FDataGrid object with just one sample representing the
            variance of all the samples in the original FDataGrid object.

        """
        return self.copy(
            data_matrix=np.array([np.var(self.data_matrix, 0)]),
            sample_names=("variance",),
        )

    def cov(self: T) -> T:
        """Compute the covariance.

        Calculates the covariance matrix representing the covariance of the
        functional samples at the observation points.

        Returns:
            Covariance function.

        """
        dataset_name = (
            f"{self.dataset_name} - covariance"
            if self.dataset_name is not None else None
        )

        if self.dim_domain != 1 or self.dim_codomain != 1:
            raise NotImplementedError(
                "Covariance only implemented "
                "for univariate functions",
            )

        return self.copy(
            data_matrix=np.cov(
                self.data_matrix[..., 0],
                rowvar=False,
            )[np.newaxis, ...],
            grid_points=[
                self.grid_points[0],
                self.grid_points[0],
            ],
            domain_range=[
                self.domain_range[0],
                self.domain_range[0],
            ],
            dataset_name=dataset_name,
            argument_names=self.argument_names * 2,
            sample_names=("covariance",),
        )

    def gmean(self: T) -> T:
        """Compute the geometric mean of all samples in the FDataGrid object.

        Returns:
            A FDataGrid object with just one sample representing
            the geometric mean of all the samples in the original
            FDataGrid object.

        """
        return self.copy(
            data_matrix=[
                scipy.stats.mstats.gmean(self.data_matrix, 0),
            ],
            sample_names=("geometric mean",),
        )

    def equals(self, other: object) -> bool:
        """Comparison of FDataGrid objects."""
        if not super().equals(other):
            return False

        other = cast(FDataGrid, other)

        if not np.array_equal(self.data_matrix, other.data_matrix):
            return False

        # Comparison of the domain
        if (
            not np.array_equal(self.domain_range, other.domain_range)
            or len(self.grid_points) != len(other.grid_points)
            or not all(
                np.array_equal(a, b)
                for a, b in zip(self.grid_points, other.grid_points)
            )
        ):
            return False

        return self.interpolation == other.interpolation

    def _eq_elemenwise(self: T, other: T) -> NDArrayBool:
        """Elementwise equality of FDataGrid."""
        return np.all(  # type: ignore[no-any-return]
            self.data_matrix == other.data_matrix,
            axis=tuple(range(1, self.data_matrix.ndim)),
        )

    def _get_op_matrix(
        self,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> Union[None, float, NDArrayFloat, NDArrayInt]:
        if isinstance(other, numbers.Real):
            return float(other)
        elif isinstance(other, np.ndarray):

            if other.shape in {(), (1,)}:
                return other
            elif other.shape == (self.n_samples,):
                other_index = (
                    (slice(None),) + (np.newaxis,)
                    * (self.data_matrix.ndim - 1)
                )

                return other[other_index]
            elif other.shape == (
                self.n_samples,
                self.dim_codomain,
            ):
                other_index = (
                    (slice(None),) + (np.newaxis,)
                    * (self.data_matrix.ndim - 2)
                    + (slice(None),)
                )

                return other[other_index]

            raise ValueError(
                f"Invalid dimensions in operator between FDataGrid and Numpy "
                f"array: {other.shape}"
            )

        elif isinstance(other, FDataGrid):
            self._check_same_dimensions(other)
            return other.data_matrix

        return None

    def __add__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix + data_matrix)

    def __radd__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        return self.__add__(other)

    def __sub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix - data_matrix)

    def __rsub__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self.copy(data_matrix=data_matrix - self.data_matrix)

    def __mul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix * data_matrix)

    def __rmul__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        return self.__mul__(other)

    def __truediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=self.data_matrix / data_matrix)

    def __rtruediv__(
        self: T,
        other: Union[T, NDArrayFloat, NDArrayInt, float],
    ) -> T:

        data_matrix = self._get_op_matrix(other)
        if data_matrix is None:
            return NotImplemented

        return self._copy_op(other, data_matrix=data_matrix / self.data_matrix)

    def __neg__(self: T) -> T:
        """Negation of FData object."""
        return self.copy(data_matrix=-self.data_matrix)

    def concatenate(self: T, *others: T, as_coordinates: bool = False) -> T:
        """Join samples from a similar FDataGrid object.

        Joins samples from another FDataGrid object if it has the same
        dimensions and sampling points.

        Args:
            others: Objects to be concatenated.
            as_coordinates:  If False concatenates as
                new samples, else, concatenates the other functions as
                new components of the image. Defaults to false.

        Returns:
            FDataGrid object with the samples from the original objects.

        Examples:
            >>> fd = FDataGrid([1,2,4,5,8], range(5))
            >>> fd_2 = FDataGrid([3,4,7,9,2], range(5))
            >>> fd.concatenate(fd_2)
            FDataGrid(
                array([[[ 1.],
                        [ 2.],
                        [ 4.],
                        [ 5.],
                        [ 8.]],
            <BLANKLINE>
                       [[ 3.],
                        [ 4.],
                        [ 7.],
                        [ 9.],
                        [ 2.]]]),
                grid_points=(array([ 0., 1., 2., 3., 4.]),),
                domain_range=((0.0, 4.0),),
                ...)

        """
        # Checks
        if not as_coordinates:
            for other in others:
                self._check_same_dimensions(other)

        elif not all(
            np.array_equal(self.grid_points, other.grid_points)
            for other in others
        ):
            raise ValueError(
                "All the FDataGrids must be sampled in the  same "
                "grid points.",
            )

        elif any(self.n_samples != other.n_samples for other in others):

            raise ValueError(
                f"All the FDataGrids must contain the same "
                f"number of samples {self.n_samples} to "
                f"concatenate as a new coordinate.",
            )

        data = [self.data_matrix] + [other.data_matrix for other in others]

        if as_coordinates:

            coordinate_names = [fd.coordinate_names for fd in (self, *others)]

            return self.copy(
                data_matrix=np.concatenate(data, axis=-1),
                coordinate_names=sum(coordinate_names, ()),
            )

        sample_names = [fd.sample_names for fd in (self, *others)]

        return self.copy(
            data_matrix=np.concatenate(data, axis=0),
            sample_names=sum(sample_names, ()),
        )

    def scatter(self, *args: Any, **kwargs: Any) -> Figure:
        """Scatter plot of the FDatGrid object.

        Args:
            args: Positional arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.ScatterPlot`.
            kwargs: Keyword arguments to be passed to the class
                :class:`~skfda.exploratory.visualization.representation.ScatterPlot`.

        Returns:
            Figure object in which the graphs are plotted.


        """
        from ..exploratory.visualization.representation import ScatterPlot

        return ScatterPlot(self, *args, **kwargs).plot()

    def to_basis(self, basis: Basis, **kwargs: Any) -> FDataBasis:
        """Return the basis representation of the object.

        Args:
            basis(Basis): basis object in which the functional data are
                going to be represented.
            kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.

        Examples:
            >>> import numpy as np
            >>> import skfda
            >>> t = np.linspace(0, 1, 5)
            >>> x = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t) + 2
            >>> x
            array([ 3.,  3.,  1.,  1.,  3.])

            >>> fd = FDataGrid(x, t)
            >>> basis = skfda.representation.basis.FourierBasis(n_basis=3)
            >>> fd_b = fd.to_basis(basis)
            >>> fd_b.coefficients.round(2)
            array([[ 2.  , 0.71, 0.71]])

        """
        from ..preprocessing.smoothing import BasisSmoother

        if self.dim_domain != basis.dim_domain:
            raise ValueError(
                f"The domain of the function has "
                f"dimension {self.dim_domain} "
                f"but the domain of the basis has "
                f"dimension {basis.dim_domain}",
            )
        elif self.dim_codomain != basis.dim_codomain:
            raise ValueError(
                f"The codomain of the function has "
                f"dimension {self.dim_codomain} "
                f"but the codomain of the basis has "
                f"dimension {basis.dim_codomain}",
            )

        # Readjust the domain range if there was not an explicit one
        if not basis.is_domain_range_fixed():
            basis = basis.copy(domain_range=self.domain_range)

        smoother = BasisSmoother(
            basis=basis,
            **kwargs,
            return_basis=True,
        )

        return smoother.fit_transform(self)

    def to_grid(  # noqa: D102
        self: T,
        grid_points: Optional[GridPointsLike] = None,
        *,
        sample_points: Optional[GridPointsLike] = None,
    ) -> T:

        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        grid_points = (
            self.grid_points
            if grid_points is None
            else _to_grid_points(grid_points)
        )

        return self.copy(
            data_matrix=self(grid_points, grid=True),
            grid_points=grid_points,
        )

    def copy(  # noqa: WPS211
        self: T,
        *,
        deep: bool = False,  # For Pandas compatibility
        data_matrix: Optional[ArrayLike] = None,
        grid_points: Optional[GridPointsLike] = None,
        sample_points: Optional[GridPointsLike] = None,
        domain_range: Optional[DomainRangeLike] = None,
        dataset_name: Optional[str] = None,
        argument_names: Optional[LabelTupleLike] = None,
        coordinate_names: Optional[LabelTupleLike] = None,
        sample_names: Optional[LabelTupleLike] = None,
        extrapolation: Optional[ExtrapolationLike] = None,
        interpolation: Optional[Evaluator] = None,
    ) -> T:
        """
        Return a copy of the FDataGrid.

        If an argument is provided the corresponding attribute in the new copy
        is updated.

        """
        if sample_points is not None:
            warnings.warn(
                "Parameter sample_points is deprecated. Use the "
                "parameter grid_points instead.",
                DeprecationWarning,
            )
            grid_points = sample_points

        if data_matrix is None:
            # The data matrix won't be writeable
            data_matrix = self.data_matrix

        if grid_points is None:
            # Grid points won`t be writeable
            grid_points = self.grid_points

        if domain_range is None:
            domain_range = copy.deepcopy(self.domain_range)

        if dataset_name is None:
            dataset_name = self.dataset_name

        if argument_names is None:
            # Tuple, immutable
            argument_names = self.argument_names

        if coordinate_names is None:
            # Tuple, immutable
            coordinate_names = self.coordinate_names

        if sample_names is None:
            # Tuple, immutable
            sample_names = self.sample_names

        if extrapolation is None:
            extrapolation = self.extrapolation

        if interpolation is None:
            interpolation = self.interpolation

        return FDataGrid(
            data_matrix,
            grid_points=grid_points,
            domain_range=domain_range,
            dataset_name=dataset_name,
            argument_names=argument_names,
            coordinate_names=coordinate_names,
            sample_names=sample_names,
            extrapolation=extrapolation,
            interpolation=interpolation,
        )

    def restrict(
        self: T,
        domain_range: DomainRangeLike,
    ) -> T:
        """
        Restrict the functions to a new domain range.

        Args:
            domain_range: New domain range.

        Returns:
            Restricted function.

        """
        from ..misc.validation import validate_domain_range

        domain_range = validate_domain_range(domain_range)
        assert all(
            c <= a < b <= d  # noqa: WPS228
            for ((a, b), (c, d)) in zip(domain_range, self.domain_range)
        )

        index_list = []
        new_grid_points = []

        # Eliminate points outside the new range.
        for dr, grid_points in zip(
            domain_range,
            self.grid_points,
        ):
            keep_index = (
                (dr[0] <= grid_points)
                & (grid_points <= dr[1])
            )

            index_list.append(keep_index)

            new_grid_points.append(
                grid_points[keep_index],
            )

        data_matrix = self.data_matrix[(slice(None),) + tuple(index_list)]

        return self.copy(
            domain_range=domain_range,
            grid_points=new_grid_points,
            data_matrix=data_matrix,
        )

    def shift(
        self,
        shifts: Union[ArrayLike, float],
        *,
        restrict_domain: bool = False,
        extrapolation: Optional[ExtrapolationLike] = None,
        grid_points: Optional[GridPointsLike] = None,
    ) -> FDataGrid:
        r"""
        Perform a shift of the curves.

        The i-th shifted function :math:`y_i` has the form

        .. math::
            y_i(t) = x_i(t + \delta_i)

        where :math:`x_i` is the i-th original function and :math:`delta_i` is
        the shift performed for that function, that must be a vector in the
        domain space.

        Note that a positive shift moves the graph of the function in the
        negative direction and vice versa.

        Args:
            shifts: List with the shifts
                corresponding for each sample or numeric with the shift to
                apply to all samples.
            restrict_domain: If True restricts the domain to avoid the
                evaluation of points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation: Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See extrapolation to
                more information.
            grid_points: Grid of points where
                the functions are evaluated to obtain the discrete
                representation of the object to operate. If ``None`` the
                current grid_points are used to unificate the domain of the
                shifted data.

        Returns:
            Shifted functions.

        Examples:
            >>> import numpy as np
            >>> import skfda
            >>>
            >>> t = np.linspace(0, 1, 6)
            >>> x = np.array([t, t**2, t**3])
            >>> fd = skfda.FDataGrid(x, t)
            >>> fd.domain_range[0]
            (0.0, 1.0)
            >>> fd.grid_points[0]
            array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
            >>> fd.data_matrix[..., 0]
            array([[ 0.   ,  0.2  ,  0.4  ,  0.6  ,  0.8  ,  1.   ],
                   [ 0.   ,  0.04 ,  0.16 ,  0.36 ,  0.64 ,  1.   ],
                   [ 0.   ,  0.008,  0.064,  0.216,  0.512,  1.   ]])

            Shift all curves by the same amount:

            >>> shifted = fd.shift(0.2)
            >>> shifted.domain_range[0]
            (0.0, 1.0)
            >>> shifted.grid_points[0]
            array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
            >>> shifted.data_matrix[..., 0]
            array([[ 0.2  ,  0.4  ,  0.6  ,  0.8  ,  1.   ,  1.2  ],
                   [ 0.04 ,  0.16 ,  0.36 ,  0.64 ,  1.   ,  1.36 ],
                   [ 0.008,  0.064,  0.216,  0.512,  1.   ,  1.488]])


            Different shift per curve:

            >>> shifted = fd.shift([-0.2, 0.0, 0.2])
            >>> shifted.domain_range[0]
            (0.0, 1.0)
            >>> shifted.grid_points[0]
            array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
            >>> shifted.data_matrix[..., 0]
            array([[-0.2  ,  0.   ,  0.2  ,  0.4  ,  0.6  ,  0.8  ],
                   [ 0.   ,  0.04 ,  0.16 ,  0.36 ,  0.64 ,  1.   ],
                   [ 0.008,  0.064,  0.216,  0.512,  1.   ,  1.488]])

            It is possible to restrict the domain to prevent the need for
            extrapolations:

            >>> shifted = fd.shift([-0.3, 0.1, 0.2], restrict_domain=True)
            >>> shifted.domain_range[0]
            (0.3, 0.8)

        """
        grid_points = (
            self.grid_points if grid_points is None
            else grid_points
        )

        return super().shift(
            shifts=shifts,
            restrict_domain=restrict_domain,
            extrapolation=extrapolation,
            grid_points=grid_points,
        )

    def compose(
        self: T,
        fd: T,
        *,
        eval_points: Optional[GridPointsLike] = None,
    ) -> T:
        """Composition of functions.

        Performs the composition of functions.

        Args:
            fd: FData object to make the composition. Should
                have the same number of samples and image dimension equal to 1.
            eval_points: Points to perform the evaluation.

        Returns:
            Function representing the composition.

        """
        if self.dim_domain != fd.dim_codomain:
            raise ValueError(
                f"Dimension of codomain of first function do not "
                f"match with the domain of the second function "
                f"{self.dim_domain} != {fd.dim_codomain}.",
            )

        # All composed with same function
        if fd.n_samples == 1 and self.n_samples != 1:
            fd = fd.copy(data_matrix=np.repeat(
                fd.data_matrix,
                self.n_samples,
                axis=0,
            ))

        if fd.dim_domain == 1:
            if eval_points is None:
                try:
                    eval_points = fd.grid_points[0]
                except AttributeError:
                    eval_points = np.linspace(
                        *fd.domain_range[0],
                        constants.N_POINTS_COARSE_MESH,
                    )

            eval_points_transformation = fd(eval_points)
            data_matrix = self(
                eval_points_transformation,
                aligned=False,
            )
        else:
            eval_points = (
                fd.grid_points
                if eval_points is None
                else _to_grid_points(eval_points)
            )

            grid_transformation = fd(eval_points, grid=True)

            lengths = [len(ax) for ax in eval_points]

            eval_points_transformation = np.empty((
                self.n_samples,
                np.prod(lengths),
                self.dim_domain,
            ))

            for i in range(self.n_samples):
                eval_points_transformation[i] = np.array(
                    list(map(np.ravel, grid_transformation[i].T)),
                ).T

            data_matrix = self(
                eval_points_transformation,
                aligned=False,
            )

        return self.copy(
            data_matrix=data_matrix,
            grid_points=eval_points,
            domain_range=fd.domain_range,
            argument_names=fd.argument_names,
        )

    def __str__(self) -> str:
        """Return str(self)."""
        return (
            f"Data set:    {self.data_matrix}\n"
            f"grid_points:    {self.grid_points}\n"
            f"time range:    {self.domain_range}"
        )

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"FDataGrid("  # noqa: WPS221
            f"\n{self.data_matrix!r},"
            f"\ngrid_points={self.grid_points!r},"
            f"\ndomain_range={self.domain_range!r},"
            f"\ndataset_name={self.dataset_name!r},"
            f"\nargument_names={self.argument_names!r},"
            f"\ncoordinate_names={self.coordinate_names!r},"
            f"\nextrapolation={self.extrapolation!r},"
            f"\ninterpolation={self.interpolation!r})"
        ).replace(
            '\n',
            '\n    ',
        )

    def __getitem__(
        self: T,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Return self[key]."""
        key = _check_array_key(self.data_matrix, key)

        return self.copy(
            data_matrix=self.data_matrix[key],
            sample_names=list(np.array(self.sample_names)[key]),
        )

    #####################################################################
    # Numpy methods
    #####################################################################

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:

        for i in inputs:
            if (
                isinstance(i, FDataGrid)
                and not np.array_equal(i.grid_points, self.grid_points)
            ):
                return NotImplemented

        new_inputs = [
            i.data_matrix if isinstance(i, FDataGrid)
            else self._get_op_matrix(i) for i in inputs
        ]

        outputs = kwargs.pop('out', None)
        if outputs:
            new_outputs = [
                o.data_matrix if isinstance(o, FDataGrid)
                else o for o in outputs
            ]
            kwargs['out'] = tuple(new_outputs)
        else:
            new_outputs = (None,) * ufunc.nout

        results = getattr(ufunc, method)(*new_inputs, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (result if output is None else output)
            for result, output in zip(results, new_outputs)
        )

        results = [self.copy(data_matrix=r) for r in results]

        return results[0] if len(results) == 1 else results

    #####################################################################
    # Pandas ExtensionArray methods
    #####################################################################

    def _take_allow_fill(
        self: T,
        indices: NDArrayInt,
        fill_value: T,
    ) -> T:
        result = self.copy()
        result.data_matrix = np.full(
            (len(indices),) + self.data_matrix.shape[1:],
            np.nan,
        )

        positive_mask = indices >= 0
        result.data_matrix[positive_mask] = self.data_matrix[
            indices[positive_mask]
        ]

        if fill_value is not self.dtype.na_value:
            result.data_matrix[~positive_mask] = fill_value.data_matrix[0]

        return result

    @property
    def dtype(self) -> FDataGridDType:
        """The dtype for this extension array, FDataGridDType"""
        return FDataGridDType(
            grid_points=self.grid_points,
            domain_range=self.domain_range,
            dim_codomain=self.dim_codomain,
        )

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self.data_matrix.nbytes + sum(
            p.nbytes for p in self.grid_points
        )

    def isna(self) -> NDArrayBool:
        """
        Return a 1-D array indicating if each value is missing.

        Returns:
            na_values: Positions of NA.
        """
        return np.all(  # type: ignore[no-any-return]
            np.isnan(self.data_matrix),
            axis=tuple(range(1, self.data_matrix.ndim)),
        )


class FDataGridDType(
    pandas.api.extensions.ExtensionDtype,  # type: ignore[misc]
):
    """DType corresponding to FDataGrid in Pandas."""

    name = 'FDataGrid'
    kind = 'O'
    type = FDataGrid  # noqa: WPS125
    na_value = pandas.NA

    def __init__(
        self,
        grid_points: GridPointsLike,
        dim_codomain: int,
        domain_range: Optional[DomainRangeLike] = None,
    ) -> None:
        from ..misc.validation import validate_domain_range

        grid_points = _to_grid_points(grid_points)

        self.grid_points = tuple(tuple(s) for s in grid_points)

        if domain_range is None:
            domain_range = tuple((s[0], s[-1]) for s in self.grid_points)

        self.domain_range = validate_domain_range(domain_range)
        self.dim_codomain = dim_codomain

    @classmethod
    def construct_array_type(cls) -> Type[FDataGrid]:  # noqa: D102
        return FDataGrid

    def _na_repr(self) -> FDataGrid:

        shape = (
            (1,)
            + tuple(len(s) for s in self.grid_points)
            + (self.dim_codomain,)
        )

        data_matrix = np.full(shape=shape, fill_value=np.NaN)

        return FDataGrid(
            grid_points=self.grid_points,
            domain_range=self.domain_range,
            data_matrix=data_matrix,
        )

    def __eq__(self, other: Any) -> bool:
        """
        Compare dtype equality.

        Rules for equality (similar to categorical):
        1) Any FData is equal to the string 'category'
        2) Any FData is equal to itself
        3) Otherwise, they are equal if the arguments are equal.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True

        return (
            isinstance(other, FDataGridDType)
            and self.dim_codomain == other.dim_codomain
            and self.domain_range == other.domain_range
            and self.grid_points == other.grid_points
        )

    def __hash__(self) -> int:
        return hash((self.grid_points, self.domain_range, self.dim_codomain))


class _CoordinateIterator(Sequence[T]):
    """Internal class to iterate through the image coordinates."""

    def __init__(self, fdatagrid: T) -> None:
        """Create an iterator through the image coordinates."""
        self._fdatagrid = fdatagrid

    def __getitem__(
        self,
        key: Union[int, slice, NDArrayInt, NDArrayBool],
    ) -> T:
        """Get a specific coordinate."""
        s_key = key
        if isinstance(s_key, int):
            s_key = slice(s_key, s_key + 1)

        coordinate_names = np.array(self._fdatagrid.coordinate_names)[s_key]

        return self._fdatagrid.copy(
            data_matrix=self._fdatagrid.data_matrix[..., key],
            coordinate_names=tuple(coordinate_names),
        )

    def __len__(self) -> int:
        """Return the number of coordinates."""
        return self._fdatagrid.dim_codomain

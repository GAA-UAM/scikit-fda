"""
Module to interpolate functional data objects.
"""


import abc

from scipy.interpolate import (PchipInterpolator, UnivariateSpline,
                               RectBivariateSpline, RegularGridInterpolator)

import numpy as np

from .._utils import _to_array_maybe_ragged
from .evaluator import Evaluator


class _SplineList(abc.ABC):
    r"""ABC for list of interpolations."""

    def __init__(self, fdatagrid,
                 interpolation_order=1,
                 smoothness_parameter=0.):

        super().__init__()

        self.fdatagrid = fdatagrid
        self.interpolation_order = interpolation_order
        self.smoothness_parameter = smoothness_parameter

    @abc.abstractmethod
    def _evaluate_one(self, spl, t, derivative=0):
        """Evaluates one spline of the list."""
        pass

    def _evaluate_codomain(self, spl_m, t, derivative=0):
        """Evaluator of multidimensional sample"""
        return np.array([self._evaluate_one(spl, t, derivative)
                         for spl in spl_m]).T

    def evaluate(self, fdata, eval_points, *, derivative=0, aligned=True):

        if aligned:
            # Points evaluated inside the domain
            res = np.apply_along_axis(
                self._evaluate_codomain, 1,
                self.splines, eval_points, derivative)
            res = res.reshape(fdata.n_samples, eval_points.shape[0],
                              fdata.dim_codomain)

        else:
            res = _to_array_maybe_ragged([self._evaluate_codomain(
                s, e, derivative=derivative)
                for s, e in zip(self.splines, eval_points)])

        return res


class _SplineList1D(_SplineList):
    r"""List of interpolations for curves.

    List of interpolations for objects with domain
    dimension = 1. Calling internally during the creation of the
    evaluator.

    Uses internally the scipy interpolation UnivariateSpline or
    PchipInterpolator.

    Args:
        fdatagrid (FDatagrid): Fdatagrid to interpolate.
        interpolation_order (int, optional): Order of the interpolation, 1
            for linear interpolation, 2 for cuadratic, 3 for cubic and so
            on. In case of curves and surfaces there is available
            interpolation up to degree 5. For higher dimensional objects
            only linear or nearest interpolation is available. Default
            lineal interpolation.
        smoothness_parameter (float, optional): Penalisation to perform
            smoothness interpolation. Option only available for curves and
            surfaces. If 0 the residuals of the interpolation will be 0.
            Defaults 0.
        monotone (boolean, optional): Performs monotone interpolation in
            curves using a PCHIP interpolator. Only valid for curves (domain
            dimension equal to 1) and interpolation order equal to 1 or 3.
            Defaults false.

    Returns:
        (np.ndarray): Array of size n_samples x dim_codomain with the
        corresponding interpolation of the sample i, and image dimension j
        in the entry (i,j) of the array.

    Raises:
        ValueError: If the value of the interpolation k is not valid.

    """

    def __init__(self, fdatagrid,
                 interpolation_order=1,
                 smoothness_parameter=0.,
                 monotone=False):

        super().__init__(
            fdatagrid=fdatagrid,
            interpolation_order=interpolation_order,
            smoothness_parameter=smoothness_parameter)

        self.monotone = monotone

        if self.interpolation_order > 5 or self.interpolation_order < 1:
            raise ValueError(f"Invalid degree of interpolation "
                             f"({self.interpolation_order}). Must be "
                             f"an integer greater than 0 and lower or "
                             f"equal than 5.")

        if self.monotone and self.smoothness_parameter != 0:
            raise ValueError("Smoothing interpolation is not supported with "
                             "monotone interpolation")

        if self.monotone and (self.interpolation_order == 2
                              or self.interpolation_order == 4):
            raise ValueError(f"monotone interpolation of degree "
                             f"{self.interpolation_order}"
                             f"not supported.")

        # Monotone interpolation of degree 1 is performed with linear spline
        monotone = self.monotone
        if self.monotone and self.interpolation_order == 1:
            monotone = False

        grid_points = fdatagrid.grid_points[0]

        if monotone:
            def constructor(data):
                """Constructs an unidimensional cubic monotone interpolation"""
                return PchipInterpolator(grid_points, data)

        else:

            def constructor(data):
                """Constructs an unidimensional interpolation"""
                return UnivariateSpline(
                    grid_points, data,
                    s=self.smoothness_parameter,
                    k=self.interpolation_order)

        self.splines = np.apply_along_axis(
            constructor, 1, fdatagrid.data_matrix)

    def _evaluate_one(self, spl, t, derivative=0):
        try:
            return spl(t, derivative)[:, 0]
        except ValueError:
            return np.zeros_like(t)


class _SplineList2D(_SplineList):
    r"""List of interpolations for surfaces.

    List of interpolations for objects with domain
    dimension = 2. Calling internally during the creationg of the
    evaluator.

    Uses internally the scipy interpolation RectBivariateSpline.

    Args:
        fdatagrid (FDatagrid): Fdatagrid to interpolate.
        interpolation_order (int, optional): Order of the interpolation, 1
            for linear interpolation, 2 for cuadratic, 3 for cubic and so
            on. In case of curves and surfaces there is available
            interpolation up to degree 5. For higher dimensional objects
            only linear or nearest interpolation is available. Default
            lineal interpolation.
        smoothness_parameter (float, optional): Penalisation to perform
            smoothness interpolation. Option only available for curves and
            surfaces. If 0 the residuals of the interpolation will be 0.
            Defaults 0.
        monotone (boolean, optional): Performs monotone interpolation in
            curves using a PCHIP interpolator. Only valid for curves (domain
            dimension equal to 1) and interpolation order equal to 1 or 3.
            Defaults false.

    Returns:
        (np.ndarray): Array of size n_samples x dim_codomain with the
        corresponding interpolation of the sample i, and image dimension j
        in the entry (i,j) of the array.

    Raises:
        ValueError: If the value of the interpolation k is not valid.

    """

    def __init__(self, fdatagrid,
                 interpolation_order=1,
                 smoothness_parameter=0.):

        super().__init__(
            fdatagrid=fdatagrid,
            interpolation_order=interpolation_order,
            smoothness_parameter=smoothness_parameter)

        if np.isscalar(self.interpolation_order):
            kx = ky = self.interpolation_order
        elif len(self.interpolation_order) != 2:
            raise ValueError("k should be numeric or a tuple of length 2.")
        else:
            kx = self.interpolation_order[0]
            ky = self.interpolation_order[1]

        if kx > 5 or kx <= 0 or ky > 5 or ky <= 0:
            raise ValueError(f"Invalid degree of interpolation ({kx},{ky}). "
                             f"Must be an integer greater than 0 and lower or "
                             f"equal than 5.")

        # Matrix of splines
        self.splines = np.empty(
            (fdatagrid.n_samples, fdatagrid.dim_codomain), dtype=object)

        for i in range(fdatagrid.n_samples):
            for j in range(fdatagrid.dim_codomain):
                self.splines[i, j] = RectBivariateSpline(
                    fdatagrid.grid_points[0],
                    fdatagrid.grid_points[1],
                    fdatagrid.data_matrix[i, :, :, j],
                    kx=kx, ky=ky,
                    s=self.smoothness_parameter)

    def _evaluate_one(self, spl, t, derivative=0):
        if np.isscalar(derivative):
            derivative = 2 * [derivative]
        elif len(derivative) != 2:
            raise ValueError("derivative should be a numeric value "
                             "or a tuple of length 2 with (dx,dy).")

        return spl(t[:, 0], t[:, 1], dx=derivative[0], dy=derivative[1],
                   grid=False)


class _SplineListND(_SplineList):
    r"""List of interpolations.

    List of interpolations for objects with domain
    dimension > 2. Calling internally during the creationg of the
    evaluator.

    Only linear and nearest interpolations are available for objects with
    domain dimension >= 3. Uses internally the scipy interpolation
    RegularGridInterpolator.

    Args:
        grid_points (np.ndarray): Sample points of the fdatagrid.
        data_matrix (np.ndarray): Data matrix of the fdatagrid.
        k (integer): Order of the spline interpolations.

    Returns:
        (np.ndarray): Array of size n_samples x dim_codomain with the
        corresponding interpolation of the sample i, and image dimension j
        in the entry (i,j) of the array.

    Raises:
        ValueError: If the value of the interpolation k is not valid.

    """

    def __init__(self, fdatagrid,
                 interpolation_order=1,
                 smoothness_parameter=0.):

        super().__init__(
            fdatagrid=fdatagrid,
            interpolation_order=interpolation_order,
            smoothness_parameter=smoothness_parameter)

        if self.smoothness_parameter != 0:
            raise ValueError("Smoothing interpolation is only supported with "
                             "domain dimension up to 2, s should be 0.")

        # Parses method of interpolation
        if self.interpolation_order == 0:
            method = 'nearest'
        elif self.interpolation_order == 1:
            method = 'linear'
        else:
            raise ValueError("interpolation order should be 0 (nearest) or 1 "
                             "(linear).")

        self.splines = np.empty(
            (fdatagrid.n_samples, fdatagrid.dim_codomain), dtype=object)

        for i in range(fdatagrid.n_samples):
            for j in range(fdatagrid.dim_codomain):
                self.splines[i, j] = RegularGridInterpolator(
                    fdatagrid.grid_points, fdatagrid.data_matrix[i, ..., j],
                    method, False)

    def _evaluate_one(self, spl, t, derivative=0):

        if derivative != 0:
            raise ValueError("derivates not suported for functional data "
                             " with domain dimension greater than 2.")

        return spl(t)


class SplineInterpolation(Evaluator):
    r"""Spline interpolation of :class:`FDataGrid`.

    Spline interpolation of discretized functional objects. Implements
    different interpolation methods based in splines, using the sample
    points of the grid as nodes to interpolate.

    See the interpolation example to a detailled explanation.

    Attributes:
        interpolation_order (int, optional): Order of the interpolation, 1
            for linear interpolation, 2 for cuadratic, 3 for cubic and so
            on. In case of curves and surfaces there is available
            interpolation up to degree 5. For higher dimensional objects
            only linear or nearest interpolation is available. Default
            lineal interpolation.
        smoothness_parameter (float, optional): Penalisation to perform
            smoothness interpolation. Option only available for curves and
            surfaces. If 0 the residuals of the interpolation will be 0.
            Defaults 0.
        monotone (boolean, optional): Performs monotone interpolation in
            curves using a PCHIP interpolator. Only valid for curves (domain
            dimension equal to 1) and interpolation order equal to 1 or 3.
            Defaults false.

    """

    def __init__(self, interpolation_order=1, *, smoothness_parameter=0.,
                 monotone=False):
        r"""Constructor of the SplineInterpolation.

        Args:
            interpolation_order (int, optional): Order of the interpolation, 1
                for linear interpolation, 2 for cuadratic, 3 for cubic and so
                on. In case of curves and surfaces there is available
                interpolation up to degree 5. For higher dimensional objects
                only linear or nearest interpolation is available. Default
                lineal interpolation.
            smoothness_parameter (float, optional): Penalisation to perform
                smoothness interpolation. Option only available for curves and
                surfaces. If 0 the residuals of the interpolation will be 0.
                Defaults 0.
            monotone (boolean, optional): Performs monotone interpolation in
                curves using a PCHIP interpolation. Only valid for curves
                (domain dimension equal to 1) and interpolation order equal
                to 1 or 3.
                Defaults false.

        """
        self._interpolation_order = interpolation_order
        self._smoothness_parameter = smoothness_parameter
        self._monotone = monotone

    @property
    def interpolation_order(self):
        "Returns the interpolation order"
        return self._interpolation_order

    @property
    def smoothness_parameter(self):
        "Returns the smoothness parameter"
        return self._smoothness_parameter

    @property
    def monotone(self):
        "Returns flag to perform monotone interpolation"
        return self._monotone

    def _build_interpolator(self, fdatagrid):

        if fdatagrid.dim_domain == 1:
            return _SplineList1D(
                fdatagrid=fdatagrid,
                interpolation_order=self.interpolation_order,
                smoothness_parameter=self.smoothness_parameter,
                monotone=self.monotone)

        elif self.monotone:
            raise ValueError("Monotone interpolation is only supported with "
                             "domain dimension equal to 1.")

        elif fdatagrid.dim_domain == 2:
            return _SplineList2D(
                fdatagrid=fdatagrid,
                interpolation_order=self.interpolation_order,
                smoothness_parameter=self.smoothness_parameter)

        else:
            return _SplineListND(
                fdatagrid=fdatagrid,
                interpolation_order=self.interpolation_order,
                smoothness_parameter=self.smoothness_parameter)

    def evaluate(self, fdata, eval_points, *, aligned=True):

        spline_list = self._build_interpolator(fdata)

        return spline_list.evaluate(fdata, eval_points, aligned=aligned)

    def __repr__(self):
        """repr method of the interpolation"""
        return (f"{type(self).__name__}("
                f"interpolation_order={self.interpolation_order}, "
                f"smoothness_parameter={self.smoothness_parameter}, "
                f"monotone={self.monotone})")

    def __eq__(self, other):
        """Equality operator between SplineInterpolation"""
        return (super().__eq__(other) and
                self.interpolation_order == other.interpolation_order and
                self.smoothness_parameter == other.smoothness_parameter and
                self.monotone == other.monotone)

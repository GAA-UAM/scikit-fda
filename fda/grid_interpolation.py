"""
Module to interpolate FDataGrid


"""

from abc import ABC, abstractmethod

import numbers
import numpy

from .functional_data import _list_of_arrays
from .extrapolation import _extrapolation_index

from scipy.interpolate import (PchipInterpolator, UnivariateSpline,
                               RectBivariateSpline, RegularGridInterpolator)


class GridInterpolator(ABC):

    @abstractmethod
    def _construct_interpolator(self, fdatagrid):
        pass


class _GridInterpolatorEvaluator(ABC):

    @abstractmethod
    def evaluate(self, eval_points, derivative=0, extrapolation=None,
                 grid=False, keepdims=None):
        pass

    def __call__(self, eval_points, derivative=0, extrapolation=None,
                 grid=False, keepdims=None):

        return self.evaluate(eval_points, derivative=derivative,
                             extrapolation=extrapolation,grid=grid,
                             keepdims=keepdims)


class GridSplineInterpolator(GridInterpolator):

    def __init__(self, interpolation_order=1, smoothness_parameter=0.,
                 monotone=False):

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


    def _construct_interpolator(self, fdatagrid):

        return _GridSplineInterpolatorEvaluator(fdatagrid,
                                                self.interpolation_order,
                                                self.smoothness_parameter,
                                                self.monotone)

    def __repr__(self):
        return (f"{type(self).__name__}("
                f"interpolation_order={self.interpolation_order}, "
                f"smoothness_parameter={self.smoothness_parameter}, "
                f"monotone={self.monotone})")


class _GridSplineInterpolatorEvaluator(_GridInterpolatorEvaluator):

    def __init__(self, fdatagrid, k=1, s=0., monotone=False):

        sample_points = fdatagrid.sample_points
        data_matrix = fdatagrid.data_matrix

        self._fdatagrid = fdatagrid
        self._ndim_image = fdatagrid.ndim_image
        self._ndim_domain = fdatagrid.ndim_domain
        self._nsamples = fdatagrid.nsamples
        self._keepdims = fdatagrid.keepdims
        self._domain_range = fdatagrid.domain_range

        if self._ndim_domain == 1:
            self._splines = self._construct_spline_1_m(sample_points,
                                                       data_matrix,
                                                       k, s, monotone)
        elif monotone:
            raise ValueError("Monotone interpolation is only supported with "
                             "domain dimension equal to 1.")

        elif self._ndim_domain == 2:
            self._splines = self._construct_spline_2_m(sample_points,
                                                       data_matrix, k, s)

        elif s != 0:
            raise ValueError("Smoothing interpolation is only supported with "
                             "domain dimension up to 2, s should be 0.")

        else:
            self._splines = self._construct_spline_n_m(sample_points,
                                                       data_matrix, k)

    def _construct_spline_1_m(self, sample_points, data_matrix, k, s, monotone):

        if k > 5 or k < 1:
            raise ValueError(f"Invalid degree of interpolation ({k}). Must be "
                             f"an integer greater than 0 and lower or "
                             f"equal than 5.")

        if monotone and s != 0:
            raise ValueError("Smoothing interpolation is not supported with "
                             "monotone interpolation")

        if monotone and (k == 2 or k == 4):
            raise ValueError(f"monotone interpolation of degree {k}"
                             f"not supported.")

        # Monotone interpolation of degree 1 is performed with linear spline
        if monotone and k == 1:
            monotone = False

        # Evaluator of splines called in evaluate

        def _spline_evaluator_1_m(spl, t, der):

            return spl(t, der)

        def _process_derivative_1_m(derivative):

            return derivative

        self._spline_evaluator = _spline_evaluator_1_m

        self._process_derivative = _process_derivative_1_m

        sample_points = sample_points[0]

        if monotone:
            constructor =  lambda data: PchipInterpolator(sample_points, data)

        else:
            constructor = lambda data: UnivariateSpline(sample_points, data,
                                                        s=s, k=k)

        return numpy.apply_along_axis(constructor, 1, data_matrix)

    def _construct_spline_2_m(self, sample_points, data_matrix, k, s):


        if numpy.isscalar(k):
            kx = ky = k
        elif len(k) != 2:
            raise ValueError("k should be numeric or a tuple of length 2.")
        else:
            kx = k[0]
            ky = k[1]

        if kx > 5 or kx <= 0 or ky > 5 or ky <= 0:
            raise ValueError(f"Invalid degree of interpolation ({kx},{ky}). "
                             f"Must be an integer greater than 0 and lower or "
                             f"equal than 5.")

        def _spline_evaluator_2_m(spl, t, der):

            return spl(t[:,0], t[:,1], dx=der[0], dy=der[1], grid=False)

        def _process_derivative_2_m(derivative):
            if numpy.isscalar(derivative):
                derivative = 2*[derivative]
            elif len(derivative) != 2:
                raise ValueError("derivative should be a numeric value "
                                 "or a tuple of length 2 with (dx,dy).")

            return derivative

        # Evaluator of splines called in evaluate
        self._spline_evaluator = _spline_evaluator_2_m
        self._process_derivative = _process_derivative_2_m

        # Matrix of splines
        spline = numpy.empty((self._nsamples, self._ndim_image), dtype=object)

        for i in range(self._nsamples):
            for j in range(self._ndim_image):
                spline[i,j] = RectBivariateSpline(sample_points[0],
                                                  sample_points[1],
                                                  data_matrix[i,:,:,j],
                                                  kx=kx, ky=ky, s=s)

        return spline

    def _construct_spline_n_m(self, sample_points, data_matrix, k):

        # Parses method of interpolation
        if k == 0:
            method = 'nearest'
        elif k == 1:
            method = 'linear'
        else:
            raise ValueError("interpolation order should be 0 (nearest) or 1 "
                             "(linear).")

        def _process_derivative_n_m(derivative):
            if derivative != 0:
                raise ValueError("derivates not suported for functional data "
                                 " with domain dimension greater than 2.")

            return derivative

        def _spline_evaluator_n_m(spl, t, derivative):

            return spl(t)

        # Method to process derivative argument
        self._process_derivative = _process_derivative_n_m

        # Evaluator of splines called in evaluate
        self._spline_evaluator = _spline_evaluator_n_m

        spline = numpy.empty((self._nsamples, self._ndim_image), dtype=object)

        for i in range(self._nsamples):
            for j in range(self._ndim_image):
                spline[i,j] = RegularGridInterpolator(
                    sample_points, data_matrix[i,...,j], method, False)

        return spline

    #def evaluate(self, t, derivative=0, grid=False):
    def evaluate(self, eval_points, derivative=0, extrapolation=None,
                 grid=False, keepdims=None):

        derivative = self._process_derivative(derivative)

        # Case evaluation of a numeric value
        if isinstance(eval_points, numbers.Number):
            eval_points = [eval_points]

        eval_points = numpy.array(eval_points, dtype=numpy.float)

        if grid:
            return self.evaluate_grid(eval_points, derivative,
                                      extrapolation=extrapolation,
                                      keepdims=keepdims)

        if ((self._ndim_domain == 1 and len(eval_points.shape) == 2) or
            len(eval_points.shape) == 3):

            return self.evaluate_composed(eval_points, derivative,
                                          extrapolation=extrapolation,
                                          keepdims=keepdims)

        return self.evaluate_spline(eval_points, derivative,
                                    extrapolation=extrapolation,
                                    keepdims=keepdims)

    def evaluate_spline(self, t, derivative=0, extrapolation=None, keepdims=None):

        if extrapolation is not None:
            # Coordinates with shape (n_evalpoints x ndim_image)
            eval_points_coord = t.reshape((len(t), self._ndim_domain))


            # Boolean index of points where the extrapolation should be applied
            index_ext = _extrapolation_index(eval_points_coord, self._domain_range)

            # Flag to apply extrapolation
            extrapolate = index_ext.any()

        else:
            extrapolate = False


        if extrapolate: # Index of points with extrapolation
            index = ~ index_ext
            t_eval = t[index]

        else: # Else all points will be evaluated
            t_eval = t

        # Constructs the evaluator for t_eval
        if self._ndim_image == 1:
            evaluator = lambda spl: self._spline_evaluator(spl[0], t_eval,
                                                           derivative)
        else:
            evaluator = lambda spl_m: numpy.dstack(
                self._spline_evaluator(spl, t_eval, derivative) for spl in spl_m
                ).flatten()

        # Points evaluated inside the domain
        res_eval = numpy.apply_along_axis(evaluator, 1, self._splines)
        res_eval = res_eval.reshape(self._nsamples, t_eval.shape[0], self._ndim_image)

        if extrapolate: # Points evaluated with extrapolation

            res_ext = extrapolation(self._fdatagrid,
                                    eval_points_coord[index_ext, ...],
                                    derivative = derivative,
                                    keepdims=True)

            res = numpy.empty((self._nsamples, t.shape[0], self._ndim_image))
            res[:,index_ext,:] = res_ext
            res[:, index,:] = res_eval

        else:
            res = res_eval

        if keepdims is None:
            keepdims = self._keepdims

        if self._ndim_image == 1 and not keepdims:
            res = res.reshape(self._nsamples, t.shape[0])

        return res

    def evaluate_grid(self, axes, derivative=0, extrapolation=None,
                      keepdims=None):

        axes = _list_of_arrays(axes)
        lengths = [len(ax) for ax in axes]

        if len(axes) != self._ndim_domain:
            raise ValueError(f"Length of axes should be {self._ndim_domain}")

        t = numpy.meshgrid(*axes, indexing='ij')
        t = numpy.array(t).reshape(self._ndim_domain, numpy.prod(lengths)).T


        res = self.evaluate_spline(t, derivative, extrapolation=extrapolation,
                                   keepdims=keepdims)


        shape = [self._nsamples] + lengths

        if keepdims is None:
            keepdims = self._keepdims

        if self._ndim_image != 1 or keepdims:
            shape += [self._ndim_image]

        return res.reshape(shape)

    def evaluate_composed(self, t, derivative=0, extrapolation=None,
                          keepdims=None):


        if t.shape[0] != self._nsamples:
            raise ValueError(f"First dimension of eval_points should have the "
                             f"same length than 'nsamples' ({t.shape[0]}) !="
                             f"({self._nsamples}).")

        # Applies extrapolation in points outside the domain range
        if extrapolation is not None:

            # Coordinates with shape (nsamples x n_evalpoints x ndim_image)
            eval_points_coord = t.reshape((t.shape[0], t.shape[1], self._ndim_domain))

            # Boolean index of points where the extrapolation should be applied
            index_ext = _extrapolation_index(eval_points_coord, self._domain_range)
            index_ext = numpy.logical_or.reduce(index_ext, axis=0)

            # Flag to apply extrapolation
            extrapolate = index_ext.any()

        else:
            extrapolate = False

        if extrapolate: # Index of points without extrapolation

            index = ~ index_ext

            if t.ndim == 3:
                t_eval = t[:, index, :]
            else:
                t_eval = t[:, index]

        else: # Else all points will be evaluated
            t_eval = t


        if self._ndim_image == 1:
            shape = (self._nsamples, t_eval.shape[1])


            evaluator = lambda t, spl: self._spline_evaluator(
                spl[0], t, derivative)

        else:
            shape = (self._nsamples, t_eval.shape[1], self._ndim_image)

            evaluator = lambda t, spl_m: numpy.array(
                [self._spline_evaluator(spl, t, derivative) for spl in spl_m]).T

        res_eval = numpy.empty(shape)

        # Evaluation of points without extrapolation
        for i in range(self._nsamples):

            res_eval[i] = evaluator(t_eval[i], self._splines[i])

        if extrapolate:

            res_ext = extrapolation(self._fdatagrid,
                                    eval_points_coord[:, index_ext, ...],
                                    derivative = derivative,
                                    keepdims=False)

            if self._ndim_image == 1:
                res = numpy.empty((self._nsamples, t.shape[1]))

                res[:, index] = res_eval
                res[:, index_ext] = res_ext
            else:
                res = numpy.empty((self._nsamples, t.shape[1], self._ndim_image))
                res[:, index, :] = res_eval
                res[:, index_ext, :] = res_ext

        else:
            res = res_eval

        if keepdims is None:
            keepdims = self._keepdims

        if self._ndim_image == 1 and keepdims:
            res = res.reshape(self._nsamples, t.shape[1], self._ndim_image)


        return res

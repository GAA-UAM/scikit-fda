"""
Module to interpolate FDataGrid


"""

from abc import ABCMeta, abstractmethod

import numbers
import numpy

from .functional_data import _list_of_arrays

from scipy.interpolate import (PchipInterpolator, UnivariateSpline,
                               RectBivariateSpline, RegularGridInterpolator)


class GridInterpolator:

    __metaclass__ = ABCMeta

    @abstractmethod
    def _construct_interpolator(self, fdatagrid):
        pass


class _GridInterpolatorEvaluator:

    __metaclass__ = ABCMeta

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

        self.interpolation_order = interpolation_order
        self.smoothness_parameter = smoothness_parameter
        self.monotone = monotone

    def _construct_interpolator(self, fdatagrid):

        return _GridSplineInterpolatorEvaluator(fdatagrid.sample_points,
                                                fdatagrid.data_matrix,
                                                fdatagrid.ndim_domain,
                                                fdatagrid.ndim_image,
                                                self.interpolation_order,
                                                self.smoothness_parameter,
                                                self.monotone,
                                                fdatagrid.keepdims)


class _GridSplineInterpolatorEvaluator(_GridInterpolatorEvaluator):

    def __init__(self, sample_points, data_matrix, ndim_domain, ndim_image, k=1,
                 s=0., monotone=False, keepdims=False):

        self._ndim_image = ndim_image
        self._ndim_domain = ndim_domain
        self._nsamples = data_matrix.shape[0]
        self._keepdims = keepdims

        if ndim_domain == 1:
            self._splines = self._construct_spline_1_m(sample_points,
                                                       data_matrix,
                                                       k, s, monotone)
        elif monotone:
            raise ValueError("Monotone interpolation is only supported with "
                             "domain dimension equal to 1.")

        elif ndim_domain == 2:
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

        if extrapolation is not None:
            NotImplementedError("Extrapolation to do implemented.")

        derivative = self._process_derivative(derivative)

        # Case evaluation of a numeric value
        if isinstance(eval_points, numbers.Number):
            eval_points = [eval_points]

        eval_points = numpy.asarray(eval_points)

        if grid:
            return self.evaluate_grid(eval_points, derivative,
                                      extrapolation=extrapolation,
                                      keepdims=keepdims)

        if ((self._ndim_image == 1 and len(eval_points.shape) == 2) or
            len(eval_points.shape) == 3):

            return self.evaluate_composed(eval_points, derivative,
                                          extrapolation=extrapolation,
                                          keepdims=keepdims)

        return self.evaluate_spline(eval_points, derivative,
                                    extrapolation=extrapolation,
                                    keepdims=keepdims)

    def evaluate_spline(self, t, derivative=0, extrapolation=None,
                        keepdims=None):

        if extrapolation is not None:
            raise NotImplementedError("Extrapolation to do implemented.")

        if self._ndim_image == 1:
            evaluator = lambda spl: self._spline_evaluator(spl[0], t,
                                                           derivative)

        else:
            evaluator = lambda spl_m: numpy.dstack(
                self._spline_evaluator(spl, t, derivative) for spl in spl_m
                ).flatten()

        res = numpy.apply_along_axis(evaluator, 1, self._splines)

        if keepdims is None:
            keepdims = self._keepdims

        if self._ndim_image != 1 or keepdims:
            res = res.reshape(self._nsamples, t.shape[0], self._ndim_image)

        return res

    def evaluate_grid(self, axes, derivative=0, extrapolation=None,
                      keepdims=None):

        if extrapolation is not None:
            raise NotImplementedError("Extrapolation to do implemented.")

        axes = _list_of_arrays(axes)
        lengths = [len(ax) for ax in axes]

        if len(axes) != self._ndim_domain:
            raise ValueError(f"Length of axes should be {self._ndim_domain}")

        t = numpy.meshgrid(*axes, indexing='ij')
        t = numpy.array(t).reshape(self._ndim_domain, numpy.prod(lengths)).T


        res = self.evaluate_spline(t, derivative)


        shape = [self._nsamples] + lengths

        if keepdims is None:
            keepdims = self._keepdims

        if self._ndim_image != 1 or keepdims:
            shape += [self._ndim_image]

        return res.reshape(shape)

    def evaluate_composed(self, t, derivative=0, extrapolation=None,
                          keepdims=None):

        if extrapolation is not None:
            raise NotImplementedError("Extrapolation to do implemented.")

        if t.shape[0] != self._nsamples:
            raise ValueError("t should be a list of length 'nsamples'.")

        if self._ndim_image == 1:
            shape = (self._nsamples, t.shape[1])


            evaluator = lambda t, spl: self._spline_evaluator(
                spl[0], t, derivative)

        else:
            shape = (self._nsamples, t.shape[1], self._ndim_image)

            evaluator = lambda t, spl_m: numpy.array(
                [self._spline_evaluator(spl, t, derivative) for spl in spl_m])

        res = numpy.empty(shape)

        for i in range(self._nsamples):
            res[i] = evaluator(t[i], self._splines[i])

        if keepdims is None:
            keepdims = self._keepdims

        if self._ndim_image == 1 and keepdims:
            res = res.reshape(self._nsamples, t.shape[1], self._ndim_image)


        return res

    def evaluate_shifted(self, eval_points, delta, derivative=0):

        t = numpy.outer(numpy.ones(self._nsamples), eval_points)

        t += numpy.asarray(delta).T

        return self.evaluate_composed(t, self._process_derivative(derivative))

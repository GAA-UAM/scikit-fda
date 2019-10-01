"""Shift Registration of functional data module.

This module contains methods to perform the registration of
functional data using shifts, in basis as well in discretized form.
"""

import scipy.integrate

import numpy as np

from ..._utils import constants
from .base import RegistrationTransformer
from ... import FData, FDataGrid


__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"

class ShiftRegistration(RegistrationTransformer):

    def __init__(self, maxiter=5, tol=1e-2, restrict_domain=False,
                 template="mean", extrapolation=None, step_size=1,
                 initial=None, output_points=None, **kwargs):
        self.max_iter = maxiter
        self.tol = tol
        self.template = template
        self.restrict_domain = restrict_domain
        self.extrapolation = extrapolation
        self.step_size = step_size
        self.initial = initial
        self.output_points = output_points


    def _shift_registration_deltas(self, fd, template):
        r"""Return the lists of shifts used in the shift registration procedure.

            Realizes a registration of the curves, using shift aligment, as is
            defined in [RS05-7-2-1]_. Calculates :math:`\delta_{i}` for each sample
            such that :math:`x_i(t + \delta_{i})` minimizes the least squares
            criterion:

            .. math::
                \text{REGSSE} = \sum_{i=1}^{N} \int_{\mathcal{T}}
                [x_i(t + \delta_i) - \hat\mu(t)]^2 ds

            Estimates the shift parameter :math:`\delta_i` iteratively by
            using a modified Newton-Raphson algorithm, updating the mean
            in each iteration, as is described in detail in [RS05-7-9-1-1]_.

            Method only implemented for Funtional objects with domain and image
            dimension equal to 1.

        Args:
            fd (:class:`FData`): Functional data object to be registered.
            maxiter (int, optional): Maximun number of iterations.
                Defaults to 5.
            tol (float, optional): Tolerance allowable. The process will stop if
                :math:`\max_{i}|\delta_{i}^{(\nu)}-\delta_{i}^{(\nu-1)}|<tol`.
                Default sets to 1e-2.
            restrict_domain (bool, optional): If True restricts the domain to
                avoid evaluate points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation (str or :class:`Extrapolation`, optional): Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See :module:
                `extrapolation` to obtain more information.
            step_size (int or float, optional): Parameter to adjust the rate of
                convergence in the Newton-Raphson algorithm, see [RS05-7-9-1-1]_.
                Defaults to 1.
            initial (array_like, optional): Initial estimation of shifts.
                Default uses a list of zeros for the initial shifts.
            output_points (array_like, optional): Set of points where the
                functions are evaluated to obtain the discrete
                representation of the object to integrate. If None is
                passed it calls numpy.linspace in FDataBasis and uses the
                `sample_points` in FDataGrids.

        Returns:
            :class:`numpy.ndarray`: list with the shifts.

        Raises:
            ValueError: If the initial array has different length than the
                number of samples.

        Examples:

            >>> from skfda.datasets import make_sinusoidal_process
            >>> from skfda.representation.basis import Fourier
            >>> from skfda.preprocessing.registration import (
            ...      shift_registration_deltas)
            >>> fd = make_sinusoidal_process(n_samples=2, error_std=0,
            ...                              random_state=1)

            Registration of data in discretized form:

            >>> shift_registration_deltas(fd).round(3)
            array([-0.022,  0.03 ])

            Registration of data in basis form:

            >>> fd = fd.to_basis(Fourier())
            >>> shift_registration_deltas(fd).round(3)
            array([-0.022,  0.03 ])


        References:
            ..  [RS05-7-2-1] Ramsay, J., Silverman, B. W. (2005). Shift
                registration. In *Functional Data Analysis* (pp. 129-132).
                Springer.
            ..  [RS05-7-9-1-1] Ramsay, J., Silverman, B. W. (2005). Shift
                registration by the Newton-Raphson algorithm. In *Functional
                Data Analysis* (pp. 142-144). Springer.
        """

        # Initial estimation of the shifts

        if fd.dim_codomain > 1 or fd.dim_domain > 1:
            raise NotImplementedError("Method for unidimensional data.")

        domain_range = fd.domain_range[0]

        if self.initial is None:
            delta = np.zeros(fd.n_samples)

        elif len(self.initial) != fd.n_samples:
            raise ValueError(f"the initial shift ({len(self.initial)}) must have the "
                             f"same length than the number of samples "
                             f"({fd.n_samples})")
        else:
            delta = np.asarray(self.initial)

        # Fine equispaced mesh to evaluate the samples
        if self.output_points is None:

            try:
                output_points = fd.sample_points[0]
                nfine = len(output_points)
            except AttributeError:
                nfine = max(fd.n_basis * constants.BASIS_MIN_FACTOR + 1,
                            constants.N_POINTS_COARSE_MESH)
                output_points = np.linspace(*domain_range, nfine)

        else:
            nfine = len(self.output_points)
            output_points = np.asarray(self.output_points)

        # Auxiliar array to avoid multiple memory allocations
        delta_aux = np.empty(fd.n_samples)

        # Computes the derivate of originals curves in the mesh points
        D1x = fd.evaluate(output_points, derivative=1, keepdims=False)

        # Second term of the second derivate estimation of REGSSE. The
        # first term has been dropped to improve convergence (see references)
        d2_regsse = scipy.integrate.trapz(np.square(D1x), output_points,
                                          axis=1)

        max_diff = self.tol + 1
        self.n_iter_ = 0

        # Case template fixed
        if isinstance(template, FData):
            original_template = template
            tfine_aux = template.evaluate(output_points, keepdims=False)

            if self.restrict_domain:
                template_points_aux = tfine_aux

            template="fixed"
        else:
            tfine_aux = np.empty(nfine)

        # Auxiliar array if the domain will be restricted
        if self.restrict_domain:
            D1x_tmp = D1x
            tfine_tmp = output_points
            tfine_aux_tmp = tfine_aux
            domain = np.empty(nfine, dtype=np.dtype(bool))

        ones = np.ones(fd.n_samples)
        output_points_rep = np.outer(ones, output_points)

        # Newton-Rhapson iteration
        while max_diff > self.tol and self.n_iter_ < self.max_iter:

            # Updates the limits for non periodic functions ignoring the ends
            if self.restrict_domain:
                # Calculates the new limits
                a = domain_range[0] - min(np.min(delta), 0)
                b = domain_range[1] - max(np.max(delta), 0)

                # New interval is (a,b)
                np.logical_and(tfine_tmp >= a, tfine_tmp <= b, out=domain)
                output_points = tfine_tmp[domain]
                tfine_aux = tfine_aux_tmp[domain]
                D1x = D1x_tmp[:, domain]
                # Reescale the second derivate could be other approach
                # d2_regsse =
                #     d2_regsse_original * ( 1 + (a - b) / (domain[1] - domain[0]))
                d2_regsse = scipy.integrate.trapz(np.square(D1x),
                                                  output_points, axis=1)

                # Recompute base points for evaluation
                output_points_rep = np.outer(ones, output_points)

            # Computes the new values shifted
            x = fd.evaluate(output_points_rep + np.atleast_2d(delta).T,
                            aligned_evaluation=False,
                            extrapolation=self.extrapolation,
                            keepdims=False)

            if template == "mean":
                print("Updating mean")
                x.mean(axis=0, out=tfine_aux)
            elif template == "fixed" and self.restrict_domain:
                print("Restricting mean")
                tfine_aux = template_points_aux[domain]

            # Calculates x - mean
            np.subtract(x, tfine_aux, out=x)

            d1_regsse = scipy.integrate.trapz(np.multiply(x, D1x, out=x),
                                              output_points, axis=1)
            # Updates the shifts by the Newton-Rhapson iteration
            # delta = delta - step_size * d1_regsse / d2_regsse
            np.divide(d1_regsse, d2_regsse, out=delta_aux)
            np.multiply(delta_aux, self.step_size, out=delta_aux)
            np.subtract(delta, delta_aux, out=delta)

            # Updates convergence criterions
            max_diff = np.abs(delta_aux, out=delta_aux).max()
            self.n_iter_ += 1


        if template == "fixed":

            # Stores the original template instead of build it again
            template = original_template
        else:

            # Stores the template in an FDataGrid
            template = FDataGrid(tfine_aux, sample_points=output_points)

        return delta, template


    def fit_transform(self, X: FData, y=None):

        deltas, template = self._shift_registration_deltas(X, self.template)
        self.template_ = template
        self.deltas_ = deltas

        # Computes the values with the final shift to construct the FDataBasis
        return X.shift(deltas, restrict_domain=self.restrict_domain,
                       extrapolation=self.extrapolation,
                       eval_points=self.output_points)

    def fit(self, X: FData, y=None):

        deltas, template = self._shift_registration_deltas(X, self.template)

        self.template_ = template

        return self

    def transform(self, X: FData, y=None):

        deltas, template = self._shift_registration_deltas(X, self.template_)
        self.template_ = template
        self.deltas_ = deltas

        # Computes the values with the final shift to construct the FDataBasis
        return X.shift(deltas, restrict_domain=self.restrict_domain,
                       extrapolation=self.extrapolation,
                       eval_points=self.output_points)

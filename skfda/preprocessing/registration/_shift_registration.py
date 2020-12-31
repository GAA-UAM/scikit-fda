"""Class to apply Shift Registration to functional data"""

# Pablo Marcos Manchón
# pablo.marcosm@protonmail.com

import numpy as np
from scipy.integrate import simps
from sklearn.utils.validation import check_is_fitted

from ... import FData, FDataGrid
from ..._utils import check_is_univariate, constants
from .base import RegistrationTransformer


class ShiftRegistration(RegistrationTransformer):
    r"""Register a functional dataset using shift alignment.

    Realizes the registration of a set of curves using a shift aligment
    [RaSi2005-7-2]_. Let :math:`\{x_i(t)\}_{i=1}^{N}` be a functional dataset,
    calculates :math:`\delta_{i}` for each sample such that
    :math:`x_i(t + \delta_{i})` minimizes the least squares criterion:

    .. math::
        \text{REGSSE} = \sum_{i=1}^{N} \int_{\mathcal{T}}
        [x_i(t + \delta_i) - \hat\mu(t)]^2 ds

    Estimates each shift parameter :math:`\delta_i` iteratively by
    using a modified Newton-Raphson algorithm, updating the template
    :math:`\mu` in each iteration as is described in detail in
    [RaSi2005-7-9-1]_.

    Method only implemented for univariate functional data.

    Args:
        max_iter (int, optional): Maximun number of iterations.
            Defaults sets to 5. Generally 2 or 3 iterations are sufficient to
            obtain a good alignment.
        tol (float, optional): Tolerance allowable. The process will stop if
            :math:`\max_{i}|\delta_{i}^{(\nu)}-\delta_{i}^{(\nu-1)}|<tol`.
            Default sets to 1e-2.
        template (str, callable or FData, optional): Template to use in the
            least squares criterion. If template="mean" it is use the
            functional mean as in the original paper. The template can be a
            callable that will receive an FDataGrid with the samples and will
            return another FDataGrid as a template, such as any of the means or
            medians of the module `skfda.explotatory.stats`.
            If the template is an FData is used directly as the final
            template to the registration, if it is a callable or "mean" the
            template is computed iteratively constructing a temporal template
            in each iteration. In [RaSi2005-7-9-1]_ is described in detail this
            procedure. Defaults to "mean".
        extrapolation (str or :class:`Extrapolation`, optional): Controls the
            extrapolation mode for points outside the :term:`domain` range.
            By default uses the method defined in the data to be transformed.
            See the `extrapolation` documentation to obtain more information.
        step_size (int or float, optional): Parameter to adjust the rate of
            convergence in the Newton-Raphson algorithm, see [RaSi2005-7-9-1]_.
            Defaults to 1.
        restrict_domain (bool, optional): If True restricts the :term:`domain`
            to avoid the need of using extrapolation, in which
            case only the fit_transform method will be available, as training
            and transformation must be done together. Defaults to False.
        initial (str or array_like, optional): Array with an initial estimation
            of shifts. Default uses a list of zeros for the initial shifts.
        output_points (array_like, optional): Set of points where the
            functions are evaluated to obtain the discrete
            representation of the object to integrate. If None is
            passed it calls numpy.linspace in FDataBasis and uses the
            `grid_points` in FDataGrids.

    Attributes:
        template_ (FData): Template :math:`\mu` learned during the fitting
            used to the transformation.
        deltas_ (numpy.ndarray): List of shifts :math:`\delta_i` applied
            during the last transformation.
        n_iter_ (int): Number of iterations performed during the last
            transformation.

    Note:
        Due to the use of derivatives for the estimation of the shifts, the
        samples to be registered may be smooth for the correct convergence of
        the method.

    Examples:

        >>> from skfda.preprocessing.registration import ShiftRegistration
        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.representation.basis import Fourier


        Registration and creation of dataset in discretized form:

        >>> fd = make_sinusoidal_process(n_samples=10, error_std=0,
        ...                              random_state=1)
        >>> reg = ShiftRegistration(extrapolation="periodic")
        >>> fd_registered = reg.fit_transform(fd)
        >>> fd_registered
        FDataGrid(...)

        Shifts applied during the transformation

        >>> reg.deltas_.round(3)
        array([-0.128,  0.187,  0.027,  0.034, -0.106,  0.114, ..., -0.06 ])


        Registration and creation of a dataset in basis form using the
        transformation previosly fitted:

        >>> fd = make_sinusoidal_process(n_samples=2, error_std=0,
        ...                              random_state=2)
        >>> fd_basis = fd.to_basis(Fourier())
        >>> reg.transform(fd_basis)
        FDataBasis(...)


    References:
        ..  [RaSi2005-7-2] Ramsay, J., Silverman, B. W. (2005). Shift
            registration. In *Functional Data Analysis* (pp. 129-132).
            Springer.
        ..  [RaSi2005-7-9-1] Ramsay, J., Silverman, B. W. (2005). Shift
            registration by the Newton-Raphson algorithm. In *Functional
            Data Analysis* (pp. 142-144). Springer.
    """

    def __init__(self, max_iter=5, tol=1e-2, template="mean",
                 extrapolation=None, step_size=1, restrict_domain=False,
                 initial="zeros", output_points=None):
        self.max_iter = max_iter
        self.tol = tol
        self.template = template
        self.restrict_domain = restrict_domain
        self.extrapolation = extrapolation
        self.step_size = step_size
        self.initial = initial
        self.output_points = output_points

    def _compute_deltas(self, fd, template):
        r"""Compute the shifts to perform the registration.

        Args:
            fd (FData: Functional object to be registered.
            template (str, FData or callable): Template to align the
                the samples. "mean" to compute the mean iteratively as in
                the original paper, an FData with the templated calculated or
                a callable wich constructs the template.

        Returns:
            tuple: A tuple with an array of deltas and an FDataGrid with the
                template.

        """
        check_is_univariate(fd)

        domain_range = fd.domain_range[0]

        # Initial estimation of the shifts
        if self.initial == "zeros":
            delta = np.zeros(fd.n_samples)

        elif len(self.initial) != fd.n_samples:
            raise ValueError(f"the initial shift ({len(self.initial)}) must "
                             f"have the same length than the number of samples"
                             f" ({fd.n_samples})")
        else:
            delta = np.asarray(self.initial)

        # Fine equispaced mesh to evaluate the samples
        if self.output_points is None:

            try:
                output_points = fd.grid_points[0]
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
        fd_deriv = fd.derivative(order=1)
        D1x = fd_deriv(output_points)[..., 0]

        # Second term of the second derivate estimation of REGSSE. The
        # first term has been dropped to improve convergence (see references)
        d2_regsse = simps(np.square(D1x), output_points, axis=1)

        max_diff = self.tol + 1
        self.n_iter_ = 0

        # Case template fixed
        if isinstance(template, FData):
            original_template = template
            tfine_aux = template.evaluate(output_points)[0, ..., 0]

            if self.restrict_domain:
                template_points_aux = tfine_aux

            template = "fixed"
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
                # d2_regsse_original * ( 1 + (a - b) / (domain[1] - domain[0]))
                d2_regsse = simps(np.square(D1x), output_points, axis=1)

                # Recompute base points for evaluation
                output_points_rep = np.outer(ones, output_points)

            # Computes the new values shifted
            x = fd(output_points_rep + np.atleast_2d(delta).T,
                   aligned=False,
                   extrapolation=self.extrapolation)[..., 0]

            if template == "mean":
                x.mean(axis=0, out=tfine_aux)
            elif template == "fixed" and self.restrict_domain:
                tfine_aux = template_points_aux[domain]
            elif callable(template):  # Callable
                fd_x = FDataGrid(x, grid_points=output_points)
                fd_tfine = template(fd_x)
                tfine_aux = fd_tfine.data_matrix.ravel()

            # Calculates x - mean
            np.subtract(x, tfine_aux, out=x)

            d1_regsse = simps(np.multiply(x, D1x, out=x),
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

            # Stores the original template instead of building it again
            template = original_template
        else:

            # Stores the template in an FDataGrid
            template = FDataGrid(tfine_aux, grid_points=output_points)

        return delta, template

    def fit_transform(self, X: FData, y=None):
        """Fit the estimator and transform the data.

        Args:
            X (FData): Functional dataset to be transformed.
            y (ignored): not used, present for API consistency by convention.

        Returns:
            FData: Functional data registered.

        """
        self.deltas_, self.template_ = self._compute_deltas(X, self.template)

        return X.shift(self.deltas_, restrict_domain=self.restrict_domain,
                       extrapolation=self.extrapolation,
                       eval_points=self.output_points)

    def fit(self, X: FData, y=None):
        """Fit the estimator.

        Args:
            X (FData): Functional dataset used to construct the template for
                the alignment.
            y (ignored): not used, present for API consistency by convention.

        Returns:
            RegistrationTransformer: self

        Raises:
            AttributeError: If this method is call when restrict_domain=True.

        """
        if self.restrict_domain:
            raise AttributeError("fit and predict are not available when "
                                 "restrict_domain=True, fitting and "
                                 "transformation should be done together. Use "
                                 "an extrapolation method with "
                                 "restrict_domain=False or fit_predict")

        # If the template is an FData, fit doesnt learn anything
        if isinstance(self.template, FData):
            self.template_ = self.template

        else:
            _, self.template_ = self._compute_deltas(X, self.template)

        return self

    def transform(self, X: FData, y=None):
        """Register the data.

        Transforms the data using the template previously learned during
        fitting.

        Args:
            X (FData): Functional dataset to be transformed.
            y (ignored): not used, present for API consistency by convention.

        Returns:
            FData: Functional data registered.

        Raises:
            AttributeError: If this method is call when restrict_domain=True.

        """

        if self.restrict_domain:
            raise AttributeError("fit and predict are not available when "
                                 "restrict_domain=True, fitting and "
                                 "transformation should be done together. Use "
                                 "an extrapolation method with "
                                 "restrict_domain=False or fit_predict")

        # Check is fitted
        check_is_fitted(self, 'template_')

        deltas, template = self._compute_deltas(X, self.template_)
        self.template_ = template
        self.deltas_ = deltas

        return X.shift(deltas, restrict_domain=self.restrict_domain,
                       extrapolation=self.extrapolation,
                       eval_points=self.output_points)

    def inverse_transform(self, X: FData, y=None):
        """Applies the inverse transformation.

        Applies the opossite shift used in the last call to `transform`.

        Args:
            X (FData): Functional dataset to be transformed.
            y (ignored): not used, present for API consistency by convention.

        Returns:
            FData: Functional data registered.

        Examples:

        Creates a synthetic functional dataset.

        >>> from skfda.preprocessing.registration import ShiftRegistration
        >>> from skfda.datasets import make_sinusoidal_process
        >>> fd = make_sinusoidal_process(error_std=0, random_state=1)
        >>> fd.extrapolation = 'periodic'

        Dataset registration and centering.

        >>> reg = ShiftRegistration()
        >>> fd_registered = reg.fit_transform(fd)
        >>> fd_centered = fd_registered - fd_registered.mean()

        Reverse the translation applied during the registration.

        >>> reg.inverse_transform(fd_centered)
        FDataGrid(...)

        """
        if not hasattr(self, "deltas_"):
            raise AttributeError("Data must be previously transformed to learn"
                                 " the inverse transformation")
        elif len(X) != len(self.deltas_):
            raise ValueError("Data must contain the same number of samples "
                             "than the dataset previously transformed")

        return X.shift(-self.deltas_, restrict_domain=self.restrict_domain,
                       extrapolation=self.extrapolation,
                       eval_points=self.output_points)

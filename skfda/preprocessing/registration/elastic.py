

import scipy.integrate
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


import numpy as np
import optimum_reparam


from . import invert_warping
from .base import RegistrationTransformer
from ._registration_utils import _normalize_scale
from ... import FDataGrid
from ..._utils import check_is_univariate
from ...representation.interpolation import SplineInterpolator


__author__ = "Pablo Marcos Manchón"
__email__ = "pablo.marcosm@estudiante.uam.es"

###############################################################################
# Based on the original implementation of J. Derek Tucker in                  #
# *fdasrsf_python* (https://github.com/jdtuck/fdasrsf_python)                 #
# and *ElasticFDA.jl* (https://github.com/jdtuck/ElasticFDA.jl).              #
###############################################################################

class SRSF(BaseEstimator, TransformerMixin):
    r"""Square-Root Slope Function (SRSF) transform.

    Let :math:`f : [a,b] \rightarrow \mathbb{R}` be an absolutely continuous
    function, the SRSF transform is defined as

    .. math::
        SRSF(f(t)) = sgn(f(t)) \sqrt{|\dot f(t)|} = q(t)

    This representation it is used to compute the extended non-parametric
    Fisher-Rao distance between functions, wich under the SRSF representation
    becomes the usual :math:`\mathbb{L}^2` distance between functions.
    See [SK16-4-6]_ .

    The inverse SRSF transform is defined as

    .. math::
        f(t) = f(a) + \int_{a}^t q(t)|q(t)|dt .

    This transformation is a mapping up to constant. Given the SRSF and the
    initial value :math:`f(a)` the original function can be obtained, for this
    reason it is necessary to store the value :math:`f(a)` during the fit,
    which is dropped due to derivation. If it is applied the inverse
    transformation without fit the estimator it is assumed that :math:`f(a)=0`.

    Attributes:
        eval_points (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of the
            fdatagrid.
        store_initial (bool): If true stores the value :math:`f(a)` of the
            samples during fitting to apply the inverse transform.
            Defaults True.

    Note:
        Due to the use of derivatives it is recommended that the samples are
        sufficiently smooth, or have passed a smoothing preprocessing before,
        in order to achieve good results.

    References:
        ..  [SK16-4-6] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Square-Root Slope Function
            Representation* (pp. 91-93). Springer.

    Examples:

        Create a toy dataset and apply the transformation and its inverse.

        >>> from skfda.datasets import make_sinusoidal_process
        >>> from skfda.preprocessing.registration.elastic import SRSF
        >>> fd = make_sinusoidal_process(error_std=0, random_state=0)
        >>> srsf = SRSF()
        >>> srsf
        SRSF(output_points=None, store_initial=True)

        Fits the estimator (to apply the inverse transform) and apply the SRSF

        >>> q = srsf.fit_transform(fd)

        Apply the inverse transform.

        >>> fd_pull_back = srsf.inverse_transform(q)

        The original and the pull back `fd` are almost equal

        >>> zero = fd - fd_pull_back
        >>> zero.data_matrix.flatten().round(3)
        array([ 0.,  0.,  0., ...])

    """
    def __init__(self, output_points=None, store_initial=True):
        """Initializes the transformer.
        Args:
            eval_points: (array_like, optional): Set of points where the
                functions are evaluated, by default uses the sample points of
                the :class:`FDataGrid <skfda.FDataGrid>` transformed.
            store_initial (bool): If true stores the value :math:`f(a)` of the
                samples during fitting to apply the inverse transform.
                Defaults True.

        """
        self.output_points = output_points
        self.store_initial = store_initial


    def fit(self, X: FDataGrid):
        """Fits the transformer.
        Stores the initial value of the functions to be transformed, in order
        to apply its inverse transform.
        Args:
            X (:class:`FDataGrid <skfda.FDataGrid`): Functional data to be
                transformed.
        Returns:
            (Estimator): self
        """
        check_is_univariate(X)

        if self.store_initial:
            a = X.domain_range[0][0] # Stores initial value
            self.initial_ = X(a).reshape(X.n_samples, 1, X.dim_codomain)

        return self

    def transform(self, X: FDataGrid):
        r"""Computes the square-root slope function (SRSF) transform.
        Let :math:`f : [a,b] \rightarrow \mathbb{R}` be an absolutely continuous
        function, the SRSF transform is defined as [SK16-4-6-1]_:
        .. math::
            SRSF(f(t)) = sgn(f(t)) \sqrt{\dot f(t)|} = q(t)
        Args:
            X (:class:`FDataGrid`): Functions to be transformed.
        Returns:
            :class:`FDataGrid`: SRSF functions.
        Raises:
            ValueError: If functions are not univariate.
        References:
            ..  [SK16-4-6-1] Srivastava, Anuj & Klassen, Eric P. (2016).
                Functional and shape data analysis. In *Square-Root Slope
                Function Representation* (pp. 91-93). Springer.

        """
        check_is_univariate(X)

        if self.store_initial:
            check_is_fitted(self, 'initial_')

        if self.output_points is None:
            output_points = X.sample_points[0]
        else:
            output_points = self.output_points

        g = X.derivative()

        # Evaluation with the corresponding interpolation
        data_matrix = g(output_points, keepdims=False)

        # SRSF(f) = sign(f) * sqrt|Df| (avoiding multiple allocation)
        sign_g = np.sign(data_matrix)
        data_matrix = np.abs(data_matrix, out=data_matrix)
        data_matrix = np.sqrt(data_matrix, out=data_matrix)
        data_matrix *= sign_g


        return X.copy(data_matrix=data_matrix, sample_points=output_points)


    def inverse_transform(self, X: FDataGrid):
        r"""Computes the inverse SRSF transform.
        Given the srsf and the initial value the original function can be
        obtained as [SK16-4-6-2]_ :
        .. math::
            f(t) = f(a) + \int_{a}^t q(t)|q(t)|dt
        where :math:`q(t)=SRSF(f(t))`.
        If it is applied this inverse transformation without fitting the
        estimator it is assumed that :math:`f(a)=0`.
        Args:
            X (:class:`FDataGrid`): SRSF to be transformed.
        Returns:
            :class:`FDataGrid`: Functions in the original space.
        Raises:
            ValueError: If functions are multidimensional.
        References:
            ..  [SK16-4-6-2] Srivastava, Anuj & Klassen, Eric P. (2016).
                Functional and shape data analysis. In *Square-Root Slope
                Function Representation* (pp. 91-93). Springer.

        """
        check_is_univariate(X)

        if self.store_initial:
            check_is_fitted(self, 'initial_')

        if self.output_points is None:
            output_points = X.sample_points[0]
        else:
            output_points = self.output_points

        data_matrix = X(output_points, keepdims=True)

        data_matrix *= np.abs(data_matrix)

        f_data_matrix = scipy.integrate.cumtrapz(data_matrix, x=output_points,
                                                 axis=1, initial=0)

        # If the transformer was fitted, sum the initial value
        if hasattr(self, 'initial_'):
            f_data_matrix += self.initial_

        return X.copy(data_matrix=f_data_matrix, sample_points=output_points)


def _elastic_alignment_array(template_data, q_data,
                             eval_points, lam, grid_dim):
    r"""Wrapper between the cython interface and python.

    Selects the corresponding routine depending on the dimensions of the
    arrays.

    Args:
        template_data (numpy.ndarray): Array with the srsf of the template.
        q_data (numpy.ndarray): Array with the srsf of the curves
                                to be aligned.
        eval_points (numpy.ndarray): Discretisation points of the functions.
        lam (float): Penalisation term.
        grid_dim (int): Dimension of the grid used in the alignment algorithm.

    Return:
        (numpy.ndarray): Array with the same shape than q_data with the srsf of
        the functions aligned to the template(s).
    """

    # Select cython function
    if template_data.ndim == 1 and q_data.ndim == 1:
        reparam = optimum_reparam.coptimum_reparam

    elif template_data.ndim == 1:
        reparam = optimum_reparam.coptimum_reparam_n

    else:
        reparam = optimum_reparam.coptimum_reparam_n2

    return reparam(np.ascontiguousarray(template_data.T),
                   np.ascontiguousarray(eval_points),
                   np.ascontiguousarray(q_data.T),
                   lam, grid_dim).T



class ElasticRegistration(RegistrationTransformer):
    r"""Align a FDatagrid using the SRSF framework.

    Let :math:`f` be a function of the functional data object wich will be
    aligned to the template :math:`g`. Calculates the warping wich minimises
    the Fisher-Rao distance between :math:`g` and the registered function
    :math:`f^*(t)=f(\gamma^*(t))=f \circ \gamma^*`.

    .. math::
        \gamma^* = argmin_{\gamma \in \Gamma} d_{\lambda}(f \circ
        \gamma, g)

    Where :math:`d_{\lambda}` denotes the extended Fisher-Rao distance with a
    penalty term, used to control the amount of warping.

    .. math::
        d_{\lambda}^2(f \circ \gamma, g) = \| SRSF(f \circ \gamma)
        \sqrt{\dot{\gamma}} - SRSF(g)\|_{\mathbb{L}^2}^2 + \lambda
        \mathcal{R}(\gamma)

    In the implementation it is used as penalty term

    .. math::
        \mathcal{R}(\gamma) = \|\sqrt{\dot{\gamma}}- 1 \|_{\mathbb{L}^2}^2

    Wich restrict the amount of elasticity employed in the alignment.

    The registered function :math:`f^*(t)` can be calculated using the
    composition :math:`f^*(t)=f(\gamma^*(t))`.

    If the template is not specified it is used the Karcher mean of the set of
    functions under the elastic metric to perform the alignment, also known as
    `elastic mean`, wich is the local minimum of the sum of squares of elastic
    distances. See :func:`elastic_mean`.

    In [SK16-4-2]_ are described extensively the algorithms employed and
    the SRSF framework.

    Args:
        template (str, :class:`FDataGrid` or callable, optional): Template to
            align the curves. Can contain 1 sample to align all the curves to
            it or the same number of samples than the fdatagrid. By default
            `elastic mean`, in which case :func:`elastic_mean` is called.
        penalty_term (float, optional): Controls the amount of elasticity.
            Defaults to 0.
        output_points (array_like, optional): Set of points where the
            functions are evaluated, by default uses the sample points of the
            fdatagrid which will be transformed.
        grid_dim (int, optional): Dimension of the grid used in the DP
            alignment algorithm. Defaults 7.
        **kwargs: Named arguments to be passed to be passed to the callable
            which constructs the template or to :func:`elastic_mean` by
            default.

    Attributes:
        template_ (:class:`FDataGrid`): Template learned during fitting,
            used for alignment in :meth:`transform`.
        warping_ (:class:`FDataGrid`): Warping applied during the last
            transformation.

    References:
        ..  [SK16-4-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Functional Data and Elastic
            Registration* (pp. 73-122). Springer.

    """
    def __init__(self, template="elastic mean", penalty=0., output_points=None,
                 grid_dim=7, **kwargs):
        """Initializes the registration transformer"""

        self.template = template
        self.penalty = penalty
        self.output_points = output_points
        self.grid_dim = grid_dim
        self.kwargs = kwargs

    def fit(self, X: FDataGrid=None, y=None):
        """Fit the transformer.

        Learns the template used during the transformation.

        Args:
            X (FDataGrid, optionl): Functional samples used as training
                samples. If the template provided it is an FDataGrid this
                samples are it is not need to construct the template from the
                samples and this argument is ignored.
            y (Ignored): Present for API conventions.

        Returns:
            RegistrationTransformer: self.

        """
        if isinstance(self.template, FDataGrid):
            self.template_ = self.template # Template already constructed
        elif X is None:
            raise ValueError("Must be provided a dataset X to construct the "
                             "template.")
        elif self.template == "elastic mean":
            self.template_ = elastic_mean(X, **self.kwargs)
        else:
            self.template_ = self.template(X, **self.kwargs)

        # Constructs the SRSF of the template
        srsf = SRSF(output_points=self.output_points, store_initial=False)
        self._template_srsf = srsf.fit_transform(self.template_)

        return self


    def transform(self, X: FDataGrid, y=None):
        """Apply elastic registration to the data.

        Args:
            X (:class:`FDataGrid`): Functional data to be registered.
            y (ignored):

        Returns:
            :class:`FDataGrid`: Registered samples.

        """
        check_is_fitted(self, '_template_srsf')
        check_is_univariate(X)

        if (len(self._template_srsf) != 1 and
            len(X) != len(self._template_srsf)):

            raise ValueError("The template should contain one sample to align "
                             "all the curves to the same function or the "
                             "same number of samples than X.")

        srsf = SRSF(output_points=self.output_points, store_initial=False)
        fdatagrid_srsf = srsf.fit_transform(X)

        # Points of discretization
        if self.output_points is None:
            output_points = fdatagrid_srsf.sample_points[0]
        else:
            output_points = self.output_points

        # Discretizacion in evaluation points
        q_data = fdatagrid_srsf(output_points, keepdims=False)
        template_data = self._template_srsf(output_points, keepdims=False)

        if q_data.shape[0] == 1:
            q_data = q_data[0]

        if template_data.shape[0] == 1:
            template_data = template_data[0]

        # Values of the warping
        gamma = _elastic_alignment_array(template_data, q_data,
                                         _normalize_scale(output_points),
                                         self.penalty, self.grid_dim)

        # Normalize warping to original interval
        gamma = _normalize_scale(
            gamma, a=output_points[0], b=output_points[-1])

        # Interpolator
        interpolator = SplineInterpolator(interpolation_order=3, monotone=True)

        self.warping_ = FDataGrid(gamma, output_points,
                                  interpolator=interpolator)


        return X.compose(self.warping_, eval_points=output_points)

    def inverse_transform(self, X: FDataGrid):
        r"""Reverse the registration procedure previosly applied.

        Let :math:`gamma(t)` the warping applied to construct a registered
        functional datum :math:`f^*(t)=f(\gamma(t))`.

        Given a functional datum :math:`f^*(t) it is computed
        :math:`\gamma^{-1}(t)` to reverse the registration procedure
        :math:`f(t)=f^*(\gamma^{-1}(t))`.

        Args:
            X (:class:`FDataGrid`): Functional data to apply the reverse
                transform.

        Returns:
            :class:`FDataGrid`: Functional data compose by the inverse warping.

        Raises:
            ValueError: If the warpings :math:`\gamma` were not build via
            :meth:`transform` or if the number of samples of `X` is different
            than the number of samples of the dataset previosly transformed.

        Examples:

            Center the datasets taking into account the misalignment.

            >>> from skfda.preprocessing.registration import \
            ...                                             ElasticRegistration
            >>> from skfda.datasets import make_multimodal_samples
            >>> X = make_multimodal_samples(random_state=0)

            Registration of the dataset.

            >>> elastic_registration = ElasticRegistration()
            >>> X = elastic_registration.fit_transform(X)

            Substract the elastic mean build as template during the
            registration and reverse the transformation.

            >>> X = X - elastic_registration.template_
            >>> X_center = elastic_registration.inverse_transform(X)
            >>> X_center
            FDataGrid(...)


        See also:
            :func:`invert_warping`

        """
        if not hasattr(self, 'warping_'):
            raise ValueError("Data must be previosly transformed to apply the "
                             "inverse transform")
        elif len(X) != len(self.warping_):
            raise ValueError("Data must contain the same number of samples "
                             "than the dataset previously transformed")

        inverse_warping = invert_warping(self.warping_)

        return X.compose(inverse_warping, eval_points=self.output_points)



def warping_mean(warping, *, iter=20, tol=1e-5, step_size=1., eval_points=None,
                 return_shooting=False):
    r"""Compute the karcher mean of a set of warpings.

    Let :math:`\gamma_i i=1...n` be a set of warping functions
    :math:`\gamma_i:[a,b] \rightarrow [a,b]` in :math:`\Gamma`, i.e.,
    monotone increasing and with the restriction :math:`\gamma_i(a)=a \,
    \gamma_i(b)=b`.

    The karcher mean :math:`\bar \gamma` is defined as the warping that
    minimises locally the sum of Fisher-Rao squared distances.
    [SK16-8-3-2]_.

    .. math::
        \bar \gamma = argmin_{\gamma \in \Gamma} \sum_{i=1}^{n}
         d_{FR}^2(\gamma, \gamma_i)

    The computation is performed using the structure of Hilbert Sphere obtained
    after a transformation of the warpings, see [S11-3-3]_.

    Args:
        warping (:class:`FDataGrid`): Set of warpings.
        iter (int): Maximun number of interations. Defaults to 20.
        tol (float): Convergence criterion, if the norm of the mean of the
            shooting vectors, :math:`| \bar v |<tol`, the algorithm will stop.
            Defaults to 1e-5.
        step_size (float): Step size :math:`\epsilon` used to update the mean.
            Default to 1.
        eval_points (array_like): Discretisation points of the warpings.
        shooting (boolean): If true it is returned a tuple with the mean and
            the shooting vectors, otherwise only the mean is returned.

    Return:
        (:class:`FDataGrid`) Fdatagrid with the mean of the warpings. If
        shooting is True the shooting vectors will be returned in a tuple with
        the mean.

    References:
        ..  [SK16-8-3-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Template: Center of the Mean Orbit*
            (pp. 274-277). Springer.

        ..  [S11-3-3] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Center of an Orbit* (pp. 9-10).
            arXiv:1103.3817v2.
    """

    if eval_points is None:
        eval_points = warping.sample_points[0]

    original_eval_points = eval_points

    if warping.sample_points[0][0] != 0 or warping.sample_points[0][-1] != 1:

        eval_points = _normalize_scale(eval_points)
        warping = FDataGrid(_normalize_scale(warping.data_matrix[..., 0]),
                            _normalize_scale(warping.sample_points[0]))

    srsf = SRSF(output_points=eval_points, store_initial=False)
    psi = srsf.fit_transform(warping).data_matrix[..., 0].T
    mu = srsf.fit_transform(warping.mean()).data_matrix[0]

    dot_aux = np.empty(psi.shape)

    n_points = mu.shape[0]

    sine = np.empty((warping.n_samples, 1))

    for _ in range(iter):
        # Dot product
        # <psi, mu> = S psi(t) mu(t) dt
        dot = scipy.integrate.simps(np.multiply(psi, mu, out=dot_aux),
                                    eval_points, axis=0)

        # Theorically is not possible (Cauchy–Schwarz inequallity), but due to
        # numerical approximation could be greater than 1
        dot[dot < -1] = -1
        dot[dot > 1] = 1
        theta = np.arccos(dot)[:, np.newaxis]

        # Be carefully with tangent vectors and division by 0
        idx = theta[:, 0] > tol
        sine[idx] = theta[idx] / np.sin(theta[idx])
        sine[~idx] = 0.

        # compute shooting vector
        cos_theta = np.repeat(np.cos(theta), n_points, axis=1)
        shooting = np.multiply(sine, (psi - np.multiply(cos_theta.T, mu)).T)

        # Mean of shooting vectors
        vmean = shooting.mean(axis=0, keepdims=True)
        v_norm = scipy.integrate.simps(np.square(vmean[0]))**(.5)

        # Convergence criterion
        if v_norm < tol:
            break

        # Update of mu
        mu *= np.cos(step_size * v_norm)
        vmean += np.sin(step_size * v_norm) / v_norm
        mu += vmean.T

    # Recover mean in original gamma space
    warping_mean = scipy.integrate.cumtrapz(np.square(mu, out=mu)[:, 0],
                                            x=eval_points, initial=0)

    # Affine traslation
    warping_mean = _normalize_scale(warping_mean,
                                    a=original_eval_points[0],
                                    b=original_eval_points[-1])

    monotone_interpolator = SplineInterpolator(interpolation_order=3,
                                               monotone=True)

    mean = FDataGrid([warping_mean], sample_points=original_eval_points,
                     interpolator=monotone_interpolator)

    # Shooting vectors are used in models based in the amplitude-phase
    # decomposition under this metric.
    if return_shooting:
        return mean, shooting

    return mean


def elastic_mean(fdatagrid, *, lam=0., center=True, iter=20, tol=1e-3,
                 initial=None, eval_points=None, fdatagrid_srsf=None,
                 grid_dim=7, **kwargs):
    r"""Compute the karcher mean under the elastic metric.

    Calculates the karcher mean of a set of functional samples in the amplitude
    space :math:`\mathcal{A}=\mathcal{F}/\Gamma`.

    Let :math:`q_i` the corresponding SRSF of the observation :math:`f_i`.
    The space :math:`\mathcal{A}` is defined using the equivalence classes
    :math:`[q_i]=\{ q_i \circ \gamma \| \gamma \in \Gamma \}`, where
    :math:`\Gamma` denotes the space of warping functions. The karcher mean
    in this space is defined as

    .. math::
        [\mu_q] = argmin_{[q] \in \mathcal{A}} \sum_{i=1}^n
        d_{\lambda}^2([q],[q_i])

    Once :math:`[\mu_q]` is obtained it is selected the element of the
    equivalence class which makes the mean of the warpings employed be the
    identity.

    See [SK16-8-3-1]_ and [S11-3]_.

    Args:
        fdatagrid (:class:`FDataGrid`): Set of functions to compute the mean.
        lam (float): Penalisation term. Defaults to 0.
        center (boolean): If true it is computed the mean of the warpings and
            used to select a central mean. Defaults True.
        iter (int): Maximun number of iterations. Defaults to 20.
        tol (float): Convergence criterion, the algorithm will stop if
            :math:´\|mu^{(\nu)} - mu^{(\nu - 1)} \|_2 / \| mu^{(\nu-1)} \|_2
            < tol´.
        initial (float): Value of the mean at the starting point. By default
            takes the average of the initial points of the samples.
        eval_points (array_like): Points of discretization of the fdatagrid.
        fdatagrid_srsf (:class:`FDataGrid`): SRSF if the fdatagrid, if it is
            passed it is not computed in the algorithm.
        grid_dim (int, optional): Dimension of the grid used in the alignment
            algorithm. Defaults 7.
        ** kwargs : Named options to be pased to :func:`warping_mean`.

    Return:
        (:class:`FDataGrid`): FDatagrid with the mean of the functions.

    Raises:
        ValueError: If the object is multidimensional or the shape of the srsf
            do not match with the fdatagrid.

    References:
        ..  [SK16-8-3-1] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Karcher Mean of Amplitudes*
            (pp. 273-274). Springer.

        .. [S11-3] Srivastava, Anuj et. al. Registration of Functional Data
            Using Fisher-Rao Metric (2011). In *Karcher Mean and Function
            Alignment* (pp. 7-10). arXiv:1103.3817v2.

    """

    check_is_univariate(fdatagrid)
    srsf_transformer = SRSF(store_initial=False, output_points=eval_points)

    if fdatagrid_srsf is not None:
        check_is_univariate(fdatagrid_srsf)

    else:
        fdatagrid_srsf = srsf_transformer.fit_transform(fdatagrid)

    if eval_points is not None:
        eval_points = np.asarray(eval_points)
    else:
        eval_points = fdatagrid.sample_points[0]

    eval_points_normalized = _normalize_scale(eval_points)
    y_scale = eval_points[-1] - eval_points[0]

    interpolator = SplineInterpolator(interpolation_order=3, monotone=True)

    # Discretisation points
    fdatagrid_normalized = FDataGrid(fdatagrid(eval_points) / y_scale,
                                     sample_points=eval_points_normalized)

    srsf = fdatagrid_srsf(eval_points, keepdims=False)

    # Initialize with function closest to the L2 mean with the L2 distance
    centered = (srsf.T - srsf.mean(axis=0, keepdims=True).T).T

    distances = scipy.integrate.simps(np.square(centered, out=centered),
                                      eval_points_normalized, axis=1)

    # Initialization of iteration
    mu = srsf[np.argmin(distances)]
    mu_aux = np.empty(mu.shape)
    mu_1 = np.empty(mu.shape)

    # Main iteration
    for _ in range(iter):

        gammas = _elastic_alignment_array(
            mu, srsf, eval_points_normalized, lam, grid_dim)
        gammas = FDataGrid(gammas, sample_points=eval_points_normalized,
                           interpolator=interpolator)

        fdatagrid_normalized = fdatagrid_normalized.compose(gammas)
        srsf = srsf_transformer.fit_transform(
            fdatagrid_normalized).data_matrix[..., 0]

        # Next iteration
        mu_1 = srsf.mean(axis=0, out=mu_1)

        # Convergence criterion
        mu_norm = np.sqrt(scipy.integrate.simps(np.square(mu, out=mu_aux),
                                                eval_points_normalized))

        mu_diff = np.sqrt(scipy.integrate.simps(np.square(mu - mu_1,
                                                          out=mu_aux),
                                                eval_points_normalized))

        if mu_diff / mu_norm < tol:
            break

        mu = mu_1


    if initial is None:
        initial = fdatagrid.data_matrix[:, 0].mean()

    srsf_transformer.set_params(store_initial=True)
    srsf_transformer.initial_ = initial


    # Karcher mean orbit in space L2/Gamma
    karcher_mean = srsf_transformer.inverse_transform(
        fdatagrid.copy(data_matrix=[mu], sample_points=eval_points))

    if center:
        # Gamma mean in Hilbert Sphere
        mean_normalized = warping_mean(gammas, return_shooting=False, **kwargs)

        gamma_mean = FDataGrid(_normalize_scale(
            mean_normalized.data_matrix[..., 0],
            a=eval_points[0],
            b=eval_points[-1]),
            sample_points=eval_points)

        gamma_inverse = invert_warping(gamma_mean)

        karcher_mean = karcher_mean.compose(gamma_inverse)

    # Return center of the orbit
    return karcher_mean

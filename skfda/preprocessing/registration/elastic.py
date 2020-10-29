
from fdasrsf.utility_functions import optimum_reparam
import scipy.integrate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np

from . import invert_warping
from ... import FDataGrid
from ..._utils import check_is_univariate
from ...representation.interpolation import SplineInterpolation
from ._warping import _normalize_scale
from .base import RegistrationTransformer


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
        initial_value (float, optional): Initial value to apply in the
            inverse transformation. If `None` there are stored the initial
            values of the functions during the transformation to apply
            during the inverse transformation. Defaults None.

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
        SRSF(...)

        Fits the estimator (to apply the inverse transform) and apply the SRSF

        >>> q = srsf.fit_transform(fd)

        Apply the inverse transform.

        >>> fd_pull_back = srsf.inverse_transform(q)

        The original and the pull back `fd` are almost equal

        >>> zero = fd - fd_pull_back
        >>> zero.data_matrix.flatten().round(3)
        array([ 0.,  0.,  0., ..., -0., -0., -0.])

    """

    def __init__(self, output_points=None, initial_value=None):
        """Initializes the transformer.

        Args:
            eval_points: (array_like, optional): Set of points where the
                functions are evaluated, by default uses the sample points of
                the :class:`FDataGrid <skfda.FDataGrid>` transformed.
            initial_value (float, optional): Initial value to apply in the
                inverse transformation. If `None` there are stored the initial
                values of the functions during the transformation to apply
                during the inverse transformation. Defaults None.

        """
        self.output_points = output_points
        self.initial_value = initial_value

    def fit(self, X=None, y=None):
        """This transformer do not need to be fitted.

        Args:
            X (Ignored): Present for API conventions.
            y (Ignored): Present for API conventions.

        Returns:
            (Estimator): self

        """
        return self

    def transform(self, X: FDataGrid, y=None):
        r"""Computes the square-root slope function (SRSF) transform.

        Let :math:`f : [a,b] \rightarrow \mathbb{R}` be an absolutely continuous
        function, the SRSF transform is defined as [SK16-4-6-1]_:

        .. math::

            SRSF(f(t)) = sgn(f(t)) \sqrt{\dot f(t)|} = q(t)

        Args:
            X (:class:`FDataGrid`): Functions to be transformed.
            y (Ignored): Present for API conventions.

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

        if self.output_points is None:
            output_points = X.grid_points[0]
        else:
            output_points = self.output_points

        g = X.derivative()

        # Evaluation with the corresponding interpolation
        data_matrix = g(output_points)[..., 0]

        # SRSF(f) = sign(f) * sqrt|Df| (avoiding multiple allocation)
        sign_g = np.sign(data_matrix)
        data_matrix = np.abs(data_matrix, out=data_matrix)
        data_matrix = np.sqrt(data_matrix, out=data_matrix)
        data_matrix *= sign_g

        # Store the values of the transformation
        if self.initial_value is None:
            a = X.domain_range[0][0]
            self.initial_value_ = X(a).reshape(X.n_samples, 1, X.dim_codomain)

        return X.copy(data_matrix=data_matrix, grid_points=output_points)

    def inverse_transform(self, X: FDataGrid, y=None):
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
            y (Ignored): Present for API conventions.

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

        if self.initial_value is None and not hasattr(self, 'initial_value_'):
            raise AttributeError("When initial_value=None is expected a "
                                 "previous transformation of the data to "
                                 "store the initial values to apply in the "
                                 "inverse transformation. Also it is possible "
                                 "to fix these values setting the attribute"
                                 "initial value without a previous "
                                 "transformation.")

        if self.output_points is None:
            output_points = X.grid_points[0]
        else:
            output_points = self.output_points

        data_matrix = X(output_points)

        data_matrix *= np.abs(data_matrix)

        f_data_matrix = scipy.integrate.cumtrapz(data_matrix, x=output_points,
                                                 axis=1, initial=0)

        # If the transformer was fitted, sum the initial value
        if self.initial_value is None:
            f_data_matrix += self.initial_value_
        else:
            f_data_matrix += self.initial_value

        return X.copy(data_matrix=f_data_matrix, grid_points=output_points)


def _elastic_alignment_array(template_data, q_data,
                             eval_points, penalty, grid_dim):
    r"""Wrapper between the cython interface and python.

    Selects the corresponding routine depending on the dimensions of the
    arrays.

    Args:
        template_data (numpy.ndarray): Array with the srsf of the template.
        q_data (numpy.ndarray): Array with the srsf of the curves
                                to be aligned.
        eval_points (numpy.ndarray): Discretisation points of the functions.
        penalty (float): Penalisation term.
        grid_dim (int): Dimension of the grid used in the alignment algorithm.

    Return:
        (numpy.ndarray): Array with the same shape than q_data with the srsf of
        the functions aligned to the template(s).
    """

    return optimum_reparam(np.ascontiguousarray(template_data.T),
                           np.ascontiguousarray(eval_points),
                           np.ascontiguousarray(q_data.T),
                           method="DP2",
                           lam=penalty, grid_dim=grid_dim).T


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
    distances. See :func:`~elastic_mean`.

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

    Attributes:
        template_ (:class:`FDataGrid`): Template learned during fitting,
            used for alignment in :meth:`transform`.
        warping_ (:class:`FDataGrid`): Warping applied during the last
            transformation.

    References:
        ..  [SK16-4-2] Srivastava, Anuj & Klassen, Eric P. (2016). Functional
            and shape data analysis. In *Functional Data and Elastic
            Registration* (pp. 73-122). Springer.

    Examples:

        Elastic registration of with train/test sets.

        >>> from skfda.preprocessing.registration import \
        ...                                             ElasticRegistration
        >>> from skfda.datasets import make_multimodal_samples
        >>> X_train = make_multimodal_samples(n_samples=15, random_state=0)
        >>> X_test = make_multimodal_samples(n_samples=3, random_state=1)

        Fit the transformer, which learns the elastic mean of the train
        set as template.

        >>> elastic_registration = ElasticRegistration()
        >>> elastic_registration.fit(X_train)
        ElasticRegistration(...)

        Registration of the test set.

        >>> elastic_registration.transform(X_test)
        FDataGrid(...)

    """

    def __init__(self, template="elastic mean", penalty=0., output_points=None,
                 grid_dim=7):
        """Initializes the registration transformer"""

        self.template = template
        self.penalty = penalty
        self.output_points = output_points
        self.grid_dim = grid_dim

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
            self.template_ = self.template  # Template already constructed
        elif X is None:
            raise ValueError("Must be provided a dataset X to construct the "
                             "template.")
        elif self.template == "elastic mean":
            self.template_ = elastic_mean(X)
        else:
            self.template_ = self.template(X)

        # Constructs the SRSF of the template
        srsf = SRSF(output_points=self.output_points, initial_value=0)
        self._template_srsf = srsf.fit_transform(self.template_)

        return self

    def transform(self, X: FDataGrid, y=None):
        """Apply elastic registration to the data.

        Args:
            X (:class:`FDataGrid`): Functional data to be registered.
            y (ignored): Present for API conventions.

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

        srsf = SRSF(output_points=self.output_points, initial_value=0)
        fdatagrid_srsf = srsf.fit_transform(X)

        # Points of discretization
        if self.output_points is None:
            output_points = fdatagrid_srsf.grid_points[0]
        else:
            output_points = self.output_points

        # Discretizacion in evaluation points
        q_data = fdatagrid_srsf(output_points)[..., 0]
        template_data = self._template_srsf(output_points)[..., 0]

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

        # Interpolation
        interpolation = SplineInterpolation(
            interpolation_order=3, monotone=True)

        self.warping_ = FDataGrid(gamma, output_points,
                                  interpolation=interpolation)

        return X.compose(self.warping_, eval_points=output_points)

    def inverse_transform(self, X: FDataGrid, y=None):
        r"""Reverse the registration procedure previosly applied.

        Let :math:`gamma(t)` the warping applied to construct a registered
        functional datum :math:`f^*(t)=f(\gamma(t))`.

        Given a functional datum :math:`f^*(t) it is computed
        :math:`\gamma^{-1}(t)` to reverse the registration procedure
        :math:`f(t)=f^*(\gamma^{-1}(t))`.

        Args:
            X (:class:`FDataGrid`): Functional data to apply the reverse
                transform.
            y (Ignored): Present for API conventions.

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


def warping_mean(warping, *, max_iter=100, tol=1e-6, step_size=.3):
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
        warping (:class:`~skfda.FDataGrid`): Set of warpings.
        max_iter (int): Maximum number of interations. Defaults to 100.
        tol (float): Convergence criterion, if the norm of the mean of the
            shooting vectors, :math:`| \bar v |<tol`, the algorithm will stop.
            Defaults to 1e-5.
        step_size (float): Step size :math:`\epsilon` used to update the mean.
            Default to 1.

    Return:
        (:class:`~skfda.FDataGrid`) Fdatagrid with the mean of the warpings. If
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

    eval_points = warping.grid_points[0]
    original_eval_points = eval_points

    # Rescale warping to (0, 1)
    if warping.grid_points[0][0] != 0 or warping.grid_points[0][-1] != 1:

        eval_points = _normalize_scale(eval_points)
        warping = FDataGrid(_normalize_scale(warping.data_matrix[..., 0]),
                            _normalize_scale(warping.grid_points[0]))

    # Compute srsf of warpings and their mean
    srsf = SRSF(output_points=eval_points, initial_value=0)
    psi = srsf.fit_transform(warping)

    # Find psi closest to the mean
    psi_centered = psi - srsf.fit_transform(warping.mean())
    psi_data = psi_centered.data_matrix[..., 0]
    np.square(psi_data, out=psi_data)
    d = psi_data.sum(axis=1).argmin()

    # Get raw values to calculate
    mu = psi[d].data_matrix[0, ..., 0]
    psi = psi.data_matrix[..., 0]
    vmean = np.empty((1, len(eval_points)))

    # Construction of shooting vectors
    for _ in range(max_iter):

        vmean[0] = 0.
        # Compute shooting vectors
        for i in range(len(warping)):
            psi_i = psi[i]

            inner = scipy.integrate.simps(mu * psi_i, x=eval_points)
            inner = max(min(inner, 1), -1)

            theta = np.arccos(inner)

            if theta > 1e-10:
                vmean += theta / np.sin(theta) * (psi_i - np.cos(theta) * mu)

        # Mean of shooting vectors
        vmean /= warping.n_samples
        v_norm = np.sqrt(scipy.integrate.simps(np.square(vmean)))

        # Convergence criterion
        if v_norm < tol:
            break

        # Calculate exponential map of mu
        a = np.cos(step_size * v_norm)
        b = np.sin(step_size * v_norm) / v_norm
        mu = a * mu + b * vmean

    # Recover mean in original gamma space
    warping_mean = scipy.integrate.cumtrapz(np.square(mu, out=mu)[0],
                                            x=eval_points, initial=0)

    # Affine traslation to original scale
    warping_mean = _normalize_scale(warping_mean,
                                    a=original_eval_points[0],
                                    b=original_eval_points[-1])

    monotone_interpolation = SplineInterpolation(interpolation_order=3,
                                                 monotone=True)

    mean = FDataGrid([warping_mean], grid_points=original_eval_points,
                     interpolation=monotone_interpolation)

    return mean


def elastic_mean(fdatagrid, *, penalty=0., center=True, max_iter=20, tol=1e-3,
                 initial=None, grid_dim=7, **kwargs):
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
        fdatagrid (:class:`~skfda.FDataGrid`): Set of functions to compute the
            mean.
        penalty (float): Penalisation term. Defaults to 0.
        center (boolean): If true it is computed the mean of the warpings and
            used to select a central mean. Defaults True.
        max_iter (int): Maximum number of iterations. Defaults to 20.
        tol (float): Convergence criterion, the algorithm will stop if
            :math:`|mu_{(\nu)} - mu_{(\nu - 1)}|_2 / | mu_{(\nu-1)} |_2 < tol`.
        initial (float): Value of the mean at the starting point. By default
            takes the average of the initial points of the samples.
        grid_dim (int, optional): Dimension of the grid used in the alignment
            algorithm. Defaults 7.
        ** kwargs : Named options to be pased to :func:`warping_mean`.

    Return:
        :class:`~skfda.FDataGrid`: FDatagrid with the mean of the functions.

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

    srsf_transformer = SRSF(initial_value=0)
    fdatagrid_srsf = srsf_transformer.fit_transform(fdatagrid)
    eval_points = fdatagrid.grid_points[0]

    eval_points_normalized = _normalize_scale(eval_points)
    y_scale = eval_points[-1] - eval_points[0]

    interpolation = SplineInterpolation(interpolation_order=3, monotone=True)

    # Discretisation points
    fdatagrid_normalized = FDataGrid(fdatagrid(eval_points) / y_scale,
                                     grid_points=eval_points_normalized)

    srsf = fdatagrid_srsf(eval_points)[..., 0]

    # Initialize with function closest to the L2 mean with the L2 distance
    centered = (srsf.T - srsf.mean(axis=0, keepdims=True).T).T

    distances = scipy.integrate.simps(np.square(centered, out=centered),
                                      eval_points_normalized, axis=1)

    # Initialization of iteration
    mu = srsf[np.argmin(distances)]
    mu_aux = np.empty(mu.shape)
    mu_1 = np.empty(mu.shape)

    # Main iteration
    for _ in range(max_iter):

        gammas = _elastic_alignment_array(
            mu, srsf, eval_points_normalized, penalty, grid_dim)
        gammas = FDataGrid(gammas, grid_points=eval_points_normalized,
                           interpolation=interpolation)

        fdatagrid_normalized = fdatagrid_normalized.compose(gammas)
        srsf = srsf_transformer.transform(
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

    srsf_transformer.set_params(initial_value=initial)

    # Karcher mean orbit in space L2/Gamma
    karcher_mean = srsf_transformer.inverse_transform(
        fdatagrid.copy(data_matrix=[mu], grid_points=eval_points,
                       sample_names=("Karcher mean",)))

    if center:
        # Gamma mean in Hilbert Sphere
        mean_normalized = warping_mean(gammas, **kwargs)

        gamma_mean = FDataGrid(_normalize_scale(
            mean_normalized.data_matrix[..., 0],
            a=eval_points[0],
            b=eval_points[-1]),
            grid_points=eval_points)

        gamma_inverse = invert_warping(gamma_mean)

        karcher_mean = karcher_mean.compose(gamma_inverse)

    # Return center of the orbit
    return karcher_mean

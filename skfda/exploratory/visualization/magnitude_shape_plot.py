"""Magnitude-Shape Plot Module.

This module contains the necessary functions to construct the Magnitude-Shape Plot.
First the directional outlingness is calculated and then, an outliers detection method is implemented.

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate
from sklearn.covariance import MinCovDet
from scipy.stats import f, variation
from numpy import linalg as la
from io import BytesIO
import scipy

from ... import FDataGrid
from skfda.exploratory.depth import modified_band_depth

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def directional_outlyingness(fdatagrid, depth_method=modified_band_depth,
                             dim_weights=None, pointwise_weights=None):
    r"""Computes the directional outlyingness of the functional data.

    Furthermore, it calculates both the mean and the variation of the
    directional outlyingness of the samples in the data set, which are also
    returned.

    The first one, the mean directional outlyingness, describes the relative
    position (including both distance and direction) of the samples on average
    to the center curve; its norm can be regarded as the magnitude outlyingness.

    The second one, the variation of the directional outlyingness, measures
    the change of the directional outlyingness in terms of both norm and
    direction across the whole design interval and can be regarded as the
    shape outlyingness.

    Firstly, the directional outlyingness is calculated as follows:

    .. math::
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) =
        \left\{\frac{1}{d\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right)} - 1\right\} \cdot \mathbf{v}(t)

    where :math:`\mathbf{X}` is a stochastic process with probability distribution
    :math:`F`, :math:`d` a depth function and :math:`\mathbf{v}(t) = \left\{\mathbf{X}(t) -
    \mathbf{Z}(t)\right\} / \lVert \mathbf{X}(t) - \mathbf{Z}(t) \rVert`
    is the spatial sign of :math:`\left\{\mathbf{X}(t) - \mathbf{Z}(t)\right\}`,
    :math:`\mathbf{Z}(t)` denotes the median and ∥ · ∥ denotes the :math:`L_2` norm.

    From the above formula, we define the mean directional outlyingness as:

    .. math::
        \mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I
        \mathbf{O}\left(\mathbf{X}(t) , F_{\mathbf{X}(t)}\right) \cdot w(t) dt ;

    and the variation of the directional outlyingness as:

    .. math::
        VO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \int_I  \lVert\mathbf{O}\left(\mathbf{X}(t) ,
        F_{\mathbf{X}(t)}\right)-\mathbf{MO}\left(\mathbf{X} , F_{\mathbf{X}}\right)  \rVert^2 \cdot w(t) dt

    where :math:`w(t)` a weight function defined on the domain of :math:`\mathbf{X}`, :math:`I`.

    Then, the total functional outlyingness can be computed using these values:

    .. math::
        FO\left(\mathbf{X} , F_{\mathbf{X}}\right) = \lVert \mathbf{MO}\left(\mathbf{X} ,
        F_{\mathbf{X}}\right)\rVert^2 +  VO\left(\mathbf{X} , F_{\mathbf{X}}\right) .

    Args:
        fdatagrid (FDataGrid): Object containing the samples to be ordered according to
            the directional outlyingness.
        depth_method (:ref:`depth measure <depth-measures>`, optional): Method used to
            order the data. Defaults to :func:`modified band depth <fda.depth_measures.modified_band_depth>`.
        dim_weights (array_like, optional): an array containing the weights of each of
            the dimensions of the image. Defaults to the same weight for each of the
            dimensions: 1/ndim_image.
        pointwise_weights (array_like, optional): an array containing the weights of each
            point of discretisation where values have been recorded. Defaults to the same
            weight for each of the points: 1/len(interval).

    Returns:
        (tuple): tuple containing:

            dir_outlyingness (numpy.array((fdatagrid.shape))): List containing
            the values of the directional outlyingness of the FDataGrid object.

            mean_dir_outl (numpy.array((fdatagrid.nsamples, 2))): List containing
            the values of the magnitude outlyingness for each of the samples.

            variation_dir_outl (numpy.array((fdatagrid.nsamples,))): List
            containing the values of the shape outlyingness for each of the samples.

    Example:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> directional_outlyingness(fd)
        (array([[[ 1. ],
                [ 1. ],
                [ 1. ],
                [ 1. ],
                [ 1. ],
                [ 1. ]],
        <BLANKLINE>
               [[ 0. ],
                [ 0. ],
                [ 0. ],
                [ 0. ],
                [ 0. ],
                [ 0. ]],
        <BLANKLINE>
               [[-1. ],
                [-1. ],
                [-0.2],
                [-0.2],
                [-0.2],
                [-0.2]],
        <BLANKLINE>
               [[-0.2],
                [-0.2],
                [-0.2],
                [-1. ],
                [-1. ],
                [-1. ]]]), array([[ 1.66666667],
               [ 0.        ],
               [-0.73333333],
               [-1.        ]]), array([ 0.74074074,  0.        ,  0.36740741,  0.53333333]))


    """

    if fdatagrid.ndim_domain > 1:
        raise NotImplementedError("Only support 1 dimension on the domain.")

    if dim_weights is not None and (len(
            dim_weights) != fdatagrid.ndim_image or dim_weights.sum() != 1):
        raise ValueError(
            "There must be a weight in dim_weights for each dimension of the "
            "image and altogether must sum 1.")

    if pointwise_weights is not None and (len(
            pointwise_weights) != fdatagrid.ncol or pointwise_weights.sum() != 1):
        raise ValueError(
            "There must be a weight in pointwise_weights for each recorded time "
            "point and altogether must sum 1.")

    depth, depth_pointwise = depth_method(fdatagrid, pointwise=True)

    if dim_weights is None:
        dim_weights = np.ones(fdatagrid.ndim_image) / fdatagrid.ndim_image

    if pointwise_weights is None:
        pointwise_weights = np.ones(fdatagrid.ncol) / fdatagrid.ncol

    # Calculation of the depth of each multivariate sample with the
    # corresponding weight.
    weighted_depth = depth * dim_weights
    sample_depth = weighted_depth.sum(axis=-1)

    # Obtaining the median sample Z, to caculate v(t) = {X(t) − Z(t)}/∥ X(t) − Z(t)∥
    median_index = np.argmax(sample_depth)
    median = fdatagrid.data_matrix[median_index]
    v = fdatagrid.data_matrix - median
    v_norm = la.norm(v, axis=-1, keepdims=True)
    # To avoid ZeroDivisionError, the zeros are substituted by ones.
    v_norm[np.where(v_norm == 0)] = 1
    v_unitary = v / v_norm

    # Calculation of the depth of each point of each sample with
    # the corresponding weight.
    weighted_depth_pointwise = depth_pointwise * dim_weights
    sample_depth_pointwise = weighted_depth_pointwise.sum(axis=-1,
                                                          keepdims=True)

    # Calcuation directinal outlyingness
    dir_outlyingness = (1 / sample_depth_pointwise - 1) * v_unitary

    # Calcuation mean directinal outlyingness
    pointwise_weights_1 = np.tile(pointwise_weights,
                                  (fdatagrid.ndim_image, 1)).T
    weighted_dir_outlyingness = dir_outlyingness * pointwise_weights_1
    mean_dir_outl = scipy.integrate.simps(weighted_dir_outlyingness,
                                          fdatagrid.sample_points[0], axis = 1)

    # Calcuation variation directinal outlyingness
    mean_dir_outl_pointwise = np.repeat(mean_dir_outl, fdatagrid.ncol,
                                        axis=0).reshape(fdatagrid.shape)
    norm = np.square(
        la.norm(dir_outlyingness - mean_dir_outl_pointwise, axis=-1))
    weighted_norm = norm * pointwise_weights
    variation_dir_outl = scipy.integrate.simps(weighted_norm,
                                          fdatagrid.sample_points[0], axis=1)

    return dir_outlyingness, mean_dir_outl, variation_dir_outl

class MagnitudeShapePlot:
    r"""Implementation of the magnitude-shape plot

    This plot, which is based on the calculation of the
    :func:`directional outlyingness <fda.magnitude_shape_plot.directional_outlyingness>`
    of each of the samples, serves as a visualization tool for the centrality
    of curves. Furthermore, an outlier detection procedure is included.

    The norm of the mean of the directional outlyingness (:math:`\lVert\mathbf{MO}\rVert`)
    is plotted in the x-axis, and the variation of the directional outlyingness (:math:`VO`)
    in the y-axis.

    Considering :math:`\mathbf{Y} = \left(\mathbf{MO}^T, VO\right)^T`, the outlier detection method
    is implemented as described below.

    First, the square robust Mahalanobis distance is calculated based on a
    sample of size :math:`h \leq fdatagrid.nsamples`:

    .. math::
        {RMD}^2\left( \mathbf{Y}, \mathbf{\tilde{Y}}^*_J\right) = \left(
        \mathbf{Y} - \mathbf{\tilde{Y}}^*_J\right)^T  {\mathbf{S}^*_J}^{-1}
        \left( \mathbf{Y} - \mathbf{\tilde{Y}}^*_J\right)

    where :math:`J` denotes the group of :math:`h` samples that minimizes the
    determinant of the corresponding covariance matrix, :math:`\mathbf{\tilde{Y}}^*_J
    = h^{-1}\sum_{i\in{J}}\mathbf{Y}_i` and :math:`\mathbf{S}^*_J
    = h^{-1}\sum_{i\in{J}}\left( \mathbf{Y}_i - \mathbf{\tilde{Y}}^*_J\right)
    \left( \mathbf{Y}_i - \mathbf{\tilde{Y}}^*_J\right)^T`. The
    sub-sample of size h controls the robustness of the method.

    Then, the tail of this distance distribution is approximated as follows:

    .. math::
        \frac{c\left(m − p\right)}{m\left(p + 1\right)}RMD^2\left(
        \mathbf{Y}, \mathbf{\tilde{Y}}^*_J\right)\sim F_{p+1, m-p}

    where :math:`p` is the dmension of the image, and :math:`c` and :math:`m`
    are parameters determining the degrees of freedom of the :math:`F`-distribution
    and the scaling factor.

    .. math::
        c = E \left[s^*_{jj}\right]

    where :math:`s^*_{jj}` are the diagonal elements of MCD and

    .. math::
        m = \frac{2}{CV^2}

    where :math:`CV` is the estimated coefficient of variation of the diagonal elements of the  MCD shape estimator.

    Finally, we choose a cutoff value to determine the outliers, C ,
    as the α quantile of :math:`F_{p+1, m-p}`. We set :math:`\alpha = 0.993`,
    which is used in the classical boxplot for detecting outliers under a normal
    distribution.

    Attributes:
        fdatagrid (FDataGrid): Object to be visualized.
        depth_method (:ref:`depth measure <depth-measures>`, optional): Method
            used to order the data. Defaults to :func:`modified band depth
            <fda.depth_measures.modified_band_depth>`.
        dim_weights (array_like, optional): an array containing the weights
            of each of the dimensions of the image.
        pointwise_weights (array_like, optional): an array containing the
            weights of each points of discretisation where values have been
            recorded.
        alpha(float, optional): Denotes the quantile to choose the cutoff
            value for detecting outliers Defaults to 0.993, which is used
            in the classical boxplot.
        points(numpy.ndarray): 2-dimensional matrix where each row
            contains the points plotted in the graph.
        outliers (1-D array, (fdatagrid.nsamples,)): Contains 1 or 0 to denote
            if a sample is an outlier or not, respecively.
        colormap(matplotlib.pyplot.LinearSegmentedColormap, optional): Colormap
            from which the colors of the plot are extracted. Defaults to 'seismic'.
        color (float, optional): Tone of the colormap in which the nonoutlier
            points are  plotted. Defaults to 0.2.
        outliercol (float, optional): Tone of the colormap in which the outliers
            are plotted. Defaults to 0.8.
        xlabel (string, optional): Label of the x-axis. Defaults to 'MO',
            mean of the  directional outlyingness.
        ylabel (string, optional): Label of the y-axis. Defaults to 'VO',
            variation of the  directional outlyingness.
        title (string, optional): Title of the plot. defaults to 'MS-Plot'.

    Example:

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5], [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points)
        >>> MagnitudeShapePlot(fd)
        MagnitudeShapePlot(
            FDataGrid=FDataGrid(
                array([[[ 1. ],
                        [ 1. ],
                        [ 2. ],
                        [ 3. ],
                        [ 2.5],
                        [ 2. ]],
        <BLANKLINE>
                       [[ 0.5],
                        [ 0.5],
                        [ 1. ],
                        [ 2. ],
                        [ 1.5],
                        [ 1. ]],
        <BLANKLINE>
                       [[-1. ],
                        [-1. ],
                        [-0.5],
                        [ 1. ],
                        [ 1. ],
                        [ 0.5]],
        <BLANKLINE>
                       [[-0.5],
                        [-0.5],
                        [-0.5],
                        [-1. ],
                        [-1. ],
                        [-1. ]]]),
                sample_points=[array([ 0,  2,  4,  6,  8, 10])],
                domain_range=array([[ 0, 10]]),
                dataset_label=None,
                axes_labels=None,
                extrapolation=None,
                interpolator=SplineInterpolator(interpolation_order=1, smoothness_parameter=0.0, monotone=False),
                keepdims=False),
            depth_method=modified_band_depth,
            dim_weights=None,
            pointwise_weights=None,
            alpha=0.993,
            points=array([[ 1.66666667,  0.74074074],
                   [ 0.        ,  0.        ],
                   [-0.73333333,  0.36740741],
                   [-1.        ,  0.53333333]]),
            outliers=array([0, 0, 0, 0]),
            colormap=seismic,
            color=0.2,
            outliercol=(0.8,),
            xlabel='MO',
            ylabel='VO',
            title='MS-Plot')
    """

    def __init__(self, fdatagrid, depth_method=modified_band_depth,
                 dim_weights=None, pointwise_weights=None, alpha=0.993,
                 assume_centered=False, support_fraction=None,random_state=0):
        """Initialization of the MagnitudeShapePlot class.

        Args:
            fdatagrid (FDataGrid): Object containing the data.
            depth_method (:ref:`depth measure <depth-measures>`, optional): Method
                used to order the data. Defaults to :func:`modified band depth
                <fda.depth_measures.modified_band_depth>`.
            dim_weights (array_like, optional): an array containing the weights
                of each of the dimensions of the image.
            pointwise_weights (array_like, optional): an array containing the
                weights of each points of discretisati on where values have been
                recorded.
            alpha (float, optional): Denotes the quantile to choose the cutoff
                value for detecting outliers Defaults to 0.993, which is used
                in the classical boxplot.
            assume_centered (boolean, optional): If True, the support of the robust
                location and the covariance estimates is computed, and a
                covariance estimate is recomputed from it, without centering
                the data. Useful to work with data whose mean is significantly
                equal to zero but is not exactly zero. If False, default value,
                the robust location and covariance are directly computed with
                the FastMCD algorithm without additional treatment.
            support_fraction (float, 0 < support_fraction < 1, optional): The proportion
                of points to be included in the support of the raw MCD estimate.
                Default is None, which implies that the minimum value of
                support_fraction will be used within the algorithm:
                [n_sample + n_features + 1] / 2
            random_state (int, RandomState instance or None, optional): If int,
                random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number
                generator; If None, the random number generator is the
                RandomState instance used by np.random. By default, it is 0.

        """

        if fdatagrid.ndim_image > 1:
            raise NotImplementedError("Only support 1 dimension on the image.")

        # The depths of the samples are calculated giving them an ordering.
        _, mean_dir_outl, variation_dir_outl = directional_outlyingness(
            fdatagrid,
            depth_method,
            dim_weights,
            pointwise_weights)

        points = np.array(list(zip(mean_dir_outl, variation_dir_outl))).astype(
            float)

        # The square mahalanobis distances of the samples are calulated using MCD.
        cov = MinCovDet(store_precision=False, assume_centered=assume_centered,
                        support_fraction=support_fraction,
                        random_state=random_state).fit(points)
        rmd_2 = cov.mahalanobis(points)

        # Calculation of the degrees of freedom of the F-distribution
        # (approximation of the tail of the distance distribution).
        s_jj = np.diag(cov.covariance_)
        c = np.mean(s_jj)
        m = 2 / np.square(variation(s_jj))
        p = fdatagrid.ndim_image
        dfn = p + 1
        dfd = m - p

        # Calculation of the cutoff value and scaling factor to identify outliers.
        cutoff_value = f.ppf(alpha, dfn, dfd, loc=0, scale=1)
        scaling = c * dfd / m / dfn
        outliers = (scaling * rmd_2 > cutoff_value) * 1

        self._fdatagrid = fdatagrid
        self._depth_method = depth_method
        self._dim_weights = dim_weights
        self._pointwise_weights = pointwise_weights
        self._alpha = alpha
        self._mean_dir_outl = mean_dir_outl
        self._variation_dir_outl = variation_dir_outl
        self._points = points
        self._outliers = outliers
        self._colormap = plt.cm.get_cmap('seismic')
        self._color = 0.2
        self._outliercol = 0.8,
        self.xlabel = 'MO'
        self.ylabel = 'VO'
        self.title = 'MS-Plot'

    @property
    def fdatagrid(self):
        return self._fdatagrid

    @property
    def depth_method(self):
        return self._depth_method

    @property
    def dim_weights(self):
        return self._dim_weights

    @property
    def pointwise_weights(self):
        return self._pointwise_weights

    @property
    def alpha(self):
        return self._alpha

    @property
    def points(self):
        return self._points

    @property
    def outliers(self):
        return self._outliers

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if not isinstance(value, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type matplotlib.colors.LinearSegmentedColormap")
        self._colormap = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                "color must be a number between 0 and 1.")

        self._color = value

    @property
    def outliercol(self):
        return self._outliercol

    @outliercol.setter
    def outliercol(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                "outcol must be a number between 0 and 1.")
        self._outliercol = value

    def plot(self, ax=None):
        """Visualization of the magnitude shape plot of the fdatagrid.

        Args:
            ax (axes object, optional): axes over where the graph is plotted.
                Defaults to matplotlib current axis.

        Returns:
            fig (figure object): figure object in which the graph is plotted.
            ax (axes object): axes in which the graph is plotted.

        """
        colors = np.zeros((self.fdatagrid.nsamples, 4))
        colors[np.where(self.outliers == 1)] = self.colormap(self.outliercol)
        colors[np.where(self.outliers == 0)] = self.colormap(self.color)

        if ax is None:
            ax = matplotlib.pyplot.gca()

        colors_rgba = [tuple(i) for i in colors]
        ax.scatter(self._mean_dir_outl, self._variation_dir_outl,
                   color=colors_rgba)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)

        return ax.get_figure(), ax

    def __repr__(self):
        """Return repr(self)."""
        return (f"MagnitudeShapePlot("
                f"\nFDataGrid={repr(self.fdatagrid)},"
                f"\ndepth_method={self.depth_method.__name__},"
                f"\ndim_weights={repr(self.dim_weights)},"
                f"\npointwise_weights={repr(self.pointwise_weights)},"
                f"\nalpha={repr(self.alpha)},"
                f"\npoints={repr(self.points)},"
                f"\noutliers={repr(self.outliers)},"
                f"\ncolormap={self.colormap.name},"
                f"\ncolor={repr(self.color)},"
                f"\noutliercol={repr(self.outliercol)},"
                f"\nxlabel={repr(self.xlabel)},"
                f"\nylabel={repr(self.ylabel)},"
                f"\ntitle={repr(self.title)})").replace('\n', '\n    ')

    def _repr_svg_(self):
        plt.figure()
        fig, _ = self.plot()
        output = BytesIO()
        fig.savefig(output, format='svg')
        data = output.getvalue()
        plt.close(fig)
        return data.decode('utf-8')

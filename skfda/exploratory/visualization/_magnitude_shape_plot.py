"""Magnitude-Shape Plot Module.

This module contains the necessary functions to construct the Magnitude-Shape
Plot. First the directional outlingness is calculated and then, an outliers
detection method is implemented.

"""

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from ..depth import modified_band_depth
from ..outliers import DirectionalOutlierDetector
from ._utils import _figure_to_svg, _get_figure_and_axes, _set_figure_layout


__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


class MagnitudeShapePlot:
    r"""Implementation of the magnitude-shape plot

    This plot, which is based on the calculation of the :func:`directional
    outlyingness <fda.magnitude_shape_plot.directional_outlyingness>`
    of each of the samples, serves as a visualization tool for the centrality
    of curves. Furthermore, an outlier detection procedure is included.

    The norm of the mean of the directional outlyingness (:math:`\lVert
    \mathbf{MO}\rVert`) is plotted in the x-axis, and the variation of the
    directional outlyingness (:math:`VO`) in the y-axis.

    The outliers are detected using an instance of
    :class:`DirectionalOutlierDetector`.

    Attributes:
        fdatagrid (FDataGrid): Object to be visualized.
        depth_method (:ref:`depth measure <depth-measures>`, optional): Method
            used to order the data. Defaults to :func:`modified band depth
            <fda.depth_measures.modified_band_depth>`.
        pointwise_weights (array_like, optional): an array containing the
            weights of each points of discretisation where values have been
            recorded.
        alpha(float, optional): Denotes the quantile to choose the cutoff
            value for detecting outliers Defaults to 0.993, which is used
            in the classical boxplot.
        points(numpy.ndarray): 2-dimensional matrix where each row
            contains the points plotted in the graph.
        outliers (1-D array, (fdatagrid.n_samples,)): Contains 1 or 0 to denote
            if a sample is an outlier or not, respecively.
        colormap(matplotlib.pyplot.LinearSegmentedColormap, optional): Colormap
            from which the colors of the plot are extracted. Defaults to
            'seismic'.
        color (float, optional): Tone of the colormap in which the nonoutlier
            points are  plotted. Defaults to 0.2.
        outliercol (float, optional): Tone of the colormap in which the
            outliers are plotted. Defaults to 0.8.
        xlabel (string, optional): Label of the x-axis. Defaults to 'MO',
            mean of the  directional outlyingness.
        ylabel (string, optional): Label of the y-axis. Defaults to 'VO',
            variation of the  directional outlyingness.
        title (string, optional): Title of the plot. defaults to 'MS-Plot'.

    Example:

        >>> import skfda
        >>> from skfda.exploratory.depth import modified_band_depth
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = skfda.FDataGrid(data_matrix, sample_points)
        >>> MagnitudeShapePlot(fd)
        MagnitudeShapePlot(
            FDataGrid=FDataGrid(
                array([[[ 1. ],
                        [ 1. ],
                        [ 2. ],
                        [ 3. ],
                        [ 2.5],
                        [ 2. ]],
                       [[ 0.5],
                        [ 0.5],
                        [ 1. ],
                        [ 2. ],
                        [ 1.5],
                        [ 1. ]],
                       [[-1. ],
                        [-1. ],
                        [-0.5],
                        [ 1. ],
                        [ 1. ],
                        [ 0.5]],
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
                interpolator=SplineInterpolator(interpolation_order=1,
                smoothness_parameter=0.0, monotone=False),
                keepdims=False),
            depth_method=projection_depth,
            pointwise_weights=None,
            alpha=0.993,
            points=array([[ 1.12415127,  0.05813094],
                          [ 0.        ,  0.        ],
                          [-0.53959261,  0.08037234],
                          [-1.17661166,  0.4294388 ]]),
            outliers=array([False, False, False, False]),
            colormap=seismic,
            color=0.2,
            outliercol=(0.8,),
            xlabel='MO',
            ylabel='VO',
            title='MS-Plot')
    """

    def __init__(self, fdatagrid, **kwargs):
        """Initialization of the MagnitudeShapePlot class.

        Args:
            fdatagrid (FDataGrid): Object containing the data.
            depth_method (:ref:`depth measure <depth-measures>`, optional):
                Method used to order the data. Defaults to :func:`projection
                depth <fda.depth_measures.multivariate.projection_depth>`.
            pointwise_weights (array_like, optional): an array containing the
                weights of each points of discretisati on where values have
                been recorded.
            alpha (float, optional): Denotes the quantile to choose the cutoff
                value for detecting outliers Defaults to 0.993, which is used
                in the classical boxplot.
            assume_centered (boolean, optional): If True, the support of the
                robust location and the covariance estimates is computed, and a
                covariance estimate is recomputed from it, without centering
                the data. Useful to work with data whose mean is significantly
                equal to zero but is not exactly zero. If False, default value,
                the robust location and covariance are directly computed with
                the FastMCD algorithm without additional treatment.
            support_fraction (float, 0 < support_fraction < 1, optional): The
                proportion of points to be included in the support of the
                raw MCD estimate.
                Default is None, which implies that the minimum value of
                support_fraction will be used within the algorithm:
                [n_sample + n_features + 1] / 2
            random_state (int, RandomState instance or None, optional): If int,
                random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number
                generator; If None, the random number generator is the
                RandomState instance used by np.random. By default, it is 0.

        """

        if fdatagrid.dim_codomain > 1:
            raise NotImplementedError(
                "Only support 1 dimension on the codomain.")

        self.outlier_detector = DirectionalOutlierDetector(**kwargs)

        y = self.outlier_detector.fit_predict(fdatagrid)

        outliers = (y == -1)

        self._fdatagrid = fdatagrid
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
        return self.outlier_detector.depth_method

    @property
    def pointwise_weights(self):
        return self.outlier_detector.pointwise_weights

    @property
    def alpha(self):
        return self.outlier_detector.alpha

    @property
    def points(self):
        return self.outlier_detector.points_

    @property
    def outliers(self):
        return self._outliers

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if not isinstance(value, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError("colormap must be of type "
                             "matplotlib.colors.LinearSegmentedColormap")
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

    def plot(self, chart=None, *, fig=None, axes=None,):
        """Visualization of the magnitude shape plot of the fdatagrid.

        Args:
            ax (axes object, optional): axes over where the graph is plotted.
                Defaults to matplotlib current axis.

        Returns:
            fig (figure object): figure object in which the graph is plotted.

        """

        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout(fig, axes)

        colors = np.zeros((self.fdatagrid.n_samples, 4))
        colors[np.where(self.outliers == 1)] = self.colormap(self.outliercol)
        colors[np.where(self.outliers == 0)] = self.colormap(self.color)

        colors_rgba = [tuple(i) for i in colors]
        axes[0].scatter(self.points[:, 0].ravel(), self.points[:, 1].ravel(),
                        color=colors_rgba)

        axes[0].set_xlabel(self.xlabel)
        axes[0].set_ylabel(self.ylabel)
        axes[0].set_title(self.title)

        return fig

    def __repr__(self):
        """Return repr(self)."""
        return (f"MagnitudeShapePlot("
                f"\nFDataGrid={repr(self.fdatagrid)},"
                f"\ndepth_method={self.depth_method.__name__},"
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
        fig = self.plot()
        return _figure_to_svg(fig)

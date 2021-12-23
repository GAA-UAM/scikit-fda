"""Magnitude-Shape Plot Module.

This module contains the necessary functions to construct the Magnitude-Shape
Plot. First the directional outlingness is calculated and then, an outliers
detection method is implemented.

"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from ... import FDataGrid
from ...representation._typing import NDArrayFloat, NDArrayInt
from ..depth import Depth
from ..outliers import MSPlotOutlierDetector
from ._baseplot import BasePlot


class MagnitudeShapePlot(BasePlot):
    r"""
    Implementation of the magnitude-shape plot.

    This plot, which is based on the calculation of the :func:`directional
    outlyingness <fda.magnitude_shape_plot.directional_outlyingness>`
    of each of the samples, serves as a visualization tool for the centrality
    of curves. Furthermore, an outlier detection procedure is included.

    The norm of the mean of the directional outlyingness (:math:`\lVert
    \mathbf{MO}\rVert`) is plotted in the x-axis, and the variation of the
    directional outlyingness (:math:`VO`) in the y-axis.

    The outliers are detected using an instance of
    :class:`MSPlotOutlierDetector`.

    For more information see :footcite:ts:`dai+genton_2018_visualization`.

    Args:
        fdata (FDataGrid): Object containing the data.
        multivariate_depth (:ref:`depth measure <depth-measures>`, optional):
            Method used to order the data. Defaults to :class:`projection
            depth <fda.depth_measures.multivariate.ProjectionDepth>`.
        pointwise_weights (array_like, optional): an array containing the
            weights of each points of discretisation where values have
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

    Attributes:
        points(numpy.ndarray): 2-dimensional matrix where each row
            contains the points plotted in the graph.
        outliers (1-D array, (fdata.n_samples,)): Contains 1 or 0 to denote
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

    Representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.datasets import make_gaussian_process
        from skfda.misc.covariances import Exponential
        from skfda.exploratory.visualization import MagnitudeShapePlot

        fd = make_gaussian_process(
                n_samples=20, cov=Exponential(), random_state=1)

        MagnitudeShapePlot(fd)

    Example:
        >>> import skfda
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [ 0., 2., 4., 6., 8., 10.]
        >>> fd = skfda.FDataGrid(data_matrix, grid_points)
        >>> MagnitudeShapePlot(fd)
        MagnitudeShapePlot(
            fdata=FDataGrid(
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
                grid_points=(array([  0.,   2.,   4.,   6.,   8.,  10.]),),
                domain_range=((0.0, 10.0),),
                ...),
            multivariate_depth=None,
            pointwise_weights=None,
            alpha=0.993,
            points=array([[ 1.66666667,  0.12777778],
                          [ 0.        ,  0.        ],
                          [-0.8       ,  0.17666667],
                          [-1.74444444,  0.94395062]]),
            outliers=array([False, False, False, False]),
            colormap=seismic,
            color=0.2,
            outliercol=0.8,
            xlabel='MO',
            ylabel='VO',
            title='MS-Plot')

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        fdata: FDataGrid,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
        **kwargs: Any,
    ) -> None:

        BasePlot.__init__(
            self,
            chart,
            fig=fig,
            axes=axes,
        )
        if fdata.dim_codomain > 1:
            raise NotImplementedError(
                "Only support 1 dimension on the codomain.")

        self.outlier_detector = MSPlotOutlierDetector(**kwargs)

        y = self.outlier_detector.fit_predict(fdata)

        outliers = (y == -1)

        self._fdata = fdata
        self._outliers = outliers
        self._colormap = plt.cm.get_cmap('seismic')
        self._color = 0.2
        self._outliercol = 0.8
        self.xlabel = 'MO'
        self.ylabel = 'VO'
        self.title = 'MS-Plot'

    @property
    def fdata(self) -> FDataGrid:
        return self._fdata

    @property
    def multivariate_depth(self) -> Optional[Depth[NDArrayFloat]]:
        return self.outlier_detector.multivariate_depth

    @property
    def pointwise_weights(self) -> Optional[NDArrayFloat]:
        return self.outlier_detector.pointwise_weights

    @property
    def alpha(self) -> float:
        return self.outlier_detector.alpha

    @property
    def points(self) -> NDArrayFloat:
        return self.outlier_detector.points_

    @property
    def outliers(self) -> NDArrayInt:
        return self._outliers

    @property
    def colormap(self) -> Colormap:
        return self._colormap

    @colormap.setter
    def colormap(self, value: Colormap) -> None:
        if not isinstance(value, matplotlib.colors.Colormap):
            raise ValueError(
                "colormap must be of type "
                "matplotlib.colors.Colormap",
            )
        self._colormap = value

    @property
    def color(self) -> float:
        return self._color

    @color.setter
    def color(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(
                "color must be a number between 0 and 1.")

        self._color = value

    @property
    def outliercol(self) -> float:
        return self._outliercol

    @outliercol.setter
    def outliercol(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(
                "outcol must be a number between 0 and 1.")
        self._outliercol = value

    @property
    def n_samples(self) -> int:
        return self.fdata.n_samples

    def _plot(
        self,
        fig: Figure,
        axes: Axes,
    ) -> None:

        self.artists = np.zeros(
            (self.n_samples, 1),
            dtype=Artist,
        )
        colors = np.zeros((self.fdata.n_samples, 4))
        colors[np.where(self.outliers == 1)] = self.colormap(self.outliercol)
        colors[np.where(self.outliers == 0)] = self.colormap(self.color)

        colors_rgba = [tuple(i) for i in colors]

        for i, _ in enumerate(self.points[:, 0].ravel()):
            self.artists[i, 0] = axes[0].scatter(
                self.points[:, 0].ravel()[i],
                self.points[:, 1].ravel()[i],
                color=colors_rgba[i],
                picker=True,
                pickradius=2,
            )

        axes[0].set_xlabel(self.xlabel)
        axes[0].set_ylabel(self.ylabel)
        axes[0].set_title(self.title)

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"MagnitudeShapePlot("
            f"\nfdata={repr(self.fdata)},"
            f"\nmultivariate_depth={self.multivariate_depth},"
            f"\npointwise_weights={repr(self.pointwise_weights)},"
            f"\nalpha={repr(self.alpha)},"
            f"\npoints={repr(self.points)},"
            f"\noutliers={repr(self.outliers)},"
            f"\ncolormap={self.colormap.name},"
            f"\ncolor={repr(self.color)},"
            f"\noutliercol={repr(self.outliercol)},"
            f"\nxlabel={repr(self.xlabel)},"
            f"\nylabel={repr(self.ylabel)},"
            f"\ntitle={repr(self.title)})"
        ).replace('\n', '\n    ')

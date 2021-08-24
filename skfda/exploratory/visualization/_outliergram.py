"""Outliergram Module.

This module contains the methods used to plot shapes in order to detect
shape outliers in our dataset. In order to do this, we plot the
Modified Band Depth and Modified Epigraph Index, that will help us detect
these outliers. The motivation of the method is that it is easy to find
magnitude outliers, but there is a necessity of capturing this other type.
"""

from typing import Optional, Sequence, Union

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ... import FDataGrid
from ..depth._depth import ModifiedBandDepth
from ..outliers import OutliergramOutlierDetector
from ..stats import modified_epigraph_index
from ._baseplot import BasePlot
from ._utils import _get_figure_and_axes, _set_figure_layout_for_fdata


class Outliergram(BasePlot):
    """
    Outliergram method of visualization.

    Plots the Modified Band Depth (MBD) on the Y axis and the Modified
    Epigraph Index (MEI) on the X axis. This points will create the form of
    a parabola. The shape outliers will be the points that appear far from
    this curve.
    Args:
        fdata: functional data set that we want to examine.
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        axes: axis where the graphs are plotted. If None, see param fig.
        n_rows: designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
        n_cols: designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
    Attributes:
        mbd: result of the calculation of the Modified Band Depth on our
            dataset. Represents the mean time a curve stays between other pair
            of curves, being a good measure of centrality.
        mei: result of the calculation of the Modified Epigraph Index on our
            dataset. Represents the mean time a curve stays below other curve.
    References:
        López-Pintado S.,  Romo J.. (2011). A half-region depth for functional
        data, Computational Statistics & Data Analysis, volume 55
        (page 1679-1695).
        Arribas-Gil A., Romo J.. Shape outlier detection and visualization for
        functional data: the outliergram
        https://academic.oup.com/biostatistics/article/15/4/603/266279
    """

    def __init__(
        self,
        fdata: FDataGrid,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        factor: float = 1.5,
        **kwargs,
    ) -> None:
        BasePlot.__init__(self)
        self.fdata = fdata
        self.factor = factor
        self.outlier_detector = OutliergramOutlierDetector(factor=factor)
        self.outlier_detector.fit(fdata)
        indices = np.argsort(self.outlier_detector.mei_)
        self._parabola_ordered = self.outlier_detector.parabola_[indices]
        self._mei_ordered = self.outlier_detector.mei_[indices]
        self._set_figure_and_axes(chart, fig, axes, n_rows, n_cols)

    def plot(
        self,
    ) -> Figure:
        """
        Plot Outliergram.

        Plots the Modified Band Depth (MBD) on the Y axis and the Modified
        Epigraph Index (MEI) on the X axis. This points will create the form of
        a parabola. The shape outliers will be the points that appear far from
        this curve.
        Returns:
            fig: figure object in which the depths will be
            scattered.
        """
        self.artists = np.zeros(
            (self.n_samples(), 1),
            dtype=Artist,
        )
        self.axScatter = self.axes[0]

        for i, (mei, mbd) in enumerate(
            zip(self.outlier_detector.mei_, self.outlier_detector.mbd_),
        ):
            self.artists[i, 0] = self.axScatter.scatter(
                mei,
                mbd,
                picker=2,
            )

        self.axScatter.plot(
            self._mei_ordered,
            self._parabola_ordered,
        )

        shifted_parabola = (
            self._parabola_ordered
            - self.outlier_detector.max_inlier_distance_
        )

        self.axScatter.plot(
            self._mei_ordered,
            shifted_parabola,
            linestyle='dashed',
        )

        # Set labels of graph
        if self.fdata.dataset_name is not None:
            self.axScatter.set_title(self.fdata.dataset_name)

        self.axScatter.set_xlabel("MEI")
        self.axScatter.set_ylabel("MBD")
        self.axScatter.set_xlim([0, 1])
        self.axScatter.set_ylim([
            0,  # Minimum MBD
            1,  # Maximum MBD
        ])

        return self.fig

    def n_samples(self) -> int:
        """Get the number of instances that will be used for interactivity."""
        return self.fdata.n_samples

    def _set_figure_and_axes(
        self,
        chart: Union[Figure, Axes, None] = None,
        fig: Optional[Figure] = None,
        axes: Union[Axes, Sequence[Axes], None] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
    ) -> None:
        """
        Initialize the axes and fig of the plot.

        Args:
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        fig: figure over with the graphs are plotted in case ax is not
            specified. If None and ax is also None, the figure is
            initialized.
        axes: axis where the graphs are plotted. If None, see param fig.
        n_rows: designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols: designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.
        """
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout_for_fdata(
            fdata=self.fdata,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        self.fig = fig
        self.axes = axes

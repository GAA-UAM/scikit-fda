"""Outliergram Module.

This module contains the methods used to plot shapes in order to detect
shape outliers in our dataset. In order to do this, we plot the
Modified Band Depth and Modified Epigraph Index, that will help us detect
these outliers. The motivation of the method is that it is easy to find
magnitude outliers, but there is a necessity of capturing this other type.
"""
from __future__ import annotations

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...representation import FDataGrid
from ..outliers import OutliergramOutlierDetector
from ._baseplot import BasePlot


class Outliergram(BasePlot):
    """
    Outliergram method of visualization.

    Plots the :class:`Modified Band Depth 
    (MBD)<skfda.exploratory.depth.ModifiedBandDepth>` on the Y axis and the
    :func:`Modified Epigraph Index
    (MEI)<skfda.exploratory.stats.modified_epigraph_index>` on the X axis.
    These points will create the form of a parabola.
    The shape outliers will be the points that appear far from this curve.

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
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Axes | None = None,
        factor: float = 1.5,
    ) -> None:
        BasePlot.__init__(
            self,
            chart,
            fig=fig,
            axes=axes,
        )
        self.fdata = fdata
        self.factor = factor
        self.outlier_detector = OutliergramOutlierDetector(factor=factor)
        self.outlier_detector.fit(fdata)
        indices = np.argsort(self.outlier_detector.mei_)
        self._parabola_ordered = self.outlier_detector.parabola_[indices]
        self._mei_ordered = self.outlier_detector.mei_[indices]

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

        for i, (mei, mbd) in enumerate(
            zip(self.outlier_detector.mei_, self.outlier_detector.mbd_),
        ):
            self.artists[i, 0] = axes[0].scatter(
                mei,
                mbd,
                picker=2,
            )

        axes[0].plot(
            self._mei_ordered,
            self._parabola_ordered,
        )

        shifted_parabola = (
            self._parabola_ordered
            - self.outlier_detector.max_inlier_distance_
        )

        axes[0].plot(
            self._mei_ordered,
            shifted_parabola,
            linestyle='dashed',
        )

        # Set labels of graph
        if self.fdata.dataset_name is not None:
            axes[0].set_title(self.fdata.dataset_name)

        axes[0].set_xlabel("MEI")
        axes[0].set_ylabel("MBD")
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([
            0,  # Minimum MBD
            1,  # Maximum MBD
        ])

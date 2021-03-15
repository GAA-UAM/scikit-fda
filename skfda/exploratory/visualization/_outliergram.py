"""Outliergram Module.

This module contains the methods used to plot shapes in order to detect
shape outliers in our dataset. In order to do this, we plot the
Modified Band Depth and Modified Epigraph Index, that will help us detect
this outliers. The motivation of the method is that it is easy to find
magnitude outliers, but there is a necessity of capturing this other type.
"""

from typing import Optional, Union

import numpy as np
import scipy.integrate as integrate
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rankdata

from ... import FDataGrid
from ..depth._depth import ModifiedBandDepth
from ._utils import _get_figure_and_axes, _set_figure_layout_for_fdata


class Outliergram:
    """
    Outliergram method of visualization.

    Plots the Modified Band Depth (MBD) on the Y axis and the Modified
    Epigraph Index (MEI) on the X axis. This points will create the form of
    a parabola. The shape outliers will be the points that appear far from
    this curve.
    Args:
        fdata: functional data set that we want to examine.
    Attributes:
        mbd: result of the calculation of the Modified Band Depth on our
            dataset. Represents the mean time a curve stays between all the
            possible pair of curves we have in our data set, being a good
            measure of centrality.
        mei: result of the calculation of the Modified Epigraph Index on our
            dataset. Represents the mean time a curve stays below each curve
            in our dataset.
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
    ) -> None:
        self.fdata = fdata
        self.depth = ModifiedBandDepth()
        self.depth.fit(fdata)
        self.mbd = self.depth(fdata)
        self.mei = self.modified_epigraph_index_list()

    def plot(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs,
    ) -> Figure:
        """
        Plot Outliergram.

        Plots the Modified Band Depth (MBD) on the Y axis and the Modified
        Epigraph Index (MEI) on the X axis. This points will create the form of
        a parabola. The shape outliers will be the points that appear far from
        this curve.
        Args:
            chart: figure over
                with the graphs are plotted or axis over where the graphs are
                plotted. If None and ax is also None, the figure is
                initialized.
            fig: figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            axes: axis where the graphs
                are plotted. If None, see param fig.
            n_rows: designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols: designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
            kwargs: if dim_domain is 1, keyword arguments to be passed to the
                matplotlib.pyplot.plot function; if dim_domain is 2, keyword
                arguments to be passed to the matplotlib.pyplot.plot_surface
                function.
        Returns:
            fig: figure object in which the depths will be
            scattered.
        """
        fig, axes_list = _get_figure_and_axes(chart, fig, axes)
        fig, axes_list = _set_figure_layout_for_fdata(
            self.fdata, fig, axes_list, n_rows, n_cols,
        )
        self.fig = fig
        self.axes = axes_list

        ax = self.axes[0]

        ax.scatter(
            self.mei,
            self.mbd,
            **kwargs,
        )

        # Set labels of graph
        fig.suptitle("Outliergram")
        ax.set_xlabel("MEI")
        ax.set_ylabel("MBD")
        ax.set_xlim([0, 1])
        ax.set_ylim([
            self.depth.min,
            self.depth.max,
        ])

        return fig

    def modified_epigraph_index_list(self) -> np.ndarray:
        """
        Calculate the Modified Epigraph Index of a FData.

        The MEI represents the mean time a curve stays below other curve.
        In this case we will calculate the MEI for each curve in relation
        with all the other curves of our dataset.
        """
        interval_len = (
            self.fdata.domain_range[0][1]
            - self.fdata.domain_range[0][0]
        )

        function = rankdata(
            -self.fdata.data_matrix,
            method='max',
            axis=0,
        ) - 1

        integrand = integrate.simps(
            function,
            x=self.fdata.grid_points[0],
            axis=1,
        )

        integrand /= (interval_len * self.fdata.n_samples)

        return integrand

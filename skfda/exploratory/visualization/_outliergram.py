"""Outliergram Module.

This module contains the methods used to plot shapes in order to detect
shape outliers in our dataset. In order to do this, we plot the
Modified Band Depth and Modified Epigraph Index, that will help us detect
this outliers. The motivation of the method is that it is easy to find
magnitude outliers, but there is a necessity of capturing this other type.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy.integrate as integrate
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rankdata

from ... import FDataGrid
from ..depth._depth import ModifiedBandDepth
from ._display import Display
from ._utils import (
    _get_figure_and_axes,
    _set_figure_layout_for_fdata,
)


class Outliergram(Display):
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
    ) -> None:
        Display.__init__(self)
        self.fdata = fdata
        self.depth = ModifiedBandDepth()
        self.depth.fit(fdata)
        self.mbd = self.depth(fdata)
        self.mei = self.modified_epigraph_index_list()
        if self.mbd.size != self.mei.size:
            raise ValueError(
                "The size of mbd and mei should be the same.",
            )

    def plot(
        self,
        chart: Union[Figure, Axes, None] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[Sequence[Axes]] = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Plot Outliergram.

        Plots the Modified Band Depth (MBD) on the Y axis and the Modified
        Epigraph Index (MEI) on the X axis. This points will create the form of
        a parabola. The shape outliers will be the points that appear far from
        this curve.
        Args:
            chart (figure object, axe or list of axes, optional): figure over
                with the graphs are plotted or axis over where the graphs are
                plotted. If None and ax is also None, the figure is
                initialized.
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            axes (list of axis objects, optional): axis where the graphs
                are plotted. If None, see param fig.
            n_rows (int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols(int, optional): designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.
            interactivity_mode (bool): if this is activated it will plot an
                aditional graph representing the functions of the functional
                data, that will allow to click and highlight the points
                represented.
            kwargs: if dim_domain is 1, keyword arguments to be passed to the
                matplotlib.pyplot.plot function; if dim_domain is 2, keyword
                arguments to be passed to the matplotlib.pyplot.plot_surface
                function.
        Returns:
            fig (figure object): figure object in which the depths will be
            scattered.
        """
        self.id_function = []

        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout_for_fdata(
            fdata=self.fdata,
            fig=fig,
            axes=axes,
        )
        self.fig = fig
        self.axes = axes

        self.axScatter = self.axes[0]

        for i in range(self.mei.size):
            self.id_function.append(self.axScatter.scatter(
                self.mei[i],
                self.mbd[i],
                picker=2,
            ))

        # Set labels of graph
        self.axScatter.set_title("Outliergram")
        self.axScatter.set_xlabel("MEI")
        self.axScatter.set_ylabel("MBD")
        self.axScatter.set_xlim([0, 1])
        self.axScatter.set_ylim([
            self.depth.min,
            self.depth.max,
        ])

        return self.fig

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

    def num_instances(self) -> int:
        return self.fdata.n_samples

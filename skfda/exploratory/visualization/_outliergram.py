"""Outliergram Module.

This module contains the methods used to plot shapes in order to detect
shape outliers in our dataset. In order to do this, we plot the
Modified Band Depth and Modified Epigraph Index, that will help us detect
this outliers. The motivation of the method is that it is easy to find
magnitude outliers, but there is a necessity of capturing this other type.
"""

from typing import List, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rankdata

from ... import FDataGrid
from ..depth._depth import ModifiedBandDepth
from ._utils import (
    _get_figure_and_axes,
    _set_figure_layout,
    _set_figure_layout_for_fdata,
)

S = TypeVar('S', Figure, Axes, List[Axes])


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
        self.fdata = fdata
        self.depth = ModifiedBandDepth()
        self.depth.fit(fdata)
        self.mbd = self.depth(fdata)
        self.mei = self.modified_epigraph_index_list()
        if self.mbd.size != self.mei.size:
            raise ValueError(
                "The size of mbd and mei should be the same."
            )

    def plot(
        self,
        chart: Optional[S] = None,
        *,
        fig: Optional[Figure] = None,
        axes: Optional[List[Axes]] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        interactivity_mode: bool = True,
        **kwargs,
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
        if interactivity_mode:
            self.point_clicked = None
            fig = plt.figure(figsize=(9, 3))
            fig, axes = _get_figure_and_axes(chart, fig, axes)
            fig, axes = _set_figure_layout(
                fig=fig, axes=axes, n_axes=2,
            )
        else:
            fig, axes = _get_figure_and_axes(chart, fig, axes)
            fig, axes = _set_figure_layout_for_fdata(
                fdata=self.fdata, fig=fig, axes=axes,
            )

        self.axScatter = axes[0]

        if interactivity_mode:
            self.id_function = []
            self.id_point = []
            self.axPlot = axes[1]
            for i in range(self.mei.size):
                self.id_function.append(self.axPlot.plot(
                    self.fdata.grid_points[0],
                    self.fdata.data_matrix[i].flatten(),
                ))
                self.id_point.append(self.axScatter.scatter(
                    self.mei[i],
                    self.mbd[i],
                    picker=2,
                ))

        else:
            self.axScatter.scatter(
                self.mei,
                self.mbd,
                **kwargs,
            )

        # Set labels of graph
        fig.suptitle("Outliergram")
        self.axScatter.set_xlabel("MEI")
        self.axScatter.set_ylabel("MBD")
        self.axScatter.set_xlim([0, 1])
        self.axScatter.set_ylim([
            self.depth.min,
            self.depth.max,
        ])

        fig.canvas.mpl_connect('pick_event', self.__call__)

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

    def __call__(self, event):
        if self.point_clicked is None:
            self.point_clicked = event.artist
            self.reduce_points_intensity()
        elif self.point_clicked == event.artist:
            self.restore_points_intensity()
            self.point_clicked = None
        else:
            self.change_points_intensity(event.artist)
            self.point_clicked = event.artist


    def reduce_points_intensity(self):
        for point, function in zip(self.id_point, self.id_function):
            if not (
                np.ma.getdata(point.get_offsets())[0][0]
                == np.ma.getdata(self.point_clicked.get_offsets())[0][0]
            ) or not (
                np.ma.getdata(point.get_offsets())[0][1]
                == np.ma.getdata(self.point_clicked.get_offsets())[0][1]
            ):
                point.set_alpha(0.5)
                function[0].set_alpha(0.1)


    def restore_points_intensity(self):
        for point, function in zip(self.id_point, self.id_function):
            point.set_alpha(1)
            function[0].set_alpha(1)
            

    def change_points_intensity(self, new_point):
        for point, function in zip(self.id_point, self.id_function):
            if (
                np.ma.getdata(point.get_offsets())[0][0]
                == np.ma.getdata(new_point.get_offsets())[0][0]
            ) and (
                np.ma.getdata(point.get_offsets())[0][1]
                == np.ma.getdata(new_point.get_offsets())[0][1]
            ):
                point.set_alpha(1)
                function[0].set_alpha(1)

            else:
                point.set_alpha(0.5)
                function[0].set_alpha(0.1)


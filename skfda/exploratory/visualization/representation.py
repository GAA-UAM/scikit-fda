"""Representation Module.

This module contains the functionality related
with plotting and scattering our different datasets.
It allows multiple modes and colors, which could
be set manually or automatically depending on values
like depth measures.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Sized, Tuple, TypeVar

import matplotlib.cm
import matplotlib.patches
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from typing_extensions import Protocol

from ..._utils import _to_grid_points, constants
from ...misc.validation import validate_domain_range
from ...representation._functional_data import FData
from ...typing._base import DomainRangeLike, GridPointsLike
from ._baseplot import BasePlot
from ._utils import ColorLike, _set_labels

K = TypeVar('K', contravariant=True)
V = TypeVar('V', covariant=True)


class Indexable(Protocol[K, V]):
    """Class Indexable used to type _get_color_info."""

    def __getitem__(self, __key: K) -> V:
        pass

    def __len__(self) -> int:
        pass


def _get_color_info(
    fdata: Sized,
    group: Sequence[K] | None = None,
    group_names: Indexable[K, str] | None = None,
    group_colors: Indexable[K, ColorLike] | None = None,
    legend: bool = False,
    kwargs: Dict[str, Any] | None = None,
) -> Tuple[
    Sequence[ColorLike] | None,
    Sequence[matplotlib.patches.Patch] | None,
]:

    if kwargs is None:
        kwargs = {}

    patches = None

    if group is not None:
        # In this case, each curve has a label, and all curves with the same
        # label should have the same color

        group_unique, group_indexes = np.unique(
            np.asarray(group),
            return_inverse=True,
        )
        n_labels = len(group_unique)

        if group_colors is not None:
            group_colors_array = np.array(
                [group_colors[g] for g in group_unique],
            )
        else:
            prop_cycle = matplotlib.rcParams['axes.prop_cycle']
            cycle_colors = prop_cycle.by_key()['color']

            group_colors_array = np.take(
                cycle_colors, np.arange(n_labels), mode='wrap',
            )

        sample_colors = list(group_colors_array[group_indexes])

        group_names_array = None

        if group_names is not None:
            group_names_array = np.array(
                [group_names[g] for g in group_unique],
            )
        elif legend is True:
            group_names_array = group_unique

        if group_names_array is not None:
            patches = [
                matplotlib.patches.Patch(color=c, label=l)
                for c, l in zip(group_colors_array, group_names_array)
            ]

    else:
        # In this case, each curve has a different color unless specified
        # otherwise

        if 'color' in kwargs:
            sample_colors = len(fdata) * [kwargs.get("color")]
            kwargs.pop('color')

        elif 'c' in kwargs:
            sample_colors = len(fdata) * [kwargs.get("c")]
            kwargs.pop('c')

        else:
            sample_colors = None

    return sample_colors, patches


class GraphPlot(BasePlot):
    """
    Class used to plot the FDataGrid object graph as hypersurfaces.

    When plotting functional data, we can either choose manually a color,
    a group of colors for the representations. Besides, we can use a list of
    variables (depths, scalar regression targets...) can be used as an
    argument to display the functions wtih a gradient of colors.

    Args:
        fdata: functional data set that we want to plot.
        gradient_criteria: list of real values used to determine the color
            in which each of the instances will be plotted.
        max_grad: maximum value that the gradient_list can take, it will be
            used to normalize the ``gradient_criteria`` in order to get values
            that can be used in the function colormap.__call__(). If not
            declared it will be initialized to the maximum value of
            gradient_list.
        min_grad: minimum value that the gradient_list can take, it will be
            used to normalize the ``gradient_criteria`` in order to get values
            that can be used in the function colormap.__call__(). If not
            declared it will be initialized to the minimum value of
            gradient_list.
        chart: figure over
            with the graphs are plotted or axis over where the graphs are
            plotted. If None and ax is also None, the figure is
            initialized.
        fig: figure over with the graphs are
            plotted in case ax is not specified. If None and ax is also
            None, the figure is initialized.
        axes: axis over where the graphs
            are plotted. If None, see param fig.
        n_rows: designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols: designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.
        n_points: Number of points to evaluate in
            the plot. In case of surfaces a tuple of length 2 can be pased
            with the number of points to plot in each axis, otherwise the
            same number of points will be used in the two axes. By default
            in unidimensional plots will be used 501 points; in surfaces
            will be used 30 points per axis, wich makes a grid with 900
            points.
        domain_range: Range where the
            function will be plotted. In objects with unidimensional domain
            the domain range should be a tuple with the bounds of the
            interval; in the case of surfaces a list with 2 tuples with
            the ranges for each dimension. Default uses the domain range
            of the functional object.
        group: contains integers from [0 to number of
            labels) indicating to which group each sample belongs to. Then,
            the samples with the same label are plotted in the same color.
            If None, the default value, each sample is plotted in the color
            assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
        group_colors: colors in which groups are
            represented, there must be one for each group. If None, each
            group is shown with distict colors in the "Greys" colormap.
        group_names: name of each of the groups which appear
            in a legend, there must be one for each one. Defaults to None
            and the legend is not shown. Implies `legend=True`.
        colormap: name of the colormap to be used. By default we will
            use autumn.
        legend: if `True`, show a legend with the groups. If
            `group_names` is passed, it will be used for finding the names
            to display in the legend. Otherwise, the values passed to
            `group` will be used.
        kwargs: if dim_domain is 1, keyword arguments to be passed to
            the matplotlib.pyplot.plot function; if dim_domain is 2,
            keyword arguments to be passed to the
            matplotlib.pyplot.plot_surface function.
    Attributes:
        gradient_list: normalization of the values from gradient color_list
            that will be used to determine the intensity of the color
            each function will have.
    """

    def __init__(
        self,
        fdata: FData,
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Axes | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
        n_points: int | Tuple[int, int] | None = None,
        domain_range: DomainRangeLike | None = None,
        group: Sequence[K] | None = None,
        group_colors: Indexable[K, ColorLike] | None = None,
        group_names: Indexable[K, str] | None = None,
        gradient_criteria: Sequence[float] | None = None,
        max_grad: float | None = None,
        min_grad: float | None = None,
        colormap: Colormap | str | None = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chart,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        self.fdata = fdata
        self.gradient_criteria = gradient_criteria
        if self.gradient_criteria is not None:
            if len(self.gradient_criteria) != fdata.n_samples:
                raise ValueError(
                    "The length of the gradient color",
                    "list should be the same as the number",
                    "of samples in fdata",
                )

            if min_grad is None:
                self.min_grad = min(self.gradient_criteria)
            else:
                self.min_grad = min_grad

            if max_grad is None:
                self.max_grad = max(self.gradient_criteria)
            else:
                self.max_grad = max_grad

            self.gradient_list: Sequence[float] | None = (
                [
                    (grad_color - self.min_grad)
                    / (self.max_grad - self.min_grad)
                    for grad_color in self.gradient_criteria
                ]
            )
        else:
            self.gradient_list = None

        self.n_points = n_points
        self.group = group
        self.group_colors = group_colors
        self.group_names = group_names
        self.legend = legend
        self.colormap = colormap
        self.kwargs = kwargs

        if domain_range is None:
            self.domain_range = self.fdata.domain_range
        else:
            self.domain_range = validate_domain_range(domain_range)

        if self.gradient_list is None:
            sample_colors, patches = _get_color_info(
                self.fdata,
                self.group,
                self.group_names,
                self.group_colors,
                self.legend,
                kwargs,
            )
        else:
            patches = None
            if self.colormap is None:
                colormap = matplotlib.cm.get_cmap("autumn")
                colormap = colormap.reversed()
            else:
                colormap = matplotlib.cm.get_cmap(self.colormap)

            sample_colors = colormap(self.gradient_list)

        self.sample_colors = sample_colors
        self.patches = patches

    @property
    def dim(self) -> int:
        return self.fdata.dim_domain + 1

    @property
    def n_subplots(self) -> int:
        return self.fdata.dim_codomain

    @property
    def n_samples(self) -> int:
        return self.fdata.n_samples

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:

        self.artists = np.zeros(
            (self.n_samples, self.fdata.dim_codomain),
            dtype=Artist,
        )

        color_dict: Dict[str, ColorLike | None] = {}

        if self.fdata.dim_domain == 1:

            if self.n_points is None:
                self.n_points = constants.N_POINTS_UNIDIMENSIONAL_PLOT_MESH

            assert isinstance(self.n_points, int)

            # Evaluates the object in a linspace
            eval_points = np.linspace(*self.domain_range[0], self.n_points)
            mat = self.fdata(eval_points)

            for i in range(self.fdata.dim_codomain):
                for j in range(self.fdata.n_samples):

                    set_color_dict(self.sample_colors, j, color_dict)

                    self.artists[j, i] = axes[i].plot(
                        eval_points,
                        mat[j, ..., i].T,
                        **self.kwargs,
                        **color_dict,
                    )[0]

        else:

            # Selects the number of points
            if self.n_points is None:
                n_points_tuple = 2 * (constants.N_POINTS_SURFACE_PLOT_AX,)
            elif isinstance(self.n_points, int):
                n_points_tuple = (self.n_points, self.n_points)
            elif len(self.n_points) != 2:
                raise ValueError(
                    "n_points should be a number or a tuple of "
                    "length 2, and has "
                    "length {0}.".format(len(self.n_points)),
                )

            # Axes where will be evaluated
            x = np.linspace(*self.domain_range[0], n_points_tuple[0])
            y = np.linspace(*self.domain_range[1], n_points_tuple[1])

            # Evaluation of the functional object
            Z = self.fdata((x, y), grid=True)

            X, Y = np.meshgrid(x, y, indexing='ij')

            for k in range(self.fdata.dim_codomain):
                for h in range(self.fdata.n_samples):

                    set_color_dict(self.sample_colors, h, color_dict)

                    self.artists[h, k] = axes[k].plot_surface(
                        X,
                        Y,
                        Z[h, ..., k],
                        **self.kwargs,
                        **color_dict,
                    )

        _set_labels(self.fdata, fig, axes, self.patches)


class ScatterPlot(BasePlot):
    """
    Class used to scatter the FDataGrid object.

    Args:
        fdata: functional data set that we want to plot.
        grid_points: points to plot.
        chart: figure over
            with the graphs are plotted or axis over where the graphs are
            plotted. If None and ax is also None, the figure is
            initialized.
        fig: figure over with the graphs are
            plotted in case ax is not specified. If None and ax is also
            None, the figure is initialized.
        axes: axis over where the graphs
            are plotted. If None, see param fig.
        n_rows: designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols: designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.
        domain_range: Range where the
            function will be plotted. In objects with unidimensional domain
            the domain range should be a tuple with the bounds of the
            interval; in the case of surfaces a list with 2 tuples with
            the ranges for each dimension. Default uses the domain range
            of the functional object.
        group: contains integers from [0 to number of
            labels) indicating to which group each sample belongs to. Then,
            the samples with the same label are plotted in the same color.
            If None, the default value, each sample is plotted in the color
            assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
        group_colors: colors in which groups are
            represented, there must be one for each group. If None, each
            group is shown with distict colors in the "Greys" colormap.
        group_names: name of each of the groups which appear
            in a legend, there must be one for each one. Defaults to None
            and the legend is not shown. Implies `legend=True`.
        legend: if `True`, show a legend with the groups. If
            `group_names` is passed, it will be used for finding the names
            to display in the legend. Otherwise, the values passed to
            `group` will be used.
        kwargs: if dim_domain is 1, keyword arguments to be passed to
            the matplotlib.pyplot.plot function; if dim_domain is 2,
            keyword arguments to be passed to the
            matplotlib.pyplot.plot_surface function.
    """

    def __init__(
        self,
        fdata: FData,
        chart: Figure | Axes | None = None,
        *,
        fig: Figure | None = None,
        axes: Axes | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
        grid_points: GridPointsLike | None = None,
        domain_range: Tuple[int, int] | DomainRangeLike | None = None,
        group: Sequence[K] | None = None,
        group_colors: Indexable[K, ColorLike] | None = None,
        group_names: Indexable[K, str] | None = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chart,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        self.fdata = fdata

        if grid_points is None:
            # This can only be done for FDataGrid
            self.grid_points = self.fdata.grid_points
            self.evaluated_points = self.fdata.data_matrix
        else:
            self.grid_points = _to_grid_points(grid_points)
            self.evaluated_points = self.fdata(
                self.grid_points, grid=True,
            )

        self.domain_range = domain_range
        self.group = group
        self.group_colors = group_colors
        self.group_names = group_names
        self.legend = legend

        if self.domain_range is None:
            self.domain_range = self.fdata.domain_range
        else:
            self.domain_range = validate_domain_range(self.domain_range)

        sample_colors, patches = _get_color_info(
            self.fdata,
            self.group,
            self.group_names,
            self.group_colors,
            self.legend,
            kwargs,
        )
        self.sample_colors = sample_colors
        self.patches = patches

    @property
    def dim(self) -> int:
        return self.fdata.dim_domain + 1

    @property
    def n_subplots(self) -> int:
        return self.fdata.dim_codomain

    @property
    def n_samples(self) -> int:
        return self.fdata.n_samples

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:
        """
        Scatter FDataGrid object.

        Returns:
        fig: figure object in which the graphs are plotted.
        """
        self.artists = np.zeros(
            (self.n_samples, self.fdata.dim_codomain),
            dtype=Artist,
        )

        color_dict: Dict[str, ColorLike | None] = {}

        if self.fdata.dim_domain == 1:

            for i in range(self.fdata.dim_codomain):
                for j in range(self.fdata.n_samples):

                    set_color_dict(self.sample_colors, j, color_dict)

                    self.artists[j, i] = axes[i].scatter(
                        self.grid_points[0],
                        self.evaluated_points[j, ..., i].T,
                        **color_dict,
                        picker=True,
                        pickradius=2,
                    )

        else:

            X = self.fdata.grid_points[0]
            Y = self.fdata.grid_points[1]
            X, Y = np.meshgrid(X, Y)

            for k in range(self.fdata.dim_codomain):
                for h in range(self.fdata.n_samples):

                    set_color_dict(self.sample_colors, h, color_dict)

                    self.artists[h, k] = axes[k].scatter(
                        X,
                        Y,
                        self.evaluated_points[h, ..., k].T,
                        **color_dict,
                        picker=True,
                        pickradius=2,
                    )

        _set_labels(self.fdata, fig, axes, self.patches)


def set_color_dict(
    sample_colors: Any,
    ind: int,
    color_dict: Dict[str, ColorLike | None],
) -> None:
    """
    Auxiliary method used to update color_dict.

    Sets the new color of the color_dict
    thanks to sample colors and index.
    """
    if sample_colors is not None:
        color_dict["color"] = sample_colors[ind]

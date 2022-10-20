"""Functional Data Boxplot Module.

This module contains the classes to construct the functional data boxplot and
visualize it.

"""
from __future__ import annotations

import math
from abc import abstractmethod
from typing import Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from skfda.exploratory.depth.multivariate import Depth

from ...representation import FData, FDataGrid
from ...typing._numpy import NDArrayBool, NDArrayFloat
from ..depth import ModifiedBandDepth
from ..outliers import _envelopes
from ._baseplot import BasePlot
from ._utils import _set_labels


class FDataBoxplot(BasePlot):
    """
    Abstract class inherited by the Boxplot and SurfaceBoxplot classes.

    It the data of the functional boxplot or surface boxplot of a FDataGrid
    object, depending on the dimensions of the :term:`domain`, 1 or 2
    respectively.

    It forces to both classes, Boxplot and SurfaceBoxplot to conain at least
    the median, central and outlying envelopes and a colormap for their
    graphical representation, obtained calling the plot method.

    """

    @abstractmethod
    def __init__(
        self,
        chart: Figure | Axes | None = None,
        *,
        factor: float = 1.5,
        fig: Figure | None = None,
        axes: Axes | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
    ) -> None:
        if factor < 0:
            raise ValueError(
                "The number used to calculate the "
                "outlying envelope must be positive.",
            )

        super().__init__(
            chart,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        self._factor = factor

    @property
    def factor(self) -> float:
        return self._factor

    @property
    def fdatagrid(self) -> FDataGrid:
        pass

    @property
    def median(self) -> NDArrayFloat:
        pass

    @property
    def central_envelope(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        pass

    @property
    def non_outlying_envelope(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        pass

    @property
    def colormap(self) -> Colormap:
        return self._colormap

    @colormap.setter
    def colormap(self, value: Colormap) -> None:
        if not isinstance(value, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type "
                "matplotlib.colors.LinearSegmentedColormap",
            )
        self._colormap = value


class Boxplot(FDataBoxplot):
    r"""
    Representation of the functional boxplot.

    Class implementing the functionl boxplot which is an informative
    exploratory tool for visualizing functional data, as well as its
    generalization, the enhanced functional boxplot. Only supports 1
    dimensional :term:`domain` functional data.

    Based on the center outward ordering induced by a :ref:`depth measure
    <depth-measures>` for functional data, the descriptive statistics of a
    functional boxplot are: the envelope of the 50% central region, the median
    curve,and the maximum non-outlying envelope. In addition, outliers can be
    detected in a functional boxplot by the 1.5 times the 50% central region
    empirical rule, analogous to the rule for classical boxplots.

    For more information see :footcite:ts:`sun+genton_2011_boxplots`.

    Args:
        fdatagrid: Object containing the data.
        chart: figure over with the graphs are plotted or axis over
            where the graphs are plotted. If None and ax is also
            None, the figure is initialized.
        depth_method: Method used to order the data. Defaults to
            :func:`~skfda.exploratory.depth.ModifiedBandDepth`.
        prob: List with float numbers (in the range from 1 to 0) that
            indicate which central regions to represent.
            Defaults to (0.5,) which represents the 50% central region.
        factor: Number used to calculate the outlying envelope.
        fig: Figure over with the graphs are
            plotted in case ax is not specified. If None and ax is also
            None, the figure is initialized.
        axes: Axis over where the graphs
            are plotted. If None, see param fig.
        n_rows: Designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols: Designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.

    Attributes:
        fdatagrid: Object containing the data.
        median: Contains the median/s.
        central_envelope: Contains the central envelope/s.
        non_outlying_envelope: Contains the non-outlying envelope/s.
        colormap: Colormap from which the colors to represent the
            central regions are selected.
        envelopes: Contains the region envelopes.
        outliers: Contains the outliers.
        barcol: Color of the envelopes and vertical lines.
        outliercol: Color of the ouliers.
        mediancol: Color of the median.
        show_full_outliers: If False (the default) then only the part
            outside the box is plotted. If True, complete outling curves are
            plotted.

    Representation in a Jupyter notebook:

    .. jupyter-execute::

        from skfda.datasets import make_gaussian_process
        from skfda.misc.covariances import Exponential
        from skfda.exploratory.visualization import Boxplot

        fd = make_gaussian_process(
                n_samples=20, cov=Exponential(), random_state=3)

        Boxplot(fd)


    Examples:
        Function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> from skfda import FDataGrid
        >>> from skfda.exploratory.visualization import Boxplot
        >>>
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2],
        ...                [0.5, 0.5, 1, 2, 1.5, 1],
        ...                [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> grid_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, grid_points, dataset_name="dataset",
        ...                argument_names=["x_label"],
        ...                coordinate_names=["y_label"])
        >>> Boxplot(fd)
        Boxplot(
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
                grid_points=(array([ 0.,  2.,  4.,  6.,  8., 10.]),),
                domain_range=((0.0, 10.0),),
                dataset_name='dataset',
                argument_names=('x_label',),
                coordinate_names=('y_label',),
                ...),
            median=array([[ 0.5],
                          [ 0.5],
                          [ 1. ],
                          [ 2. ],
                          [ 1.5],
                          [ 1. ]]),
            central envelope=(array([[-1. ],
                                     [-1. ],
                                     [-0.5],
                                     [ 1. ],
                                     [ 1. ],
                                     [ 0.5]]), array([[ 0.5],
                                     [ 0.5],
                                     [ 1. ],
                                     [ 2. ],
                                     [ 1.5],
                                     [ 1. ]])),
            non-outlying envelope=(array([[-1. ],
                                          [-1. ],
                                          [-0.5],
                                          [ 1. ],
                                          [ 1. ],
                                          [ 0.5]]), array([[ 0.5],
                                          [ 0.5],
                                          [ 1. ],
                                          [ 2. ],
                                          [ 1.5],
                                          [ 1. ]])),
            envelopes=[(array([[-1. ],
                               [-1. ],
                               [-0.5],
                               [ 1. ],
                               [ 1. ],
                               [ 0.5]]), array([[ 0.5],
                               [ 0.5],
                               [ 1. ],
                               [ 2. ],
                               [ 1.5],
                               [ 1. ]]))],
            outliers=array([ True, False, False,  True]))

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        fdatagrid: FData,
        chart: Figure | Axes | None = None,
        *,
        depth_method: Depth[FDataGrid] | None = None,
        prob: Sequence[float] = (0.5,),
        factor: float = 1.5,
        fig: Figure | None = None,
        axes: Axes | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
    ):
        """Initialize the Boxplot class.

        Args:
            fdatagrid: Object containing the data.
            depth_method: Method used to order the data.
                Defaults to :func:`modified band depth
                <skfda.exploratory.depth.ModifiedBandDepth>`.
            prob: List with float numbers (in the
                range from 1 to 0) that indicate which central regions to
                represent.
                Defaults to [0.5] which represents the 50% central region.
            factor: Number used to calculate the outlying envelope.
            chart: figure over with the graphs are plotted or axis over
                where the graphs are plotted. If None and ax is also
                None, the figure is initialized.
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

        """
        super().__init__(
            chart,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
            factor=factor,
        )

        if fdatagrid.dim_domain != 1:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 1.",
            )

        if sorted(prob, reverse=True) != list(prob):
            raise ValueError(
                "Probabilities required to be in descending order.",
            )

        if min(prob) < 0 or max(prob) > 1:
            raise ValueError("Probabilities must be between 0 and 1.")

        if depth_method is None:
            depth_method = ModifiedBandDepth()
        depth = depth_method(fdatagrid)
        indices_descending_depth = (-depth).argsort(axis=0)

        # The median is the deepest curve
        median_fdata = fdatagrid[indices_descending_depth[0]]
        self._median = median_fdata.data_matrix[0, ...]

        # Central region and envelope must be computed for outlier detection
        central_region = _envelopes.compute_region(
            fdatagrid,
            indices_descending_depth,
            0.5,
        )
        self._central_envelope = _envelopes.compute_envelope(central_region)

        # Non-outlying envelope
        non_outlying_threshold = _envelopes.non_outlying_threshold(
            self._central_envelope,
            factor,
        )
        predicted_outliers = _envelopes.predict_outliers(
            fdatagrid,
            non_outlying_threshold,
        )
        inliers = fdatagrid[predicted_outliers == 0]
        self._non_outlying_envelope = _envelopes.compute_envelope(inliers)

        # Outliers
        self._outliers = _envelopes.predict_outliers(
            fdatagrid,
            self._non_outlying_envelope,
        )

        self._envelopes = [
            _envelopes.compute_envelope(
                _envelopes.compute_region(
                    fdatagrid,
                    indices_descending_depth,
                    p,
                ),
            )
            for p in prob
        ]

        self._fdatagrid = fdatagrid
        self._prob = prob
        self._colormap = plt.cm.get_cmap('RdPu')
        self.barcol = "blue"
        self.outliercol = "red"
        self.mediancol = "black"
        self._show_full_outliers = False

    @property
    def fdatagrid(self) -> FDataGrid:
        return self._fdatagrid

    @property
    def median(self) -> NDArrayFloat:
        return self._median  # type: ignore[no-any-return]

    @property
    def central_envelope(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        return self._central_envelope

    @property
    def non_outlying_envelope(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        return self._non_outlying_envelope

    @property
    def envelopes(self) -> Sequence[Tuple[NDArrayFloat, NDArrayFloat]]:
        return self._envelopes

    @property
    def outliers(self) -> NDArrayBool:
        return self._outliers

    @property
    def show_full_outliers(self) -> bool:
        return self._show_full_outliers

    @show_full_outliers.setter
    def show_full_outliers(self, boolean: bool) -> None:
        if not isinstance(boolean, bool):
            raise ValueError("show_full_outliers must be boolean type")
        self._show_full_outliers = boolean

    @property
    def n_subplots(self) -> int:
        return self.fdatagrid.dim_codomain

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:

        tones = np.linspace(0.1, 1.0, len(self._prob) + 1, endpoint=False)[1:]
        color = self.colormap(tones)

        if self.show_full_outliers:
            var_zorder = 1
        else:
            var_zorder = 4

        outliers = self.fdatagrid[self.outliers]

        grid_points = self.fdatagrid.grid_points[0]

        for m, ax in enumerate(axes):

            # Outliers
            for o in outliers:
                ax.plot(
                    grid_points,
                    o.data_matrix[0, :, m],
                    color=self.outliercol,
                    linestyle='--',
                    zorder=1,
                )

            for envelop, col in zip(self.envelopes, color):
                # central regions
                ax.fill_between(
                    grid_points,
                    envelop[0][..., m],
                    envelop[1][..., m],
                    facecolor=col,
                    zorder=var_zorder,
                )

            # outlying envelope
            ax.plot(
                grid_points,
                self.non_outlying_envelope[0][..., m],
                grid_points,
                self.non_outlying_envelope[1][..., m],
                color=self.barcol,
                zorder=4,
            )

            # central envelope
            ax.plot(
                grid_points,
                self.central_envelope[0][..., m],
                grid_points,
                self.central_envelope[1][..., m],
                color=self.barcol,
                zorder=4,
            )

            # vertical lines
            index = math.ceil(len(grid_points) / 2)
            x = grid_points[index]
            ax.plot(
                [x, x],
                [
                    self.non_outlying_envelope[0][..., m][index],
                    self.central_envelope[0][..., m][index],
                ],
                color=self.barcol,
                zorder=4,
            )
            ax.plot(
                [x, x],
                [
                    self.non_outlying_envelope[1][..., m][index],
                    self.central_envelope[1][..., m][index],
                ],
                color=self.barcol,
                zorder=4,
            )

            # median sample
            ax.plot(
                grid_points,
                self.median[..., m],
                color=self.mediancol,
                zorder=5,
            )

        _set_labels(self.fdatagrid, fig, axes)

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"Boxplot("
            f"\nFDataGrid={repr(self.fdatagrid)},"
            f"\nmedian={repr(self.median)},"
            f"\ncentral envelope={repr(self.central_envelope)},"
            f"\nnon-outlying envelope={repr(self.non_outlying_envelope)},"
            f"\nenvelopes={repr(self.envelopes)},"
            f"\noutliers={repr(self.outliers)})"
        ).replace('\n', '\n    ')


class SurfaceBoxplot(FDataBoxplot):
    r"""
    Representation of the surface boxplot.

    Class implementing the surface boxplot. Analogously to the functional
    boxplot, it is an informative exploratory tool for visualizing functional
    data with :term:`domain` dimension 2. Nevertheless, it does not implement
    the enhanced surface boxplot.

    Based on the center outward ordering induced by a
    :ref:`depth measure <depth-measures>`
    for functional data, it represents the envelope of the
    50% central region, the median curve, and the maximum non-outlying
    envelope :footcite:`sun+genton_2011_boxplots`.

    Args:
        fdatagrid: Object containing the data.
        method: Method
            used to order the data. Defaults to :class:`modified band depth
            <skfda.exploratory.depth.ModifiedBandDepth>`.
        prob: List with float numbers (in the
            range from 1 to 0) that indicate which central regions to
            represent.
            Defaults to [0.5] which represents the 50% central region.
        factor: Number used to calculate the outlying envelope.

    Attributes:
        fdatagrid: Object containing the data.
        median: contains
            the median/s.
        central_envelope: contains the central envelope/s.
        non_outlying_envelope: contains the non-outlying envelope/s.
        colormap: Colormap from
            which the colors to represent the central regions are selected.
        boxcol: Color of the box, which includes median and central
            envelope.
        outcol: Color of the outlying envelope.
        fig: Figure over with the graphs are
            plotted in case ax is not specified. If None and ax is also
            None, the figure is initialized.
        axes: Axis over where the graphs
            are plotted. If None, see param fig.
        n_rows: Designates the number of rows of the figure
            to plot the different dimensions of the image. Only specified
            if fig and ax are None.
        n_cols: Designates the number of columns of the
            figure to plot the different dimensions of the image. Only
            specified if fig and ax are None.

    Examples:
        Function :math:`f : \mathbb{R^2}\longmapsto\mathbb{R}`.

        >>> from skfda import FDataGrid
        >>> data_matrix = [[[[1], [0.7], [1]],
        ...                 [[4], [0.4], [5]]],
        ...                [[[2], [0.5], [2]],
        ...                 [[3], [0.6], [3]]]]
        >>> grid_points = [[2, 4], [3, 6, 8]]
        >>> fd = FDataGrid(data_matrix, grid_points, dataset_name="dataset",
        ...                argument_names=["x1_label", "x2_label"],
        ...                coordinate_names=["y_label"])
        >>> SurfaceBoxplot(fd)
        SurfaceBoxplot(
            FDataGrid=FDataGrid(
                array([[[[ 1. ],
                         [ 0.7],
                         [ 1. ]],
                        [[ 4. ],
                         [ 0.4],
                         [ 5. ]]],
                       [[[ 2. ],
                         [ 0.5],
                         [ 2. ]],
                        [[ 3. ],
                         [ 0.6],
                         [ 3. ]]]]),
                grid_points=(array([ 2., 4.]), array([ 3., 6., 8.])),
                domain_range=((2.0, 4.0), (3.0, 8.0)),
                dataset_name='dataset',
                argument_names=('x1_label', 'x2_label'),
                coordinate_names=('y_label',),
                extrapolation=None,
                ...),
            median=array([[[ 1. ],
                           [ 0.7],
                           [ 1. ]],
                          [[ 4. ],
                           [ 0.4],
                           [ 5. ]]]),
            central envelope=(array([[[ 1. ],
                                      [ 0.7],
                                      [ 1. ]],
                                     [[ 4. ],
                                      [ 0.4],
                                      [ 5. ]]]),
                              array([[[ 1. ],
                                      [ 0.7],
                                      [ 1. ]],
                                     [[ 4. ],
                                      [ 0.4],
                                      [ 5. ]]])),
            outlying envelope=(array([[[ 1. ],
                                       [ 0.7],
                                       [ 1. ]],
                                      [[ 4. ],
                                       [ 0.4],
                                       [ 5. ]]]),
                               array([[[ 1. ],
                                       [ 0.7],
                                       [ 1. ]],
                                      [[ 4. ],
                                       [ 0.4],
                                       [ 5. ]]])))

    References:
        .. footbibliography::

    """

    def __init__(
        self,
        fdatagrid: FDataGrid,
        chart: Figure | Axes | None = None,
        *,
        depth_method: Depth[FDataGrid] | None = None,
        factor: float = 1.5,
        fig: Figure | None = None,
        axes: Axes | None = None,
        n_rows: int | None = None,
        n_cols: int | None = None,
    ) -> None:

        super().__init__(
            chart,
            fig=fig,
            axes=axes,
            n_rows=n_rows,
            n_cols=n_cols,
            factor=factor,
        )

        if fdatagrid.dim_domain != 2:
            raise ValueError(
                "Class only supports FDataGrid with domain dimension 2.",
            )

        if depth_method is None:
            depth_method = ModifiedBandDepth()

        depth = depth_method(fdatagrid)
        indices_descending_depth = (-depth).argsort(axis=0)

        # The mean is the deepest curve
        self._median = fdatagrid.data_matrix[indices_descending_depth[0]]

        # Central region and envelope must be computed for outlier detection
        central_region = _envelopes.compute_region(
            fdatagrid,
            indices_descending_depth,
            0.5,
        )
        self._central_envelope = _envelopes.compute_envelope(central_region)

        # Non-outlying envelope
        non_outlying_threshold = _envelopes.non_outlying_threshold(
            self._central_envelope,
            factor,
        )
        predicted_outliers = _envelopes.predict_outliers(
            fdatagrid,
            non_outlying_threshold,
        )
        inliers = fdatagrid[predicted_outliers == 0]
        self._non_outlying_envelope = _envelopes.compute_envelope(inliers)

        self._fdatagrid = fdatagrid
        self.colormap = plt.cm.get_cmap('Greys')
        self._boxcol = 1.0
        self._outcol = 0.7

    @property
    def fdatagrid(self) -> FDataGrid:
        return self._fdatagrid

    @property
    def median(self) -> NDArrayFloat:
        return self._median  # type: ignore[no-any-return]

    @property
    def central_envelope(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        return self._central_envelope

    @property
    def non_outlying_envelope(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        return self._non_outlying_envelope

    @property
    def boxcol(self) -> float:
        return self._boxcol

    @boxcol.setter
    def boxcol(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("boxcol must be a number between 0 and 1.")

        self._boxcol = value

    @property
    def outcol(self) -> float:
        return self._outcol

    @outcol.setter
    def outcol(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("outcol must be a number between 0 and 1.")
        self._outcol = value

    @property
    def dim(self) -> int:
        return 3

    def _plot(
        self,
        fig: Figure,
        axes: Sequence[Axes],
    ) -> None:

        x = self.fdatagrid.grid_points[0]
        lx = len(x)
        y = self.fdatagrid.grid_points[1]
        ly = len(y)
        X, Y = np.meshgrid(x, y)

        for m, ax in enumerate(axes):

            # mean sample
            ax.plot_wireframe(
                X,
                Y,
                np.squeeze(self.median[..., m]).T,
                rstride=ly,
                cstride=lx,
                color=self.colormap(self.boxcol),
            )
            ax.plot_surface(
                X,
                Y,
                np.squeeze(self.median[..., m]).T,
                color=self.colormap(self.boxcol),
                alpha=0.8,
            )

            # central envelope
            ax.plot_surface(
                X,
                Y,
                np.squeeze(self.central_envelope[0][..., m]).T,
                color=self.colormap(self.boxcol),
                alpha=0.5,
            )
            ax.plot_wireframe(
                X,
                Y,
                np.squeeze(self.central_envelope[0][..., m]).T,
                rstride=ly,
                cstride=lx,
                color=self.colormap(self.boxcol),
            )
            ax.plot_surface(
                X,
                Y,
                np.squeeze(self.central_envelope[1][..., m]).T,
                color=self.colormap(self.boxcol),
                alpha=0.5,
            )
            ax.plot_wireframe(
                X,
                Y,
                np.squeeze(self.central_envelope[1][..., m]).T,
                rstride=ly,
                cstride=lx,
                color=self.colormap(self.boxcol),
            )

            # box vertical lines
            for indices in (
                (0, 0),
                (0, ly - 1),
                (lx - 1, 0),
                (lx - 1, ly - 1),
            ):
                x_corner = x[indices[0]]
                y_corner = y[indices[1]]
                ax.plot(
                    [x_corner, x_corner],
                    [y_corner, y_corner],
                    [
                        self.central_envelope[1][..., m][
                            indices[0],
                            indices[1],
                        ],
                        self.central_envelope[0][..., m][
                            indices[0],
                            indices[1],
                        ],
                    ],
                    color=self.colormap(self.boxcol),
                )

            # outlying envelope
            ax.plot_surface(
                X,
                Y,
                np.squeeze(self.non_outlying_envelope[0][..., m]).T,
                color=self.colormap(self.outcol),
                alpha=0.3,
            )
            ax.plot_wireframe(
                X,
                Y,
                np.squeeze(self.non_outlying_envelope[0][..., m]).T,
                rstride=ly,
                cstride=lx,
                color=self.colormap(self.outcol),
            )
            ax.plot_surface(
                X,
                Y,
                np.squeeze(self.non_outlying_envelope[1][..., m]).T,
                color=self.colormap(self.outcol),
                alpha=0.3,
            )
            ax.plot_wireframe(
                X,
                Y,
                np.squeeze(self.non_outlying_envelope[1][..., m]).T,
                rstride=ly,
                cstride=lx,
                color=self.colormap(self.outcol),
            )

            # vertical lines from central to outlying envelope
            x_index = math.floor(lx / 2)
            x_central = x[x_index]
            y_index = math.floor(ly / 2)
            y_central = y[y_index]
            ax.plot(
                [x_central, x_central],
                [y_central, y_central],
                [
                    self.non_outlying_envelope[1][..., m][x_index, y_index],
                    self.central_envelope[1][..., m][x_index, y_index],
                ],
                color=self.colormap(self.boxcol),
            )
            ax.plot(
                [x_central, x_central],
                [y_central, y_central],
                [
                    self.non_outlying_envelope[0][..., m][x_index, y_index],
                    self.central_envelope[0][..., m][x_index, y_index],
                ],
                color=self.colormap(self.boxcol),
            )

        _set_labels(self.fdatagrid, fig, axes)

    def __repr__(self) -> str:
        """Return repr(self)."""
        return (
            f"SurfaceBoxplot("
            f"\nFDataGrid={repr(self.fdatagrid)},"
            f"\nmedian={repr(self.median)},"
            f"\ncentral envelope={repr(self.central_envelope)},"
            f"\noutlying envelope={repr(self.non_outlying_envelope)})"
        ).replace('\n', '\n    ')

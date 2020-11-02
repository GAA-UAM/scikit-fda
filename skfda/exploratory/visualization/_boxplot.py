"""Functional Data Boxplot Module.

This module contains the classes to construct the functional data boxplot and
visualize it.

"""
from abc import ABC, abstractmethod
import math

import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from ..depth import ModifiedBandDepth
from ..outliers import _envelopes
from ._utils import (_figure_to_svg, _get_figure_and_axes,
                     _set_figure_layout_for_fdata, _set_labels)


__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


class FDataBoxplot(ABC):
    """Abstract class inherited by the Boxplot and SurfaceBoxplot classes.

    It the data of the functional boxplot or surface boxplot of a FDataGrid
    object, depending on the dimensions of the :term:`domain`, 1 or 2
    respectively.

    It forces to both classes, Boxplot and SurfaceBoxplot to conain at least
    the median, central and outlying envelopes and a colormap for their
    graphical representation, obtained calling the plot method.

    """
    @abstractmethod
    def __init__(self, factor=1.5):
        if factor < 0:
            raise ValueError("The number used to calculate the "
                             "outlying envelope must be positive.")
        self._factor = factor

    @property
    def factor(self):
        return self._factor

    @property
    def fdatagrid(self):
        pass

    @property
    def median(self):
        pass

    @property
    def central_envelope(self):
        pass

    @property
    def non_outlying_envelope(self):
        pass

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if not isinstance(value, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError("colormap must be of type "
                             "matplotlib.colors.LinearSegmentedColormap")
        self._colormap = value

    @abstractmethod
    def plot(self, chart=None, *, fig=None, axes=None,
             n_rows=None, n_cols=None):
        pass

    def _repr_svg_(self):
        fig = self.plot()
        plt.close(fig)

        return _figure_to_svg(fig)


class Boxplot(FDataBoxplot):
    r"""Representation of the functional boxplot.

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

    Args:

        fdatagrid (FDataGrid): Object containing the data.
        depth_method (:ref:`depth measure <depth-measures>`, optional):
            Method used to order the data. Defaults to :func:`modified
            band depth
            <skfda.exploratory.depth.ModifiedBandDepth>`.
        prob (list of float, optional): List with float numbers (in the
            range from 1 to 0) that indicate which central regions to
            represent.
            Defaults to [0.5] which represents the 50% central region.
        factor (double): Number used to calculate the outlying envelope.

    Attributes:

        fdatagrid (FDataGrid): Object containing the data.
        median (array, (fdatagrid.dim_codomain, ngrid_points)): contains
            the median/s.
        central_envelope (array, (fdatagrid.dim_codomain, 2, ngrid_points)):
            contains the central envelope/s.
        non_outlying_envelope (array, (fdatagrid.dim_codomain, 2,
            ngrid_points)):
            contains the non-outlying envelope/s.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from
            which the colors to represent the central regions are selected.
        envelopes (array, (fdatagrid.dim_codomain * ncentral_regions, 2,
            ngrid_points)): contains the region envelopes.
        outliers (array, (fdatagrid.dim_codomain, fdatagrid.n_samples)):
            contains the outliers.
        barcol (string): Color of the envelopes and vertical lines.
        outliercol (string): Color of the ouliers.
        mediancol (string): Color of the median.
        show_full_outliers (boolean): If False (the default) then only the part
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

        Sun, Y., & Genton, M. G. (2011). Functional Boxplots. Journal of
        Computational and Graphical Statistics, 20(2), 316-334.
        https://doi.org/10.1198/jcgs.2011.09224


    """

    def __init__(self, fdatagrid, depth_method=ModifiedBandDepth(), prob=[0.5],
                 factor=1.5):
        """Initialization of the Boxplot class.

        Args:
            fdatagrid (FDataGrid): Object containing the data.
            depth_method (:ref:`depth measure <depth-measures>`, optional):
                Method used to order the data. Defaults to :func:`modified
                band depth
                <skfda.exploratory.depth.ModifiedBandDepth>`.
            prob (list of float, optional): List with float numbers (in the
                range from 1 to 0) that indicate which central regions to
                represent.
                Defaults to [0.5] which represents the 50% central region.
            factor (double): Number used to calculate the outlying envelope.

        """
        FDataBoxplot.__init__(self, factor)

        if fdatagrid.dim_domain != 1:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 1.")

        if sorted(prob, reverse=True) != prob:
            raise ValueError(
                "Probabilities required to be in descending order.")

        if min(prob) < 0 or max(prob) > 1:
            raise ValueError("Probabilities must be between 0 and 1.")

        self._envelopes = [None] * len(prob)

        depth = depth_method(fdatagrid)
        indices_descending_depth = (-depth).argsort(axis=0)

        # The median is the deepest curve
        self._median = fdatagrid[indices_descending_depth[0]
                                 ].data_matrix[0, ...]

        # Central region and envelope must be computed for outlier detection
        central_region = _envelopes._compute_region(
            fdatagrid, indices_descending_depth, 0.5)
        self._central_envelope = _envelopes._compute_envelope(central_region)

        # Non-outlying envelope
        non_outlying_threshold = _envelopes._non_outlying_threshold(
            self._central_envelope, factor)
        predicted_outliers = _envelopes._predict_outliers(
            fdatagrid, non_outlying_threshold)
        inliers = fdatagrid[predicted_outliers == 0]
        self._non_outlying_envelope = _envelopes._compute_envelope(inliers)

        # Outliers
        self._outliers = _envelopes._predict_outliers(
            fdatagrid, self._non_outlying_envelope)

        for i, p in enumerate(prob):
            region = _envelopes._compute_region(
                fdatagrid, indices_descending_depth, p)
            self._envelopes[i] = _envelopes._compute_envelope(region)

        self._fdatagrid = fdatagrid
        self._prob = prob
        self._colormap = plt.cm.get_cmap('RdPu')
        self.barcol = "blue"
        self.outliercol = "red"
        self.mediancol = "black"
        self._show_full_outliers = False

    @property
    def fdatagrid(self):
        return self._fdatagrid

    @property
    def median(self):
        return self._median

    @property
    def central_envelope(self):
        return self._central_envelope

    @property
    def non_outlying_envelope(self):
        return self._non_outlying_envelope

    @property
    def envelopes(self):
        return self._envelopes

    @property
    def outliers(self):
        return self._outliers

    @property
    def show_full_outliers(self):
        return self._show_full_outliers

    @show_full_outliers.setter
    def show_full_outliers(self, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("show_full_outliers must be boolean type")
        self._show_full_outliers = boolean

    def plot(self, chart=None, *, fig=None, axes=None,
             n_rows=None, n_cols=None):
        """Visualization of the functional boxplot of the fdatagrid
        (dim_domain=1).

        Args:
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            axes (list of axis objects, optional): axis over where the graphs
                are plotted. If None, see param fig.
            n_rows(int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            n_cols(int, optional): designates the number of columns of the
                figure to plot the different dimensions of the image. Only
                specified if fig and ax are None.

        Returns:
            fig (figure): figure object in which the graphs are plotted.

        """

        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout_for_fdata(
            self.fdatagrid, fig, axes, n_rows, n_cols)
        tones = np.linspace(0.1, 1.0, len(self._prob) + 1, endpoint=False)[1:]
        color = self.colormap(tones)

        if self.show_full_outliers:
            var_zorder = 1
        else:
            var_zorder = 4

        outliers = self.fdatagrid[self.outliers]

        for m in range(self.fdatagrid.dim_codomain):

            # Outliers
            for o in outliers:
                axes[m].plot(o.grid_points[0],
                             o.data_matrix[0, :, m],
                             color=self.outliercol,
                             linestyle='--', zorder=1)

            for i in range(len(self._prob)):
                # central regions
                axes[m].fill_between(self.fdatagrid.grid_points[0],
                                     self.envelopes[i][0][..., m],
                                     self.envelopes[i][1][..., m],
                                     facecolor=color[i], zorder=var_zorder)

            # outlying envelope
            axes[m].plot(self.fdatagrid.grid_points[0],
                         self.non_outlying_envelope[0][..., m],
                         self.fdatagrid.grid_points[0],
                         self.non_outlying_envelope[1][..., m],
                         color=self.barcol, zorder=4)

            # central envelope
            axes[m].plot(self.fdatagrid.grid_points[0],
                         self.central_envelope[0][..., m],
                         self.fdatagrid.grid_points[0],
                         self.central_envelope[1][..., m],
                         color=self.barcol, zorder=4)

            # vertical lines
            index = math.ceil(self.fdatagrid.ncol / 2)
            x = self.fdatagrid.grid_points[0][index]
            axes[m].plot([x, x],
                         [self.non_outlying_envelope[0][..., m][index],
                          self.central_envelope[0][..., m][index]],
                         color=self.barcol,
                         zorder=4)
            axes[m].plot([x, x],
                         [self.non_outlying_envelope[1][..., m][index],
                          self.central_envelope[1][..., m][index]],
                         color=self.barcol, zorder=4)

            # median sample
            axes[m].plot(self.fdatagrid.grid_points[0], self.median[..., m],
                         color=self.mediancol, zorder=5)

        _set_labels(self.fdatagrid, fig, axes)

        return fig

    def __repr__(self):
        """Return repr(self)."""
        return (f"Boxplot("
                f"\nFDataGrid={repr(self.fdatagrid)},"
                f"\nmedian={repr(self.median)},"
                f"\ncentral envelope={repr(self.central_envelope)},"
                f"\nnon-outlying envelope={repr(self.non_outlying_envelope)},"
                f"\nenvelopes={repr(self.envelopes)},"
                f"\noutliers={repr(self.outliers)})").replace('\n', '\n    ')


class SurfaceBoxplot(FDataBoxplot):
    r"""Representation of the surface boxplot.

    Class implementing the surface boxplot. Analogously to the functional
    boxplot, it is an informative exploratory tool for visualizing functional
    data with :term:`domain` dimension 2. Nevertheless, it does not implement
    the enhanced surface boxplot.

    Based on the center outward ordering induced by a
    :ref:`depth measure <depth-measures>`
    for functional data, it represents the envelope of the
    50% central region, the median curve, and the maximum non-outlying
    envelope.

    Args:

        fdatagrid (FDataGrid): Object containing the data.
        method (:ref:`depth measure <depth-measures>`, optional): Method
            used to order the data. Defaults to :class:`modified band depth
            <skfda.exploratory.depth.ModifiedBandDepth>`.
        prob (list of float, optional): List with float numbers (in the
            range from 1 to 0) that indicate which central regions to
            represent.
            Defaults to [0.5] which represents the 50% central region.
        factor (double): Number used to calculate the outlying envelope.

    Attributes:

        fdatagrid (FDataGrid): Object containing the data.
        median (array, (fdatagrid.dim_codomain, lx, ly)): contains
            the median/s.
        central_envelope (array, (fdatagrid.dim_codomain, 2, lx, ly)):
            contains the central envelope/s.
        non_outlying_envelope (array,(fdatagrid.dim_codomain, 2, lx, ly)):
            contains the non-outlying envelope/s.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from
            which the colors to represent the central regions are selected.
        boxcol (string): Color of the box, which includes median and central
            envelope.
        outcol (string): Color of the outlying envelope.

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

        Sun, Y., & Genton, M. G. (2011). Functional Boxplots. Journal of
        Computational and Graphical Statistics, 20(2), 316-334.
        https://doi.org/10.1198/jcgs.2011.09224

    """

    def __init__(self, fdatagrid, method=ModifiedBandDepth(), factor=1.5):
        """Initialization of the functional boxplot.

        Args:
            fdatagrid (FDataGrid): Object containing the data.
            method (:ref:`depth measure <depth-measures>`, optional): Method
                used to order the data. Defaults to :class:`modified band depth
                <skfda.exploratory.depth.ModifiedBandDepth>`.
            prob (list of float, optional): List with float numbers (in the
                range from 1 to 0) that indicate which central regions to
                represent.
                Defaults to [0.5] which represents the 50% central region.
            factor (double): Number used to calculate the outlying envelope.

        """
        FDataBoxplot.__init__(self, factor)

        if fdatagrid.dim_domain != 2:
            raise ValueError(
                "Class only supports FDataGrid with domain dimension 2.")

        depth = method(fdatagrid)
        indices_descending_depth = (-depth).argsort(axis=0)

        # The mean is the deepest curve
        self._median = fdatagrid.data_matrix[indices_descending_depth[0]]

        # Central region and envelope must be computed for outlier detection
        central_region = _envelopes._compute_region(
            fdatagrid, indices_descending_depth, 0.5)
        self._central_envelope = _envelopes._compute_envelope(central_region)

        # Non-outlying envelope
        non_outlying_threshold = _envelopes._non_outlying_threshold(
            self._central_envelope, factor)
        predicted_outliers = _envelopes._predict_outliers(
            fdatagrid, non_outlying_threshold)
        inliers = fdatagrid[predicted_outliers == 0]
        self._non_outlying_envelope = _envelopes._compute_envelope(inliers)

        self._fdatagrid = fdatagrid
        self.colormap = plt.cm.get_cmap('Greys')
        self._boxcol = 1.0
        self._outcol = 0.7

    @property
    def fdatagrid(self):
        return self._fdatagrid

    @property
    def median(self):
        return self._median

    @property
    def central_envelope(self):
        return self._central_envelope

    @property
    def non_outlying_envelope(self):
        return self._non_outlying_envelope

    @property
    def boxcol(self):
        return self._boxcol

    @boxcol.setter
    def boxcol(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                "boxcol must be a number between 0 and 1.")

        self._boxcol = value

    @property
    def outcol(self):
        return self._outcol

    @outcol.setter
    def outcol(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                "outcol must be a number between 0 and 1.")
        self._outcol = value

    def plot(self, chart=None, *, fig=None, axes=None,
             n_rows=None, n_cols=None):
        """Visualization of the surface boxplot of the fdatagrid (dim_domain=2).

         Args:
             fig (figure object, optional): figure over with the graphs are
                 plotted in case ax is not specified. If None and ax is also
                 None, the figure is initialized.
             axes (list of axis objects, optional): axis over where the graphs
                 are plotted. If None, see param fig.
             n_rows(int, optional): designates the number of rows of the figure
                 to plot the different dimensions of the image. Only specified
                 if fig and ax are None.
             n_cols(int, optional): designates the number of columns of the
                 figure to plot the different dimensions of the image. Only
                 specified if fig and ax are None.

        Returns:
            fig (figure): figure object in which the graphs are plotted.

        """
        fig, axes = _get_figure_and_axes(chart, fig, axes)
        fig, axes = _set_figure_layout_for_fdata(
            self.fdatagrid, fig, axes, n_rows, n_cols)

        x = self.fdatagrid.grid_points[0]
        lx = len(x)
        y = self.fdatagrid.grid_points[1]
        ly = len(y)
        X, Y = np.meshgrid(x, y)

        for m in range(self.fdatagrid.dim_codomain):

            # mean sample
            axes[m].plot_wireframe(X, Y, np.squeeze(self.median[..., m]).T,
                                   rstride=ly, cstride=lx,
                                   color=self.colormap(self.boxcol))
            axes[m].plot_surface(X, Y, np.squeeze(self.median[..., m]).T,
                                 color=self.colormap(self.boxcol), alpha=0.8)

            # central envelope
            axes[m].plot_surface(
                X, Y, np.squeeze(self.central_envelope[0][..., m]).T,
                color=self.colormap(self.boxcol), alpha=0.5)
            axes[m].plot_wireframe(
                X, Y, np.squeeze(self.central_envelope[0][..., m]).T,
                rstride=ly, cstride=lx,
                color=self.colormap(self.boxcol))
            axes[m].plot_surface(
                X, Y, np.squeeze(self.central_envelope[1][..., m]).T,
                color=self.colormap(self.boxcol), alpha=0.5)
            axes[m].plot_wireframe(
                X, Y, np.squeeze(self.central_envelope[1][..., m]).T,
                rstride=ly, cstride=lx,
                color=self.colormap(self.boxcol))

            # box vertical lines
            for indices in [(0, 0), (0, ly - 1), (lx - 1, 0),
                            (lx - 1, ly - 1)]:
                x_corner = x[indices[0]]
                y_corner = y[indices[1]]
                axes[m].plot(
                    [x_corner, x_corner], [y_corner, y_corner],
                    [
                        self.central_envelope[1][..., m][indices[0],
                                                         indices[1]],
                        self.central_envelope[0][..., m][indices[0],
                                                         indices[1]]],
                    color=self.colormap(self.boxcol))

            # outlying envelope
            axes[m].plot_surface(
                X, Y,
                np.squeeze(self.non_outlying_envelope[0][..., m]).T,
                color=self.colormap(self.outcol), alpha=0.3)
            axes[m].plot_wireframe(
                X, Y,
                np.squeeze(self.non_outlying_envelope[0][..., m]).T,
                rstride=ly, cstride=lx,
                color=self.colormap(self.outcol))
            axes[m].plot_surface(
                X, Y,
                np.squeeze(self.non_outlying_envelope[1][..., m]).T,
                color=self.colormap(self.outcol), alpha=0.3)
            axes[m].plot_wireframe(
                X, Y,
                np.squeeze(self.non_outlying_envelope[1][..., m]).T,
                rstride=ly, cstride=lx,
                color=self.colormap(self.outcol))

            # vertical lines from central to outlying envelope
            x_index = math.floor(lx / 2)
            x_central = x[x_index]
            y_index = math.floor(ly / 2)
            y_central = y[y_index]
            axes[m].plot(
                [x_central, x_central], [y_central, y_central],
                [self.non_outlying_envelope[1][..., m][x_index, y_index],
                 self.central_envelope[1][..., m][x_index, y_index]],
                color=self.colormap(self.boxcol))
            axes[m].plot(
                [x_central, x_central], [y_central, y_central],
                [self.non_outlying_envelope[0][..., m][x_index, y_index],
                 self.central_envelope[0][..., m][x_index, y_index]],
                color=self.colormap(self.boxcol))

        _set_labels(self.fdatagrid, fig, axes)

        return fig

    def __repr__(self):
        """Return repr(self)."""
        return ((f"SurfaceBoxplot("
                 f"\nFDataGrid={repr(self.fdatagrid)},"
                 f"\nmedian={repr(self.median)},"
                 f"\ncentral envelope={repr(self.central_envelope)},"
                 f"\noutlying envelope={repr(self.non_outlying_envelope)})")
                .replace('\n', '\n    '))

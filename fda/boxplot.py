"""Functional Data Boxplot Module.

This module contains the functions to construct the functional data boxplot.
It also defines a class for representing the information returned by those functions.

"""
import matplotlib
import matplotlib.pyplot as plt
import math

from .depth_measures import *
from .grid import FDataGrid
import StringIO

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"


def _generic_boxplot_checks(fdatagrid, fig, ax, nrows, ncols, colormap,
                            factor):
    fig, ax = fdatagrid.generic_plotting_checks(fig, ax, nrows, ncols)

    if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
        raise ValueError(
            "colormap must be of type matplotlib.colors.LinearSegmentedColormap")

    if factor < 0:
        raise ValueError(
            "The number used to calculate the outlying envelope must be positive.")

    return fig, ax


def fdboxplot(fdatagrid, fig=None, ax=None, method=modified_band_depth,
              prob=[0.5], fullout=False, factor=1.5,
              colormap=plt.cm.get_cmap('RdPu'), barcol="blue",
              outliercol="red", mediancol="black", nrows=None, ncols=None):
    """Implementation of the functional boxplot.

    It is an informative exploratory tool for visualizing functional data, as well as
    its generalization, the enhanced functional boxplot. Only supports 1 dimensional
    domain functional data.

    Based on the center outward ordering induced by a :ref:`depth measure <depth-measures>`
    for functional data, the descriptive statistics of a functional boxplot are: the
    envelope of the 50% central region, the median curve,and the maximum non-outlying envelope.
    In addition, outliers can be detected in a functional boxplot by the 1.5 times the 50%
    central region empirical rule, analogous to the rule for classical boxplots.

    Args:
        fdatagrid (FDataGrid): Object to be visualized.
        fig (figure object, optional): figure over with the graphs are plotted in case ax is not specified.
            If None and ax is also None, the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are plotted. If None, see param fig.
        method (:ref:`depth measure <depth-measures>`, optional): Method used to order the data.
            Defaults to :func:`modified band depth <fda.depth_measures.modified_band_depth>`.
        prob (list of float, optional): List with float numbers (in the range from 1 to 0) that indicate which central regions to
            represent. Defaults to [0.5] which represents the 50% central region.
        fullout (boolean): If true, the entire curve of the outlier samples is shown. Defaults to False.
        factor (double): Number used to calculate the outlying envelope.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from which the colors to represent the central regions are selected.
        barcol (string): Color of the envelopes and vertical lines.
        outliercol (string): Color of the outliers.
        mediancol (string): Color of the median.
        nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.

    Returns:
        :class:`FDataBoxplotInfo <fda.boxplot.FDataBoxplotInfo>` object: _plot

    """
    if fdatagrid.ndim_domain != 1:
        raise ValueError(
            "Function only supports FDataGrid with domain dimension 1.")

    fig, ax = _generic_boxplot_checks(fdatagrid, fig, ax, method, nrows, ncols,
                                      colormap, factor)

    if sorted(prob, reverse=True) != prob:
        raise ValueError("Probabilities required to be in descending order.")

    if min(prob) < 0 or max(prob) > 1:
        raise ValueError("Probabilities must be between 0 and 1.")

    # Obtaining the necessary colors of the colormap.
    tones = np.linspace(0.1, 1.0, len(prob) + 1, endpoint=False)[1:]
    color = colormap(tones)

    if fullout:
        var_zorder = 4
    else:
        var_zorder = 1

    depth = method(fdatagrid)
    indices_descencing_depth = (-depth).argsort(axis=0)

    _plot = FDataBoxplotInfo(fdatagrid, color, barcol, mediancol, outliercol,
                             nrows, ncols)

    for m in range(fdatagrid.ndim_image):

        for i in range(len(prob)):

            indices_samples = indices_descencing_depth[:, m][
                              :math.ceil(fdatagrid.nsamples * prob[i])]
            samples_used = fdatagrid.data_matrix[indices_samples, :, m]
            max_samples_used = np.amax(samples_used, axis=0)
            min_samples_used = np.amin(samples_used, axis=0)

            if prob[i] == 0.5:
                # central envelope
                ax[m].plot(fdatagrid.sample_points[0], max_samples_used.T,
                           fdatagrid.sample_points[0],
                           min_samples_used.T, color=barcol, zorder=4)
                _plot.add_central_env(
                    [[max_samples_used.T, min_samples_used.T]])

                # outlying envelope
                max_value = np.amax(fdatagrid.data_matrix[:, :, m], axis=0)
                min_value = np.amin(fdatagrid.data_matrix[:, :, m], axis=0)
                iqr = np.absolute(max_samples_used - min_samples_used)
                outlying_max_envelope = np.minimum(
                    max_samples_used + iqr * factor, max_value)
                outlying_min_envelope = np.maximum(
                    min_samples_used - iqr * factor, min_value)
                ax[m].plot(fdatagrid.sample_points[0],
                           outlying_max_envelope.flatten(),
                           fdatagrid.sample_points[0],
                           outlying_min_envelope.flatten(), color=barcol,
                           zorder=4)
                _plot.add_outlying_env([[outlying_max_envelope.flatten(),
                                         outlying_min_envelope.flatten()]])

                # vertical lines
                index = math.ceil(fdatagrid.ncol / 2)
                x = fdatagrid.sample_points[0][index]
                ax[m].plot([x, x], [outlying_max_envelope[index],
                                    max_samples_used[index]], color=barcol,
                           zorder=4)
                ax[m].plot([x, x], [outlying_min_envelope[index],
                                    min_samples_used[index]], color=barcol,
                           zorder=4)

                # outliers
                _plot.outliers = np.zeros(
                    (fdatagrid.ndim_image, fdatagrid.nsamples))
                for j in list(range(fdatagrid.nsamples)):
                    outliers_above = (
                            outlying_max_envelope < fdatagrid.data_matrix[
                                                    j, :, m])
                    outliers_below = (
                            outlying_min_envelope > fdatagrid.data_matrix[
                                                    j, :, m])
                    if (outliers_above.sum() > 0 or outliers_below.sum() > 0):
                        _plot.outliers[m, j] = 1
                        ax[m].plot(fdatagrid.sample_points[0],
                                   fdatagrid.data_matrix[j, :, m],
                                   color=outliercol,
                                   linestyle='--', zorder=1)

            # central regions
            ax[m].fill_between(fdatagrid.sample_points[0],
                               max_samples_used.flatten(),
                               min_samples_used.flatten(), facecolor=color[i],
                               zorder=var_zorder)
            _plot.add_central_region(
                [[max_samples_used.flatten(), min_samples_used.flatten()]])

        # mean sample
        ax[m].plot(fdatagrid.sample_points[0],
                   fdatagrid.data_matrix[indices_descencing_depth[0, m], :,
                   m].T,
                   color=mediancol,
                   zorder=5)
        _plot.add_median(
            [fdatagrid.data_matrix[indices_descencing_depth[0, m], :, m].T])

    fdatagrid.set_labels(fig, ax)

    return _plot


def surface_boxplot(fdatagrid, fig=None, ax=None, method=modified_band_depth,
                    factor=1.5,
                    colormap=plt.cm.get_cmap('Greys'), boxcol=1.0, outcol=0.7,
                    nrows=None, ncols=None):
    """Implementation of the surface boxplot.

    Analogously to the functional boxplot, it is an informative exploratory tool for visualizing
    functional data with domain dimension 2. Nevertheless, it does not implement the enhanced
    surface boxplot.

    Based on the center outward ordering induced by a :ref:`depth measure <depth-measures>`
    for functional data, it represents the envelope of the 50% central region, the median curve,
    and the maximum non-outlying envelope.

    Args:
        fdatagrid (FDataGrid): Object to be visualized.
        fig (figure object, optional): figure over with the graphs are plotted in case ax is not specified.
            If None and ax is also None, the figure is initialized.
        ax (list of axis objects, optional): axis over where the graphs are plotted. If None, see param fig.
        method (:ref:`depth measure <depth-measures>`, optional): Method used to order the data.
            Defaults to :func:`modified band depth <fda.depth_measures.modified_band_depth>`.
        factor (double): Number used to calculate the outlying envelope.
        colormap(matplotlib.pyplot.LinearSegmentedColormap, optional): Colormap from which the
            colors of the plot are extracted. Defaults to 'Greys'.
        boxcol (float, optional): Tone of the colormap the box: mean, central envelopes and vertical lines.
            Defaults to 0.7.
        outboxcol (float, optional): Tone of the colormap to plot the outlying envelopes.
            Defaults to 1.0.
        nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.
        ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
            image. Only specified if fig and ax are None.

    Returns:
        :class:`FDataBoxplotInfo <fda.boxplot.FDataBoxplotInfo>` object: _plot

    """
    if fdatagrid.ndim_domain != 2:
        raise ValueError(
            "Function only supports FDataGrid with domain dimension 2.")

    fig, ax = _generic_boxplot_checks(fdatagrid, fig, ax, method, nrows, ncols,
                                      colormap, factor)

    if boxcol < 0 or boxcol > 1:
        raise ValueError("Argument boxcol must be a number between 0 and 1.")

    if outcol < 0 or outcol > 1:
        raise ValueError("Argument boxcol must be a number between 0 and 1.")

    depth = method(fdatagrid)
    indices_descencing_depth = (-depth).argsort(axis=0)

    x = fdatagrid.sample_points[0]
    lx = len(x)
    y = fdatagrid.sample_points[1]
    ly = len(y)
    X, Y = np.meshgrid(x, y)

    _plot = FDataBoxplotInfo(fdatagrid, central_regions_col=[colormap(boxcol)],
                             envelopes_col=colormap(outcol),
                             median_col=colormap(boxcol), outliers_col=None,
                             nrows=nrows, ncols=ncols)

    for m in range(fdatagrid.ndim_image):

        indices_samples = indices_descencing_depth[:, m][
                          :math.ceil(fdatagrid.nsamples * 0.5)]
        samples_used = fdatagrid.data_matrix[indices_samples, :, :, m]
        max_samples_used = np.amax(samples_used, axis=0)
        min_samples_used = np.amin(samples_used, axis=0)

        # mean sample
        ax[m].plot_wireframe(X, Y, np.squeeze(
            fdatagrid.data_matrix[indices_descencing_depth[0, m], :, :, m]).T,
                             rstride=ly, cstride=lx, color=colormap(boxcol))
        ax[m].plot_surface(X, Y, np.squeeze(
            fdatagrid.data_matrix[indices_descencing_depth[0, m], :, :, m]).T,
                           color=colormap(boxcol), alpha=0.8)
        _plot.add_median(
            [fdatagrid.data_matrix[indices_descencing_depth[0, m], :, :, m]])

        # central envelope
        ax[m].plot_surface(X, Y, np.squeeze(max_samples_used).T,
                           color=colormap(boxcol), alpha=0.5)
        ax[m].plot_wireframe(X, Y, np.squeeze(max_samples_used).T, rstride=ly,
                             cstride=lx, color=colormap(boxcol))
        ax[m].plot_surface(X, Y, np.squeeze(min_samples_used).T,
                           color=colormap(boxcol), alpha=0.5)
        ax[m].plot_wireframe(X, Y, np.squeeze(min_samples_used).T, rstride=ly,
                             cstride=lx, color=colormap(boxcol))
        _plot.add_central_env([[max_samples_used, min_samples_used]])

        # box vertical lines
        for indices in [(0, 0), (0, ly - 1), (lx - 1, 0), (lx - 1, ly - 1)]:
            x_corner = x[indices[0]]
            y_corner = y[indices[1]]
            ax[m].plot([x_corner, x_corner], [y_corner, y_corner],
                       [min_samples_used[indices[0], indices[1]],
                        max_samples_used[indices[0], indices[1]]],
                       color=colormap(boxcol))

        # outlying envelope
        max_value = np.amax(fdatagrid.data_matrix[:, :, :, m], axis=0)
        min_value = np.amin(fdatagrid.data_matrix[:, :, :, m], axis=0)
        iqr = np.absolute(max_samples_used - min_samples_used)
        oulying_max_envelope = np.minimum(max_samples_used + iqr * factor,
                                          max_value)
        oulying_min_envelope = np.maximum(min_samples_used - iqr * factor,
                                          min_value)
        ax[m].plot_surface(X, Y, np.squeeze(oulying_max_envelope).T,
                           color=colormap(outcol), alpha=0.3)
        ax[m].plot_wireframe(X, Y, np.squeeze(oulying_max_envelope).T,
                             rstride=ly, cstride=lx, color=colormap(outcol))
        ax[m].plot_surface(X, Y, np.squeeze(oulying_min_envelope).T,
                           color=colormap(outcol), alpha=0.3)
        ax[m].plot_wireframe(X, Y, np.squeeze(oulying_min_envelope).T,
                             rstride=ly, cstride=lx, color=colormap(outcol))
        _plot.add_outlying_env([[oulying_max_envelope, oulying_min_envelope]])

        # vertical lines from central to outlying envelope
        x_index = math.floor(lx / 2)
        x_central = x[x_index]
        y_index = math.floor(ly / 2)
        y_central = y[y_index]
        ax[m].plot([x_central, x_central], [y_central, y_central],
                   [oulying_max_envelope[x_index, y_index],
                    max_samples_used[x_index, y_index]],
                   color=colormap(boxcol))
        ax[m].plot([x_central, x_central], [y_central, y_central],
                   [oulying_min_envelope[x_index, y_index],
                    min_samples_used[x_index, y_index]],
                   color=colormap(boxcol))

    fdatagrid.set_labels(fig, ax)

    return _plot


class FDataBoxplotInfo:
    r"""Data of the functional boxplot.

    Class containing the data of the functional boxplot or surface boxplot of a FDataGrid object,
    depending on the dimensions of the domain, 1 or 2 respectively.

    Class returned by the functions fdboxplot and surface_boxplot which contains the median, central and outlying
    envelopes. In the first case, it also includes the possibility of other central regions (apart from the 50% one,
    which is equivalent to the central envelope) and the outliers.

    Examples:
        Function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> import matplotlib.pyplot as plt
        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1], [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points, dataset_label="dataset", axes_labels=["x_label", "y_label"])
        >>> plt.figure() # doctest: +IGNORE_RESULT
        >>> fdboxplot(fd)
        FDataBoxplotInfo(
              median=array([[0.5, 0.5, 1. , 2. , 1.5, 1. ]]),
              central_env=array([[[ 0.5,  0.5,  1. ,  2. ,  1.5,  1. ],
                      [-1. , -1. , -0.5,  1. ,  1. ,  0.5]]]),
              outlying_env=array([[[ 1.  ,  1.  ,  2.  ,  3.  ,  2.25,  1.75],
                      [-1.  , -1.  , -0.5 , -0.5 ,  0.25, -0.25]]]),
              central_regions=array([[[ 0.5,  0.5,  1. ,  2. ,  1.5,  1. ],
                      [-1. , -1. , -0.5,  1. ,  1. ,  0.5]]]),
              outliers=array([[1., 0., 0., 1.]]))

        Function :math:`f : \mathbb{R^2}\longmapsto\mathbb{R^2}`.

        >>> import matplotlib.pyplot as plt
        >>> data_matrix = [[[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
        ...                [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]]]
        >>> sample_points = [[2, 4], [3, 6, 8]]
        >>> fd = FDataGrid(data_matrix, sample_points, dataset_label= "dataset",
        ...                axes_labels=["x1_label", "x2_label", "y1_label", "y2_label"])
        >>> plt.figure() # doctest: +IGNORE_RESULT
        >>> surface_boxplot(fd)
        FDataBoxplotInfo(
              median=array([[[1. , 0.3, 1. ],
                      [2. , 0.4, 2. ]],
        <BLANKLINE>
                     [[4. , 1.5, 3. ],
                      [8. , 2. , 9. ]]]),
              central_env=array([[[[1. , 0.3, 1. ],
                       [2. , 0.4, 2. ]],
        <BLANKLINE>
                      [[1. , 0.3, 1. ],
                       [2. , 0.4, 2. ]]],
        <BLANKLINE>
        <BLANKLINE>
                     [[[4. , 1.5, 3. ],
                       [8. , 2. , 9. ]],
        <BLANKLINE>
                      [[4. , 1.5, 3. ],
                       [8. , 2. , 9. ]]]]),
              outlying_env=array([[[[1. , 0.3, 1. ],
                       [2. , 0.4, 2. ]],
        <BLANKLINE>
                      [[1. , 0.3, 1. ],
                       [2. , 0.4, 2. ]]],
        <BLANKLINE>
        <BLANKLINE>
                     [[[4. , 1.5, 3. ],
                       [8. , 2. , 9. ]],
        <BLANKLINE>
                      [[4. , 1.5, 3. ],
                       [8. , 2. , 9. ]]]]),
              central_regions=[],
              outliers=[])

    """.replace('+IGNORE_RESULT', '+ELLIPSIS\n<...>')

    def __init__(self, fdatagrid, central_regions_col, envelopes_col,
                 median_col, outliers_col, nrows, ncols):
        """Initializes the attributes of the FDataBoxplotInfo object.

        Attributes:
            fdatagrid (FDataGrid obj): Object whose information about the boxplot is contained in the class.
            central_regions_col (1-D array): Array containing the colors of the central regions.
            envelopes_col (matplotlib.colors): Color of the envelopes. In the case of the surface boxplot,
                it is only the color of the outlying envelope.
            median_col(matplotlib.colors): Color of the median.
            outliers_col(matplotlib.colors, optional): Color of the outliers.
            nrows(int, optional): designates the number of rows of the figure to plot the different dimensions of the
                image.
            ncols(int, optional): designates the number of columns of the figure to plot the different dimensions of the
                image.

        """
        self.fdatagrid = fdatagrid
        self.ncentral_regions = len(central_regions_col)
        self.central_regions_col = central_regions_col
        self.envelopes_col = envelopes_col
        self.outliers_col = outliers_col
        self.median_col = median_col
        self.nrows = nrows
        self.ncols = ncols
        self.median = np.ndarray([])
        self.central_env = np.ndarray([])
        self.outlying_env = np.ndarray([])
        self.central_regions = []
        self.outliers = []

    def add_median(self, median):
        """Adds the median of the boxplot to the FDataBoxplotInfo object.

        Attributes:
            median(numpy.ndarray): Values of the median of the fdatagrid of the FDataBoxplotInfo object.
                It must have the same number elements as points of discretisation of the the fdatagrid
                with the correct shape.

        """
        if len(self.median.shape) == 0:
            self.median = np.asarray(median)
        else:
            self.median = np.append(self.median, np.asarray(median), axis=0)

    def add_central_env(self, central_env):
        """Adds the central envelope of the boxplot to the FDataBoxplotInfo object.

        Attributes:
            central_env(numpy.ndarray): Values of the central_env  of the FDataBoxplotInfo object.
                Each of the lines/surfaces defining the envelope must have the same number of
                elements as points of discretisation of the the fdatagrid with the correct shape.

        """
        if len(self.central_env.shape) == 0:
            self.central_env = np.asarray(central_env)
        else:
            self.central_env = np.append(self.central_env,
                                         np.asarray(central_env), axis=0)

    def add_outlying_env(self, outlying_env):
        """Adds the outlying envelope of the boxplot to the FDataBoxplotInfo object.

        Attributes:
            central_env(numpy.ndarray): Values of the outlying_env of the FDataBoxplotInfo object.
                Each of the lines/surfaces defining the envelope must have the same number of
                elements as points of discretisation of the the fdatagrid with the correct shape.

        """
        if len(self.outlying_env.shape) == 0:
            self.outlying_env = np.asarray(outlying_env)
        else:
            self.outlying_env = np.append(self.outlying_env,
                                          np.asarray(outlying_env), axis=0)

    def add_central_region(self, central_region):
        """Adds the central region(s) of the boxplot to the FDataBoxplotInfo object.

        Attributes:
            central_region(numpy.ndarray): Values of the central_region of the FDataBoxplotInfo object.
                Each of the lines defining the region must have the same number of
                elements as points of discretisation of the the fdatagrid with the correct shape.

        """
        if len(self.central_regions) == 0:
            self.central_regions = np.asarray(central_region)
        else:
            self.central_regions = np.append(self.central_regions,
                                             np.asarray(central_region),
                                             axis=0)

    def _visualize_fdboxplot(self):
        """Visualization of the functional boxplot of the fdatagrid (ndim_domain=1)."""

        sample_points = self.fdatagrid.sample_points[0]
        if len(
                self.central_regions) != self.ncentral_regions * self.fdatagrid.ndim_image:
            raise ValueError(
                "There must be the same number of colors as number of central regions.")

        if self.median.shape != (
                self.fdatagrid.ndim_image, len(sample_points)):
            raise ValueError(
                "There must be the same number of medians as dimensions of the image of fdatagrid"
                " and each one must have nsample_points.")

        if self.central_env.shape != (
                self.fdatagrid.ndim_image, 2, len(sample_points)):
            raise ValueError(
                "There must be the same number of central envelopes"
                "as dimensions of the image of fdatagrid"
                " and each one must have nsample_points.")

        if self.outlying_env.shape != (
                self.fdatagrid.ndim_image, 2, len(sample_points)):
            raise ValueError(
                "There must be the same number of outlying envelopes"
                "as dimensions of the image of fdatagrid"
                " and each one must have nsample_points.")

        if self.central_regions.shape != (
                self.fdatagrid.ndim_image * self.ncentral_regions, 2,
                len(sample_points)):
            raise ValueError("There must be the same number of central regions"
                             "as dimensions of the image of fdatagrid multiplied by the number of"
                             "central regions at each axis.")

        fig, ax = self.fdatagrid.set_figure_and_axes(self.nrows, self.ncols)

        for m in range(self.fdatagrid.ndim_image):
            for n in range(self.fdatagrid.nsamples):
                # outliers
                if self.outliers[m, n]:
                    ax[m].plot(sample_points,
                               self.fdatagrid.data_matrix[n, :, m],
                               color=self.outliers_col, linestyle='--')
                    # central regions
            for n in range(self.ncentral_regions):
                ax[m].fill_between(sample_points, self.central_regions[
                    m * self.ncentral_regions + n, 0],
                                   self.central_regions[
                                       m * self.ncentral_regions + n, 1],
                                   facecolor=self.central_regions_col[n])
            # outlying envelope
            ax[m].plot(sample_points, self.outlying_env[m, 0],
                       sample_points, self.outlying_env[m, 1],
                       color=self.envelopes_col)
            # central envelope
            ax[m].plot(sample_points, self.central_env[m, 0], sample_points,
                       self.central_env[m, 1], color=self.envelopes_col)
            # vertical lines joining outlying envelope and central envelope
            index = math.ceil(self.fdatagrid.ncol / 2)
            x = sample_points[index]
            ax[m].plot([x, x], [self.outlying_env[m, 0, index],
                                self.central_env[m, 0, index]],
                       color=self.envelopes_col)
            ax[m].plot([x, x], [self.outlying_env[m, 1, index],
                                self.central_env[m, 1, index]],
                       color=self.envelopes_col)
            # median
            ax[m].plot(sample_points, self.median[m], color=self.median_col)

        self.fdatagrid.set_labels(fig=fig)

        return self

    def _visualize_surface_boxplot(self):
        """Visualization of the surface boxplot of the fdatagrid (ndim_domain=2)."""

        x = self.fdatagrid.sample_points[0]
        lx = len(x)
        y = self.fdatagrid.sample_points[1]
        ly = len(y)
        X, Y = np.meshgrid(x, y)

        if self.median.shape != (self.fdatagrid.ndim_image, lx, ly):
            raise ValueError(
                "There must be the same number of medians as dimensions of the image of fdatagrid"
                " and each one must have nsample_points.")

        if self.central_env.shape != (self.fdatagrid.ndim_image, 2, lx, ly):
            raise ValueError(
                "There must be the same number of central envelopes"
                "as dimensions of the image of fdatagrid"
                " and each one must have nsample_points.")

        if self.outlying_env.shape != (self.fdatagrid.ndim_image, 2, lx, ly):
            raise ValueError(
                "There must be the same number of outlying envelopes"
                "as dimensions of the image of fdatagrid"
                " and each one must have nsample_points.")

        fig, ax = self.fdatagrid.set_figure_and_axes(self.nrows, self.ncols)

        for m in range(self.fdatagrid.ndim_image):
            # median
            ax[m].plot_wireframe(X, Y, np.squeeze(self.median[m]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.median_col)
            ax[m].plot_surface(X, Y, np.squeeze(self.median[m]).T,
                               color=self.median_col, alpha=0.8)

            # central envelope
            ax[m].plot_surface(X, Y, np.squeeze(self.central_env[m, 0]).T,
                               color=self.central_regions_col[0], alpha=0.5)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.central_env[m, 0]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.central_regions_col[0])
            ax[m].plot_surface(X, Y, np.squeeze(self.central_env[m, 1]).T,
                               color=self.central_regions_col[0], alpha=0.5)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.central_env[m, 1]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.central_regions_col[0])

            # box vertical lines
            for indices in [(0, 0), (0, ly - 1), (lx - 1, 0),
                            (lx - 1, ly - 1)]:
                x_corner = x[indices[0]]
                y_corner = y[indices[1]]
                ax[m].plot([x_corner, x_corner], [y_corner, y_corner],
                           [self.central_env[m, 1, indices[0], indices[1]],
                            self.central_env[m, 0, indices[0], indices[1]]],
                           color=self.central_regions_col[0])

            # outlying envelope
            ax[m].plot_surface(X, Y, np.squeeze(self.outlying_env[m, 0]).T,
                               color=self.envelopes_col, alpha=0.3)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.outlying_env[m, 0]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.envelopes_col)
            ax[m].plot_surface(X, Y, np.squeeze(self.outlying_env[m, 1]).T,
                               color=self.envelopes_col, alpha=0.3)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.outlying_env[m, 1]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.envelopes_col)

            # vertical lines from central to outlying envelope
            x_index = math.floor(lx / 2)
            x_central = x[x_index]
            y_index = math.floor(ly / 2)
            y_central = y[y_index]
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self.outlying_env[m, 1, x_index, y_index],
                        self.central_env[m, 1, x_index, y_index]],
                       color=self.central_regions_col[0])
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self.outlying_env[m, 0, x_index, y_index],
                        self.central_env[m, 0, x_index, y_index]],
                       color=self.central_regions_col[0])

        self.fdatagrid.set_labels(fig=fig)

        return self

    def plot(self):
        """Plot of the FDataBoxplotInfo. It calls internally to _visualize_fdboxplot or to
        _visualize_surface_boxplot depending on the dimension of the domain, 1 or 2 respectively. """

        if self.fdatagrid.ndim_domain > 2:
            raise ValueError(
                "Function only implemented if the dimension of the domain is 1 or 2.")
        if self.fdatagrid.ndim_domain == 1:
            self._visualize_fdboxplot()
        else:
            self._visualize_surface_boxplot()

    def __repr__(self):
        """Return repr(self)."""

        return ("FDataBoxplotInfo("
                "\nmedian={},"
                "\ncentral_env={},"
                "\noutlying_env={},"
                "\ncentral_regions={},"
                "\noutliers={})"
                .format(repr(self.median),
                        repr(self.central_env),
                        repr(self.outlying_env),
                        repr(self.central_regions),
                        repr(self.outliers))).replace('\n', '\n      ')


class Boxplot:
    def __init__(self, fdatagrid, method=modified_band_depth, prob=[0.5],
                 factor=1.5):
        if fdatagrid.ndim_domain != 1:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 1.")

        if sorted(prob, reverse=True) != prob:
            raise ValueError(
                "Probabilities required to be in descending order.")

        if min(prob) < 0 or max(prob) > 1:
            raise ValueError("Probabilities must be between 0 and 1.")

        if factor < 0:
            raise ValueError(
                "The number used to calculate the outlying envelope must be positive.")

        nsample_points = len(fdatagrid.sample_points[0])
        ncentral_regions = len(prob)

        self._median = np.ndarray((fdatagrid.ndim_image, nsample_points))
        self._central_envelope = np.ndarray((fdatagrid.ndim_image, 2,
                                             nsample_points))
        self._outlying_env = np.ndarray((fdatagrid.ndim_image, 2,
                                         nsample_points))
        self._cetral_regions = np.ndarray(
            (fdatagrid.ndim_image * ncentral_regions,
             2, nsample_points))
        self._outliers = np.zeros((fdatagrid.ndim_image, fdatagrid.nsamples))

        depth = method(fdatagrid)
        indices_descencing_depth = (-depth).argsort(axis=0)

        for m in range(fdatagrid.ndim_image):

            for i in range(len(prob)):

                indices_samples = indices_descencing_depth[:, m][
                                  :math.ceil(fdatagrid.nsamples * prob[i])]
                samples_used = fdatagrid.data_matrix[indices_samples, :, m]
                max_samples_used = np.amax(samples_used, axis=0)
                min_samples_used = np.amin(samples_used, axis=0)

                if prob[i] == 0.5:
                    # central envelope
                    self._central_envelope[m] = np.asarray(
                        [max_samples_used.T, min_samples_used.T])

                    # outlying envelope
                    max_value = np.amax(fdatagrid.data_matrix[:, :, m], axis=0)
                    min_value = np.amin(fdatagrid.data_matrix[:, :, m], axis=0)
                    iqr = np.absolute(max_samples_used - min_samples_used)
                    outlying_max_envelope = np.minimum(
                        max_samples_used + iqr * factor, max_value)
                    outlying_min_envelope = np.maximum(
                        min_samples_used - iqr * factor, min_value)
                    self._outlying_envelope[m] = np.asarray(
                        [outlying_max_envelope.flatten(),
                         outlying_min_envelope.flatten()])

                    # outliers
                    for j in list(range(fdatagrid.nsamples)):
                        outliers_above = (
                                outlying_max_envelope < fdatagrid.data_matrix[
                                                        j, :, m])
                        outliers_below = (
                                outlying_min_envelope > fdatagrid.data_matrix[
                                                        j, :, m])
                        if (
                                outliers_above.sum() > 0 or outliers_below.sum() > 0):
                            self._outliers[m, j] = 1
                # central regions
                self._central_region[ncentral_regions * m + i] = np.asarray(
                    [max_samples_used.flatten(), min_samples_used.flatten()])

            # mean sample
            self._median[m] = fdatagrid.data_matrix[
                              indices_descencing_depth[0, m], :, m].T

        self._fdatagrid = fdatagrid
        self._prob = prob
        self.colormap = plt.cm.get_cmap('RdPu')
        self.barcol = "blue"
        self.outliercol = "red"
        self.mediancol = "black"
        self.fullout = False

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
    def outlying_envelope(self):
        return self._outlying_envelope

    @property
    def cetral_regions(self):
        return self._cetral_regions

    @property
    def outliers(self):
        return self._outliers

    @property
    def colormap(self):
        return self.colormap

    @colormap.setter
    def colormap(self, colormap):
        if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type matplotlib.colors.LinearSegmentedColormap")
        self.colormap = colormap

    @property
    def barcol(self):
        return self.colormap

    @barcol.setter
    def barcol(self, color):
        self.barcol = color

    @property
    def outliercol(self):
        return self.colormap

    @outliercol.setter
    def outliercol(self, color):
        self.outliercol = color

    @property
    def mediancol(self):
        return self.mediancol

    @mediancol.setter
    def mediancol(self, color):
        self.mediancol = color

    @property
    def fullout(self):
        return self.fullout

    @fullout.setter
    def fullout(self, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("fullout must be boolean type")
        self.fullout = boolean

    def plot(self, fig=None, ax=None, nrows=None, ncols=None):

        fig, ax = self._fdatagrid.generic_plotting_checks(fig, ax, nrows,
                                                          ncols)
        tones = np.linspace(0.1, 1.0, len(self._prob) + 1, endpoint=False)[1:]
        color = self.colormap(tones)

        if self.fullout:
            var_zorder = 4
        else:
            var_zorder = 1

        for m in range(self._fdatagrid.ndim_image):
            # outliers
            for j in list(range(self._fdatagrid.nsamples)):
                if self._outliers[m, j]:
                    ax[m].plot(self._fdatagrid.sample_points[0],
                               self._fdatagrid.data_matrix[j, :, m],
                               color=self._outliercol,
                               linestyle='--', zorder=1)
            for i in range(len(self._prob)):
                # central regions
                ax[m].fill_between(self._fdatagrid.sample_points[0],
                                   self.cetral_regions[
                                       m * len(self._prob) + i, 0],
                                   self.cetral_regions[
                                       m * len(self._prob) + i, 1],
                                   facecolor=color[i],
                                   zorder=var_zorder)

            # outlying envelope
            ax[m].plot(self._fdatagrid.sample_points[0],
                       self._outlying_envelope[m, 0],
                       self._fdatagrid.sample_points[0],
                       self._outlying_envelope[m, 1], color=self._barcol,
                       zorder=4)

            # central envelope
            ax[m].plot(self._fdatagrid.sample_points[0],
                       self.central_envelope[m, 0],
                       self._fdatagrid.sample_points[0],
                       self.central_envelope[m, 1], color=self._barcol,
                       zorder=4)

            # vertical lines
            index = math.ceil(self._fdatagrid.ncol / 2)
            x = self._fdatagrid.sample_points[0][index]
            ax[m].plot([x, x], [self._outlying_envelope[m, 0][index],
                                self.central_envelope[m, 0][index]],
                       color=self._barcol, zorder=4)
            ax[m].plot([x, x], [self._outlying_envelope[m, 1][index],
                                self.central_envelope[m, 1][index]],
                       color=self._barcol, zorder=4)

            # median sample
            ax[m].plot(self._fdatagrid.sample_points[0], self._median[m],
                       color=self._mediancol, zorder=5)

        self._fdatagrid.set_labels(fig, ax)

        return fig, ax

    def _repr_svg_(self):

        fig, _ = self.plot()
        output = StringIO.StringIO()
        fig.savefig(output, format='svg')
        return output.getvalue()

class Boxplot:
    def __init__(self, fdatagrid, method=modified_band_depth, prob=[0.5],
                 factor=1.5):
        if fdatagrid.ndim_domain != 1:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 1.")

        if sorted(prob, reverse=True) != prob:
            raise ValueError(
                "Probabilities required to be in descending order.")

        if min(prob) < 0 or max(prob) > 1:
            raise ValueError("Probabilities must be between 0 and 1.")

        if factor < 0:
            raise ValueError(
                "The number used to calculate the outlying envelope must be positive.")

        nsample_points = len(fdatagrid.sample_points[0])
        ncentral_regions = len(prob)

        self._median = np.ndarray((fdatagrid.ndim_image, nsample_points))
        self._central_envelope = np.ndarray((fdatagrid.ndim_image, 2,
                                             nsample_points))
        self._outlying_env = np.ndarray((fdatagrid.ndim_image, 2,
                                         nsample_points))
        self._cetral_regions = np.ndarray(
            (fdatagrid.ndim_image * ncentral_regions,
             2, nsample_points))
        self._outliers = np.zeros((fdatagrid.ndim_image, fdatagrid.nsamples))

        depth = method(fdatagrid)
        indices_descencing_depth = (-depth).argsort(axis=0)

        for m in range(fdatagrid.ndim_image):

            for i in range(len(prob)):

                indices_samples = indices_descencing_depth[:, m][
                                  :math.ceil(fdatagrid.nsamples * prob[i])]
                samples_used = fdatagrid.data_matrix[indices_samples, :, m]
                max_samples_used = np.amax(samples_used, axis=0)
                min_samples_used = np.amin(samples_used, axis=0)

                if prob[i] == 0.5:
                    # central envelope
                    self._central_envelope[m] = np.asarray(
                        [max_samples_used.T, min_samples_used.T])

                    # outlying envelope
                    max_value = np.amax(fdatagrid.data_matrix[:, :, m], axis=0)
                    min_value = np.amin(fdatagrid.data_matrix[:, :, m], axis=0)
                    iqr = np.absolute(max_samples_used - min_samples_used)
                    outlying_max_envelope = np.minimum(
                        max_samples_used + iqr * factor, max_value)
                    outlying_min_envelope = np.maximum(
                        min_samples_used - iqr * factor, min_value)
                    self._outlying_envelope[m] = np.asarray(
                        [outlying_max_envelope.flatten(),
                         outlying_min_envelope.flatten()])

                    # outliers
                    for j in list(range(fdatagrid.nsamples)):
                        outliers_above = (
                                outlying_max_envelope < fdatagrid.data_matrix[
                                                        j, :, m])
                        outliers_below = (
                                outlying_min_envelope > fdatagrid.data_matrix[
                                                        j, :, m])
                        if (
                                outliers_above.sum() > 0 or outliers_below.sum() > 0):
                            self._outliers[m, j] = 1
                # central regions
                self._central_region[ncentral_regions * m + i] = np.asarray(
                    [max_samples_used.flatten(), min_samples_used.flatten()])

            # mean sample
            self._median[m] = fdatagrid.data_matrix[
                              indices_descencing_depth[0, m], :, m].T

        self._fdatagrid = fdatagrid
        self._prob = prob
        self.colormap = plt.cm.get_cmap('RdPu')
        self.barcol = "blue"
        self.outliercol = "red"
        self.mediancol = "black"
        self.fullout = False

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
    def outlying_envelope(self):
        return self._outlying_envelope

    @property
    def cetral_regions(self):
        return self._cetral_regions

    @property
    def outliers(self):
        return self._outliers

    @property
    def colormap(self):
        return self.colormap

    @colormap.setter
    def colormap(self, colormap):
        if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type matplotlib.colors.LinearSegmentedColormap")
        self.colormap = colormap

    @property
    def barcol(self):
        return self.colormap

    @barcol.setter
    def barcol(self, color):
        self.barcol = color

    @property
    def outliercol(self):
        return self.colormap

    @outliercol.setter
    def outliercol(self, color):
        self.outliercol = color

    @property
    def mediancol(self):
        return self.mediancol

    @mediancol.setter
    def mediancol(self, color):
        self.mediancol = color

    @property
    def fullout(self):
        return self.fullout

    @fullout.setter
    def fullout(self, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("fullout must be boolean type")
        self.fullout = boolean

    def plot(self, fig=None, ax=None, nrows=None, ncols=None):

        fig, ax = self._fdatagrid.generic_plotting_checks(fig, ax, nrows,
                                                          ncols)
        tones = np.linspace(0.1, 1.0, len(self._prob) + 1, endpoint=False)[1:]
        color = self.colormap(tones)

        if self.fullout:
            var_zorder = 4
        else:
            var_zorder = 1

        for m in range(self._fdatagrid.ndim_image):
            # outliers
            for j in list(range(self._fdatagrid.nsamples)):
                if self._outliers[m, j]:
                    ax[m].plot(self._fdatagrid.sample_points[0],
                               self._fdatagrid.data_matrix[j, :, m],
                               color=self._outliercol,
                               linestyle='--', zorder=1)
            for i in range(len(self._prob)):
                # central regions
                ax[m].fill_between(self._fdatagrid.sample_points[0],
                                   self.cetral_regions[
                                       m * len(self._prob) + i, 0],
                                   self.cetral_regions[
                                       m * len(self._prob) + i, 1],
                                   facecolor=color[i],
                                   zorder=var_zorder)

            # outlying envelope
            ax[m].plot(self._fdatagrid.sample_points[0],
                       self._outlying_envelope[m, 0],
                       self._fdatagrid.sample_points[0],
                       self._outlying_envelope[m, 1], color=self._barcol,
                       zorder=4)

            # central envelope
            ax[m].plot(self._fdatagrid.sample_points[0],
                       self.central_envelope[m, 0],
                       self._fdatagrid.sample_points[0],
                       self.central_envelope[m, 1], color=self._barcol,
                       zorder=4)

            # vertical lines
            index = math.ceil(self._fdatagrid.ncol / 2)
            x = self._fdatagrid.sample_points[0][index]
            ax[m].plot([x, x], [self._outlying_envelope[m, 0][index],
                                self.central_envelope[m, 0][index]],
                       color=self._barcol, zorder=4)
            ax[m].plot([x, x], [self._outlying_envelope[m, 1][index],
                                self.central_envelope[m, 1][index]],
                       color=self._barcol, zorder=4)

            # median sample
            ax[m].plot(self._fdatagrid.sample_points[0], self._median[m],
                       color=self._mediancol, zorder=5)

        self._fdatagrid.set_labels(fig, ax)

        return fig, ax

    def _repr_svg_(self):

        fig, _ = self.plot()
        output = StringIO.StringIO()
        fig.savefig(output, format='svg')
        return output.getvalue()

class Boxplot:
    def __init__(self, fdatagrid, method=modified_band_depth, prob=[0.5],
                 factor=1.5):
        if fdatagrid.ndim_domain != 1:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 1.")

        if sorted(prob, reverse=True) != prob:
            raise ValueError(
                "Probabilities required to be in descending order.")

        if min(prob) < 0 or max(prob) > 1:
            raise ValueError("Probabilities must be between 0 and 1.")

        if factor < 0:
            raise ValueError(
                "The number used to calculate the outlying envelope must be positive.")

        nsample_points = len(fdatagrid.sample_points[0])
        ncentral_regions = len(prob)

        self._median = np.ndarray((fdatagrid.ndim_image, nsample_points))
        self._central_envelope = np.ndarray((fdatagrid.ndim_image, 2,
                                             nsample_points))
        self._outlying_env = np.ndarray((fdatagrid.ndim_image, 2,
                                         nsample_points))
        self._cetral_regions = np.ndarray(
            (fdatagrid.ndim_image * ncentral_regions,
             2, nsample_points))
        self._outliers = np.zeros((fdatagrid.ndim_image, fdatagrid.nsamples))

        depth = method(fdatagrid)
        indices_descencing_depth = (-depth).argsort(axis=0)

        for m in range(fdatagrid.ndim_image):

            for i in range(len(prob)):

                indices_samples = indices_descencing_depth[:, m][
                                  :math.ceil(fdatagrid.nsamples * prob[i])]
                samples_used = fdatagrid.data_matrix[indices_samples, :, m]
                max_samples_used = np.amax(samples_used, axis=0)
                min_samples_used = np.amin(samples_used, axis=0)

                if prob[i] == 0.5:
                    # central envelope
                    self._central_envelope[m] = np.asarray(
                        [max_samples_used.T, min_samples_used.T])

                    # outlying envelope
                    max_value = np.amax(fdatagrid.data_matrix[:, :, m], axis=0)
                    min_value = np.amin(fdatagrid.data_matrix[:, :, m], axis=0)
                    iqr = np.absolute(max_samples_used - min_samples_used)
                    outlying_max_envelope = np.minimum(
                        max_samples_used + iqr * factor, max_value)
                    outlying_min_envelope = np.maximum(
                        min_samples_used - iqr * factor, min_value)
                    self._outlying_envelope[m] = np.asarray(
                        [outlying_max_envelope.flatten(),
                         outlying_min_envelope.flatten()])

                    # outliers
                    for j in list(range(fdatagrid.nsamples)):
                        outliers_above = (
                                outlying_max_envelope < fdatagrid.data_matrix[
                                                        j, :, m])
                        outliers_below = (
                                outlying_min_envelope > fdatagrid.data_matrix[
                                                        j, :, m])
                        if (
                                outliers_above.sum() > 0 or outliers_below.sum() > 0):
                            self._outliers[m, j] = 1
                # central regions
                self._central_region[ncentral_regions * m + i] = np.asarray(
                    [max_samples_used.flatten(), min_samples_used.flatten()])

            # mean sample
            self._median[m] = fdatagrid.data_matrix[
                              indices_descencing_depth[0, m], :, m].T

        self._fdatagrid = fdatagrid
        self._prob = prob
        self.colormap = plt.cm.get_cmap('RdPu')
        self.barcol = "blue"
        self.outliercol = "red"
        self.mediancol = "black"
        self.fullout = False

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
    def outlying_envelope(self):
        return self._outlying_envelope

    @property
    def cetral_regions(self):
        return self._cetral_regions

    @property
    def outliers(self):
        return self._outliers

    @property
    def colormap(self):
        return self.colormap

    @colormap.setter
    def colormap(self, colormap):
        if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type matplotlib.colors.LinearSegmentedColormap")
        self.colormap = colormap

    @property
    def barcol(self):
        return self.colormap

    @barcol.setter
    def barcol(self, color):
        self.barcol = color

    @property
    def outliercol(self):
        return self.colormap

    @outliercol.setter
    def outliercol(self, color):
        self.outliercol = color

    @property
    def mediancol(self):
        return self.mediancol

    @mediancol.setter
    def mediancol(self, color):
        self.mediancol = color

    @property
    def fullout(self):
        return self.fullout

    @fullout.setter
    def fullout(self, boolean):
        if not isinstance(boolean, bool):
            raise ValueError("fullout must be boolean type")
        self.fullout = boolean

    def plot(self, fig=None, ax=None, nrows=None, ncols=None):

        fig, ax = self._fdatagrid.generic_plotting_checks(fig, ax, nrows,
                                                          ncols)
        tones = np.linspace(0.1, 1.0, len(self._prob) + 1, endpoint=False)[1:]
        color = self.colormap(tones)

        if self.fullout:
            var_zorder = 4
        else:
            var_zorder = 1

        for m in range(self._fdatagrid.ndim_image):
            # outliers
            for j in list(range(self._fdatagrid.nsamples)):
                if self._outliers[m, j]:
                    ax[m].plot(self._fdatagrid.sample_points[0],
                               self._fdatagrid.data_matrix[j, :, m],
                               color=self._outliercol,
                               linestyle='--', zorder=1)
            for i in range(len(self._prob)):
                # central regions
                ax[m].fill_between(self._fdatagrid.sample_points[0],
                                   self.cetral_regions[
                                       m * len(self._prob) + i, 0],
                                   self.cetral_regions[
                                       m * len(self._prob) + i, 1],
                                   facecolor=color[i],
                                   zorder=var_zorder)

            # outlying envelope
            ax[m].plot(self._fdatagrid.sample_points[0],
                       self._outlying_envelope[m, 0],
                       self._fdatagrid.sample_points[0],
                       self._outlying_envelope[m, 1], color=self._barcol,
                       zorder=4)

            # central envelope
            ax[m].plot(self._fdatagrid.sample_points[0],
                       self.central_envelope[m, 0],
                       self._fdatagrid.sample_points[0],
                       self.central_envelope[m, 1], color=self._barcol,
                       zorder=4)

            # vertical lines
            index = math.ceil(self._fdatagrid.ncol / 2)
            x = self._fdatagrid.sample_points[0][index]
            ax[m].plot([x, x], [self._outlying_envelope[m, 0][index],
                                self.central_envelope[m, 0][index]],
                       color=self._barcol, zorder=4)
            ax[m].plot([x, x], [self._outlying_envelope[m, 1][index],
                                self.central_envelope[m, 1][index]],
                       color=self._barcol, zorder=4)

            # median sample
            ax[m].plot(self._fdatagrid.sample_points[0], self._median[m],
                       color=self._mediancol, zorder=5)

        self._fdatagrid.set_labels(fig, ax)

        return fig, ax

    def _repr_svg_(self):

        fig, _ = self.plot()
        output = StringIO.StringIO()
        fig.savefig(output, format='svg')
        return output.getvalue()


class SurfaceBoxplot:
    def __init__(self, fdatagrid, method=modified_band_depth, factor=1.5):
        if fdatagrid.ndim_domain != 2:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 2.")

        if factor < 0:
            raise ValueError(
                "The number used to calculate the outlying envelope must be positive.")

        nsample_points = len(fdatagrid.sample_points[0])

        self._median = np.ndarray((fdatagrid.ndim_image, nsample_points))
        self._central_envelope = np.ndarray((fdatagrid.ndim_image, 2,
                                             nsample_points))
        self._outlying_env = np.ndarray((fdatagrid.ndim_image, 2,
                                         nsample_points))
        self._cetral_regions = np.ndarray((fdatagrid.ndim_image, 2,
                                           nsample_points))

        depth = method(fdatagrid)
        indices_descencing_depth = (-depth).argsort(axis=0)

        for m in range(fdatagrid.ndim_image):

            indices_samples = indices_descencing_depth[:, m][
                              :math.ceil(fdatagrid.nsamples * 0.5)]
            samples_used = fdatagrid.data_matrix[indices_samples, :, :, m]
            max_samples_used = np.amax(samples_used, axis=0)
            min_samples_used = np.amin(samples_used, axis=0)

            # mean sample
            self._median[m] = fdatagrid.data_matrix[
                              indices_descencing_depth[0, m], :, :, m]

            # central envelope
            self._central_envelope[m] = np.asarray([max_samples_used,
                                                    min_samples_used])

            # outlying envelope
            max_value = np.amax(fdatagrid.data_matrix[:, :, :, m], axis=0)
            min_value = np.amin(fdatagrid.data_matrix[:, :, :, m], axis=0)
            iqr = np.absolute(max_samples_used - min_samples_used)
            oulying_max_envelope = np.minimum(max_samples_used + iqr * factor,
                                              max_value)
            oulying_min_envelope = np.maximum(min_samples_used - iqr * factor,
                                              min_value)
            self._outlying_env[m] = np.asarray([oulying_max_envelope,
                                                oulying_min_envelope])


        self._fdatagrid = fdatagrid
        self.colormap = plt.cm.get_cmap('RdPu')
        self.boxcol= 1.0
        self.outcol=0.7

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
    def outlying_envelope(self):
        return self._outlying_envelope

    @property
    def colormap(self):
        return self.colormap

    @colormap.setter
    def colormap(self, colormap):
        if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type matplotlib.colors.LinearSegmentedColormap")
        self.colormap = colormap

    @property
    def boxcol(self):
        return self.boxcol

    @boxcol.setter
    def boxcol(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                "boxcol must be a number between 0 and 1.")

        self.boxcol = value

    @property
    def outcol(self):
        return self.outcol

    @outcol.setter
    def outcol(self, value):
        if value < 0 or value > 1:
            raise ValueError(
                "outcol must be a number between 0 and 1.")
        self.outcol = value

    def plot(self, fig=None, ax=None, nrows=None, ncols=None):

        fig, ax = self._fdatagrid.generic_plotting_checks(fig, ax, nrows,
                                                          ncols)
        x = self._fdatagrid.sample_points[0]
        lx = len(x)
        y = self._fdatagrid.sample_points[1]
        ly = len(y)
        X, Y = np.meshgrid(x, y)

        for m in range(self._fdatagrid.ndim_image):

            # mean sample
            ax[m].plot_wireframe(X, Y, np.squeeze(self._median[m]).T,
                                 rstride=ly, cstride=lx,
                                 color=self._colormap(self._boxcol))
            ax[m].plot_surface(X, Y, np.squeeze(self._median[m]).T,
                               color=self._colormap(self._boxcol), alpha=0.8)

            # central envelope
            ax[m].plot_surface(X, Y, np.squeeze(self._central_envelope[m, 0]).T,
                               color=self._colormap(self._boxcol), alpha=0.5)
            ax[m].plot_wireframe(X, Y, np.squeeze(self._central_envelope[m, 0]).T,
                                 rstride=ly, cstride=lx,
                                 color=self._colormap(self._boxcol))
            ax[m].plot_surface(X, Y, np.squeeze(self._central_envelope[m, 1]).T,
                               color=self._colormap(self._boxcol), alpha=0.5)
            ax[m].plot_wireframe(X, Y, np.squeeze(self._central_envelope[m, 1]).T,
                                 rstride=ly, cstride=lx,
                                 color=self._colormap(self._boxcol))


            # box vertical lines
            for indices in [(0, 0), (0, ly - 1), (lx - 1, 0),
                            (lx - 1, ly - 1)]:
                x_corner = x[indices[0]]
                y_corner = y[indices[1]]
                ax[m].plot([x_corner, x_corner], [y_corner, y_corner],
                           [self._central_envelope[m, 1, indices[0], indices[1]],
                            self._central_envelope[m, 0, indices[0], indices[1]]],
                           color=self._colormap(self._boxcol))

            # outlying envelope
            ax[m].plot_surface(X, Y, np.squeeze(self.outlying_env[m, 0]).T,
                               color=self._colormap(self._outcol), alpha=0.3)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.outlying_env[m, 0]).T,
                                 rstride=ly, cstride=lx,
                                 color=self._colormap(self._outcol))
            ax[m].plot_surface(X, Y, np.squeeze(self.outlying_env[m, 1]).T,
                               color=self._colormap(self._outcol), alpha=0.3)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.outlying_env[m, 1]).T,
                                 rstride=ly, cstride=lx,
                                 color=self._colormap(self._outcol))

            # vertical lines from central to outlying envelope
            x_index = math.floor(lx / 2)
            x_central = x[x_index]
            y_index = math.floor(ly / 2)
            y_central = y[y_index]
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self._outlying_env[m, 1, x_index, y_index],
                        self._central_envelope[m, 1, x_index, y_index]],
                       color=self.central_regions_col[0])
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self._outlying_env[m, 0, x_index, y_index],
                        self._central_envelope[m, 0, x_index, y_index]],
                       color=self._colormap(self._boxcol))

        self._fdatagrid.set_labels(fig, ax)

        return fig, ax

    def _repr_svg_(self):

        fig, _ = self.plot()
        output = StringIO.StringIO()
        fig.savefig(output, format='svg')
        return output.getvalue()


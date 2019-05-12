"""Functional Data Boxplot Module.

This module contains the classes to construct the functional data boxplot and
visualize it.

"""
import matplotlib
import matplotlib.pyplot as plt
import math

import numpy as np

from skfda.exploratory.depth import modified_band_depth
from ... import FDataGrid
from io import BytesIO
from abc import ABC, abstractmethod

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"

class FDataBoxplot(ABC):
    """Abstract class inherited by the Boxplot and SurfaceBoxplot classes.

    It the data of the functional boxplot or surface boxplot of a FDataGrid object,
    depending on the dimensions of the domain, 1 or 2 respectively.

    It forces to both classes, Boxplot and SurfaceBoxplot to conain at least the median,
    central and outlying envelopes and a colormap for their graphical representation,
    obtained calling the plot method.

    """
    @abstractmethod
    def __init__(self, factor=1.5):
        if factor < 0:
            raise ValueError(
                "The number used to calculate the outlying envelope must be positive.")
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
    def outlying_envelope(self):
        pass

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if not isinstance(value, matplotlib.colors.LinearSegmentedColormap):
            raise ValueError(
                "colormap must be of type matplotlib.colors.LinearSegmentedColormap")
        self._colormap = value

    @abstractmethod
    def plot(self, fig=None, ax=None, nrows=None, ncols=None):
        pass

    def _repr_svg_(self):
        plt.figure()
        fig, _ = self.plot()
        output = BytesIO()
        fig.savefig(output, format='svg')
        data = output.getvalue()
        plt.close(fig)
        return data.decode('utf-8')


class Boxplot(FDataBoxplot):
    r"""Representation of the functional boxplot.

    Class implementing the functionl boxplot which is an informative exploratory
    tool for visualizing functional data, as well as its generalization, the
    enhanced functional boxplot. Only supports 1 dimensional domain functional data.

    Based on the center outward ordering induced by a :ref:`depth measure <depth-measures>`
    for functional data, the descriptive statistics of a functional boxplot are: the
    envelope of the 50% central region, the median curve,and the maximum non-outlying envelope.
    In addition, outliers can be detected in a functional boxplot by the 1.5 times the 50%
    central region empirical rule, analogous to the rule for classical boxplots.

    Attributes:
        fdatagrid (FDataGrid): Object containing the data.
        median (array, (fdatagrid.ndim_image, nsample_points)): contains
            the median/s.
        central_envelope (array, (fdatagrid.ndim_image, 2, nsample_points)):
            contains the central envelope/s.
        outlying_envelope (array, (fdatagrid.ndim_image, 2, nsample_points)):
            contains the outlying envelope/s.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from
            which the colors to represent the central regions are selected.
        central_regions (array, (fdatagrid.ndim_image * ncentral_regions, 2,
            nsample_points)): contains the central regions.
        outliers (array, (fdatagrid.ndim_image, fdatagrid.nsamples)):
            contains the outliers
        barcol (string): Color of the envelopes and vertical lines.
        outliercol (string): Color of the ouliers.
        mediancol (string): Color of the median.
        show_full_outliers (boolean): If False (the default) then only the part
            outside the box is plotted. If True, complete outling curves are plotted

    Example:
        Function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

        >>> data_matrix = [[1, 1, 2, 3, 2.5, 2], [0.5, 0.5, 1, 2, 1.5, 1], [-1, -1, -0.5, 1, 1, 0.5],
        ...                [-0.5, -0.5, -0.5, -1, -1, -1]]
        >>> sample_points = [0, 2, 4, 6, 8, 10]
        >>> fd = FDataGrid(data_matrix, sample_points, dataset_label="dataset", axes_labels=["x_label", "y_label"])
        >>> Boxplot(fd)
        Boxplot(
            FDataGrid=FDataGrid(
                array([[[ 1. ],
                        [ 1. ],
                        [ 2. ],
                        [ 3. ],
                        [ 2.5],
                        [ 2. ]],
        <BLANKLINE>
                       [[ 0.5],
                        [ 0.5],
                        [ 1. ],
                        [ 2. ],
                        [ 1.5],
                        [ 1. ]],
        <BLANKLINE>
                       [[-1. ],
                        [-1. ],
                        [-0.5],
                        [ 1. ],
                        [ 1. ],
                        [ 0.5]],
        <BLANKLINE>
                       [[-0.5],
                        [-0.5],
                        [-0.5],
                        [-1. ],
                        [-1. ],
                        [-1. ]]]),
                sample_points=[array([ 0,  2,  4,  6,  8, 10])],
                domain_range=array([[ 0, 10]]),
                dataset_label='dataset',
                axes_labels=['x_label', 'y_label'],
                extrapolation=None,
                interpolator=SplineInterpolator(interpolation_order=1, smoothness_parameter=0.0, monotone=False),
                keepdims=False),
            median=array([[ 0.5,  0.5,  1. ,  2. ,  1.5,  1. ]]),
            central envelope=array([[[ 0.5,  0.5,  1. ,  2. ,  1.5,  1. ],
                    [-1. , -1. , -0.5,  1. ,  1. ,  0.5]]]),
            outlying envelope=array([[[ 1.  ,  1.  ,  2.  ,  3.  ,  2.25,  1.75],
                    [-1.  , -1.  , -0.5 , -0.5 ,  0.25, -0.25]]]),
            central_regions=array([[[ 0.5,  0.5,  1. ,  2. ,  1.5,  1. ],
                    [-1. , -1. , -0.5,  1. ,  1. ,  0.5]]]),
            outliers=array([[ 1.,  0.,  0.,  1.]]))

    """

    def __init__(self, fdatagrid, method=modified_band_depth, prob=[0.5],
                 factor=1.5):
        """Initialization of the Boxplot class.

        Args:
            fdatagrid (FDataGrid): Object containing the data.
            method (:ref:`depth measure <depth-measures>`, optional): Method
                used to order the data. Defaults to :func:`modified band depth
                <fda.depth_measures.modified_band_depth>`.
            prob (list of float, optional): List with float numbers (in the range
                from 1 to 0) that indicate which central regions to represent.
                Defaults to [0.5] which represents the 50% central region.
            factor (double): Number used to calculate the outlying envelope.

        """
        FDataBoxplot.__init__(self, factor)

        if fdatagrid.ndim_domain != 1:
            raise ValueError(
                "Function only supports FDataGrid with domain dimension 1.")

        if sorted(prob, reverse=True) != prob:
            raise ValueError(
                "Probabilities required to be in descending order.")

        if min(prob) < 0 or max(prob) > 1:
            raise ValueError("Probabilities must be between 0 and 1.")

        nsample_points = len(fdatagrid.sample_points[0])
        ncentral_regions = len(prob)

        self._median = np.ndarray((fdatagrid.ndim_image, nsample_points))
        self._central_envelope = np.ndarray((fdatagrid.ndim_image, 2,
                                             nsample_points))
        self._outlying_envelope = np.ndarray((fdatagrid.ndim_image, 2,
                                              nsample_points))
        self._central_regions = np.ndarray(
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
                self._central_regions[ncentral_regions * m + i] = np.asarray(
                    [max_samples_used.flatten(), min_samples_used.flatten()])

            # mean sample
            self._median[m] = fdatagrid.data_matrix[
                              indices_descencing_depth[0, m], :, m].T

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
    def outlying_envelope(self):
        return self._outlying_envelope

    @property
    def central_regions(self):
        return self._central_regions

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

    def plot(self, fig=None, ax=None, nrows=None, ncols=None):
        """Visualization of the functional boxplot of the fdatagrid (ndim_domain=1).

        Args:
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also None,
                the figure is initialized.
            ax (list of axis objects, optional): axis over where the graphs are
                plotted. If None, see param fig.
            nrows(int, optional): designates the number of rows of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.
            ncols(int, optional): designates the number of columns of the figure
                to plot the different dimensions of the image. Only specified
                if fig and ax are None.

         Returns:
            fig (figure object): figure object in which the graphs are plotted.
            ax (axes object): axes in which the graphs are plotted.

        """

        fig, ax = self.fdatagrid.generic_plotting_checks(fig, ax, nrows,
                                                         ncols)
        tones = np.linspace(0.1, 1.0, len(self._prob) + 1, endpoint=False)[1:]
        color = self.colormap(tones)

        if self.show_full_outliers:
            var_zorder = 1
        else:
            var_zorder = 4

        for m in range(self.fdatagrid.ndim_image):

            # outliers
            for j in list(range(self.fdatagrid.nsamples)):
                if self.outliers[m, j]:
                    ax[m].plot(self.fdatagrid.sample_points[0],
                               self.fdatagrid.data_matrix[j, :, m],
                               color=self.outliercol,
                               linestyle='--', zorder=1)

            for i in range(len(self._prob)):
                # central regions
                ax[m].fill_between(self.fdatagrid.sample_points[0],
                                   self.central_regions[
                                       m * len(self._prob) + i, 0],
                                   self.central_regions[
                                       m * len(self._prob) + i, 1],
                                   facecolor=color[i], zorder=var_zorder)

            # outlying envelope
            ax[m].plot(self.fdatagrid.sample_points[0],
                       self.outlying_envelope[m, 0],
                       self.fdatagrid.sample_points[0],
                       self.outlying_envelope[m, 1], color=self.barcol,
                       zorder=4)

            # central envelope
            ax[m].plot(self.fdatagrid.sample_points[0],
                       self.central_envelope[m, 0],
                       self.fdatagrid.sample_points[0],
                       self.central_envelope[m, 1], color=self.barcol,
                       zorder=4)

            # vertical lines
            index = math.ceil(self.fdatagrid.ncol / 2)
            x = self.fdatagrid.sample_points[0][index]
            ax[m].plot([x, x], [self.outlying_envelope[m, 0][index],
                                self.central_envelope[m, 0][index]],
                       color=self.barcol,
                       zorder=4)
            ax[m].plot([x, x], [self.outlying_envelope[m, 1][index],
                                self.central_envelope[m, 1][index]],
                       color=self.barcol, zorder=4)

            # median sample
            ax[m].plot(self.fdatagrid.sample_points[0], self.median[m],
                       color=self.mediancol, zorder=5)

        self.fdatagrid.set_labels(fig, ax)

        return fig, ax

    def __repr__(self):
        """Return repr(self)."""
        return (f"Boxplot("
                f"\nFDataGrid={repr(self.fdatagrid)},"
                f"\nmedian={repr(self.median)},"
                f"\ncentral envelope={repr(self.central_envelope)},"
                f"\noutlying envelope={repr(self.outlying_envelope)},"
                f"\ncentral_regions={repr(self.central_regions)},"
                f"\noutliers={repr(self.outliers)})").replace('\n', '\n    ')


class SurfaceBoxplot(FDataBoxplot):
    r"""Representation of the surface boxplot.

    Class implementing the surface boxplot. Analogously to the functional boxplot,
    it is an informative exploratory tool for visualizing functional data with
    domain dimension 2. Nevertheless, it does not implement the enhanced
    surface boxplot.

    Based on the center outward ordering induced by a :ref:`depth measure <depth-measures>`
    for functional data, it represents the envelope of the 50% central region, the median curve,
    and the maximum non-outlying envelope.

    Attributes:
        fdatagrid (FDataGrid): Object containing the data.
        median (array, (fdatagrid.ndim_image, lx, ly)): contains
            the median/s.
        central_envelope (array, (fdatagrid.ndim_image, 2, lx, ly)):
            contains the central envelope/s.
        outlying_envelope (array,(fdatagrid.ndim_image, 2, lx, ly)):
            contains the outlying envelope/s.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from
            which the colors to represent the central regions are selected.
        boxcol (string): Color of the box, which includes median and central envelope.
        outcol (string): Color of the outlying envelope.

    Example:
        Function :math:`f : \mathbb{R^2}\longmapsto\mathbb{R^2}`.

        >>> data_matrix = [[[[1, 4], [0.3, 1.5], [1, 3]], [[2, 8], [0.4, 2], [2, 9]]],
        ...                [[[2, 10], [0.5, 3], [2, 10]], [[3, 12], [0.6, 3], [3, 15]]]]
        >>> sample_points = [[2, 4], [3, 6, 8]]
        >>> fd = FDataGrid(data_matrix, sample_points, dataset_label= "dataset",
        ...                axes_labels=["x1_label", "x2_label", "y1_label", "y2_label"])
        >>> SurfaceBoxplot(fd)
        SurfaceBoxplot(
            FDataGrid=FDataGrid(
                array([[[[  1. ,   4. ],
                         [  0.3,   1.5],
                         [  1. ,   3. ]],
        <BLANKLINE>
                        [[  2. ,   8. ],
                         [  0.4,   2. ],
                         [  2. ,   9. ]]],
        <BLANKLINE>
        <BLANKLINE>
                       [[[  2. ,  10. ],
                         [  0.5,   3. ],
                         [  2. ,  10. ]],
        <BLANKLINE>
                        [[  3. ,  12. ],
                         [  0.6,   3. ],
                         [  3. ,  15. ]]]]),
                sample_points=[array([2, 4]), array([3, 6, 8])],
                domain_range=array([[2, 4],
                       [3, 8]]),
                dataset_label='dataset',
                axes_labels=['x1_label', 'x2_label', 'y1_label', 'y2_label'],
                extrapolation=None,
                interpolator=SplineInterpolator(interpolation_order=1, smoothness_parameter=0.0, monotone=False),
                keepdims=False),
            median=array([[[ 1. ,  0.3,  1. ],
                    [ 2. ,  0.4,  2. ]],
        <BLANKLINE>
                   [[ 4. ,  1.5,  3. ],
                    [ 8. ,  2. ,  9. ]]]),
            central envelope=array([[[[ 1. ,  0.3,  1. ],
                     [ 2. ,  0.4,  2. ]],
        <BLANKLINE>
                    [[ 1. ,  0.3,  1. ],
                     [ 2. ,  0.4,  2. ]]],
        <BLANKLINE>
        <BLANKLINE>
                   [[[ 4. ,  1.5,  3. ],
                     [ 8. ,  2. ,  9. ]],
        <BLANKLINE>
                    [[ 4. ,  1.5,  3. ],
                     [ 8. ,  2. ,  9. ]]]]),
            outlying envelope=array([[[[ 1. ,  0.3,  1. ],
                     [ 2. ,  0.4,  2. ]],
        <BLANKLINE>
                    [[ 1. ,  0.3,  1. ],
                     [ 2. ,  0.4,  2. ]]],
        <BLANKLINE>
        <BLANKLINE>
                   [[[ 4. ,  1.5,  3. ],
                     [ 8. ,  2. ,  9. ]],
        <BLANKLINE>
                    [[ 4. ,  1.5,  3. ],
                     [ 8. ,  2. ,  9. ]]]]))


    """

    def __init__(self, fdatagrid, method=modified_band_depth, factor=1.5):
        """Initialization of the functional boxplot.

        Args:
            fdatagrid (FDataGrid): Object containing the data.
            method (:ref:`depth measure <depth-measures>`, optional): Method
                used to order the data. Defaults to :func:`modified band depth
                <fda.depth_measures.modified_band_depth>`.
            prob (list of float, optional): List with float numbers (in the range
                from 1 to 0) that indicate which central regions to represent.
                Defaults to [0.5] which represents the 50% central region.
            factor (double): Number used to calculate the outlying envelope.

        """
        FDataBoxplot.__init__(self, factor)

        if fdatagrid.ndim_domain != 2:
            raise ValueError(
                "Class only supports FDataGrid with domain dimension 2.")

        lx = len(fdatagrid.sample_points[0])
        ly = len(fdatagrid.sample_points[1])

        self._median = np.ndarray((fdatagrid.ndim_image, lx, ly))
        self._central_envelope = np.ndarray((fdatagrid.ndim_image, 2, lx, ly))
        self._outlying_envelope = np.ndarray((fdatagrid.ndim_image, 2, lx, ly))

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
            self._outlying_envelope[m] = np.asarray([oulying_max_envelope,
                                                     oulying_min_envelope])

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
    def outlying_envelope(self):
        return self._outlying_envelope

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

    def plot(self, fig=None, ax=None, nrows=None, ncols=None):
        """Visualization of the surface boxplot of the fdatagrid (ndim_domain=2).

         Args:
             fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also None,
                the figure is initialized.
             ax (list of axis objects, optional): axis over where the graphs are
                plotted. If None, see param fig.
             nrows(int, optional): designates the number of rows of the figure
                 to plot the different dimensions of the image. Only specified
                 if fig and ax are None.
             ncols(int, optional): designates the number of columns of the figure
                 to plot the different dimensions of the image. Only specified
                 if fig and ax are None.

        Returns:
             fig (figure object): figure object in which the graphs are plotted.
             ax (axes object): axes in which the graphs are plotted.

        """
        fig, ax = self.fdatagrid.generic_plotting_checks(fig, ax, nrows,
                                                         ncols)
        x = self.fdatagrid.sample_points[0]
        lx = len(x)
        y = self.fdatagrid.sample_points[1]
        ly = len(y)
        X, Y = np.meshgrid(x, y)

        for m in range(self.fdatagrid.ndim_image):

            # mean sample
            ax[m].plot_wireframe(X, Y, np.squeeze(self.median[m]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.colormap(self.boxcol))
            ax[m].plot_surface(X, Y, np.squeeze(self.median[m]).T,
                               color=self.colormap(self.boxcol), alpha=0.8)

            # central envelope
            ax[m].plot_surface(X, Y, np.squeeze(self.central_envelope[m, 0]).T,
                               color=self.colormap(self.boxcol), alpha=0.5)
            ax[m].plot_wireframe(X, Y,
                                 np.squeeze(self.central_envelope[m, 0]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.colormap(self.boxcol))
            ax[m].plot_surface(X, Y, np.squeeze(self.central_envelope[m, 1]).T,
                               color=self.colormap(self.boxcol), alpha=0.5)
            ax[m].plot_wireframe(X, Y,
                                 np.squeeze(self.central_envelope[m, 1]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.colormap(self.boxcol))

            # box vertical lines
            for indices in [(0, 0), (0, ly - 1), (lx - 1, 0),
                            (lx - 1, ly - 1)]:
                x_corner = x[indices[0]]
                y_corner = y[indices[1]]
                ax[m].plot([x_corner, x_corner], [y_corner, y_corner],
                           [self.central_envelope[
                                m, 1, indices[0], indices[1]],
                            self.central_envelope[
                                m, 0, indices[0], indices[1]]],
                           color=self.colormap(self.boxcol))

            # outlying envelope
            ax[m].plot_surface(X, Y,
                               np.squeeze(self.outlying_envelope[m, 0]).T,
                               color=self.colormap(self.outcol), alpha=0.3)
            ax[m].plot_wireframe(X, Y,
                                 np.squeeze(self.outlying_envelope[m, 0]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.colormap(self.outcol))
            ax[m].plot_surface(X, Y,
                               np.squeeze(self.outlying_envelope[m, 1]).T,
                               color=self.colormap(self.outcol), alpha=0.3)
            ax[m].plot_wireframe(X, Y,
                                 np.squeeze(self.outlying_envelope[m, 1]).T,
                                 rstride=ly, cstride=lx,
                                 color=self.colormap(self.outcol))

            # vertical lines from central to outlying envelope
            x_index = math.floor(lx / 2)
            x_central = x[x_index]
            y_index = math.floor(ly / 2)
            y_central = y[y_index]
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self.outlying_envelope[m, 1, x_index, y_index],
                        self.central_envelope[m, 1, x_index, y_index]],
                       color=self.colormap(self.boxcol))
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self.outlying_envelope[m, 0, x_index, y_index],
                        self.central_envelope[m, 0, x_index, y_index]],
                       color=self.colormap(self.boxcol))

        self.fdatagrid.set_labels(fig, ax)

        return fig, ax

    def __repr__(self):
        """Return repr(self)."""
        return (f"SurfaceBoxplot("
                f"\nFDataGrid={repr(self.fdatagrid)},"
                f"\nmedian={repr(self.median)},"
                f"\ncentral envelope={repr(self.central_envelope)},"
                f"\noutlying envelope={repr(self.outlying_envelope)})").replace('\n', '\n    ')

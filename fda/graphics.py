"""Graphics Module to visualize FDataGrid."""

#from mpl_toolkits.mplot3d import Axes3D
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import variation
from scipy import stats
import scipy
import math

from .depth_measures import *
from .grid import FDataGrid

__author__ = "Amanda Hernando Bernabé"
__email__ = "amanda.hernando@estudiante.uam.es"

from fda.depth_measures import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import variation
from scipy import stats
import scipy
import math


class FDataBoxplotInfo:
    """Class containing the data of the functional boxplot or surface boxplot of an FDataGrid object,
    depending on the dimensions of the domain, 1 or 2 respectively.

        Class for representing functional data as a set of curves discretised
        in a grid of points.

    Attributes:
        data_matrix (numpy.ndarray): a matrix where each entry of the first
                axis contains the values of a functional datum evaluated at the
                points of discretisation.
            sample_points (numpy.ndarray): 2 dimension matrix where each row
                contains the points of dicretisation for each axis of data_matrix.
            sample_range (numpy.ndarray): 2 dimension matrix where each row
                contains the bounds of the interval in which the functional data
                is considered to exist for each one of the axies.
            dataset_label (str): name of the dataset.
            axes_labels (list): list containing the labels of the different
                axis.

        Examples:
            Representation of a functional data object with 2 samples
            representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}`.

            >>> data_matrix = [[1, 2], [2, 3]]
            >>> sample_points = [2, 4]
            >>> FDataGrid(data_matrix, sample_points)
            FDataGrid(
                array([[[1],
                        [2]],
            <BLANKLINE>
                       [[2],
                        [3]]]),
                sample_points=[array([2, 4])],
                ...)

            The number of columns of data_matrix have to be the length of
            sample_points.

            >>> FDataGrid(numpy.array([1,2,4,5,8]), range(6))
            Traceback (most recent call last):
                ....
            ValueError: Incorrect dimension in data_matrix and sample_points...



            FDataGrid support higher dimensional data both in the domain and image.
            Representation of a functional data object with 2 samples
            representing a function :math:`f : \mathbb{R}\longmapsto\mathbb{R}^2`.

            >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
            >>> sample_points = [2, 4]
            >>> fd = FDataGrid(data_matrix, sample_points)
            >>> fd.ndim_domain, fd.ndim_image
            (1, 2)

            Representation of a functional data object with 2 samples

            >>> data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
            >>> sample_points = [[2, 4], [3,6]]
            >>> fd = FDataGrid(data_matrix, sample_points)
            >>> fd.ndim_domain, fd.ndim_image
            (2, 1)

        """
    def __init__(self, fdatagrid, central_regions_col, envelopes_col=None,
                 outliers_col=None, median_col=None):
        self.fdatagrid = fdatagrid
        self.ncentral_regions = len(central_regions_col)
        self.central_regions_col = central_regions_col
        self.envelopes_col = envelopes_col
        self.outliers_col = outliers_col
        self.median_col = median_col
        self.median = np.ndarray([])
        self.central_env = np.ndarray([])
        self.outlying_env = np.ndarray([])
        self.central_regions = np.ndarray([])
        self.outliers = np.ndarray([])

    def add_median(self, median):
        if len(self.median.shape) == 0:
            self.median = np.asarray(median)
        else:
            self.median = np.append(self.median, np.asarray(median), axis=0)

    def add_central_env(self, central_env):
        if len(self.central_env.shape) == 0:
            self.central_env = np.asarray(central_env)
        else:
            self.central_env = np.append(self.central_env, np.asarray(central_env), axis=0)

    def add_outlying_env(self, outlying_env):
        if len(self.outlying_env.shape) == 0:
            self.outlying_env = np.asarray(outlying_env)
        else:
            self.outlying_env = np.append(self.outlying_env, np.asarray(outlying_env), axis=0)

    def add_central_region(self, central_region):
        if len(self.central_regions.shape) == 0:
            self.central_regions = np.asarray(central_region)
        else:
            self.central_regions = np.append(self.central_regions, np.asarray(central_region), axis=0)

    def _visualise_fdboxplot(self):
        sample_points = self.fdatagrid.sample_points[0]
        if self.median.shape != (self.fdatagrid.ndim_image, len(sample_points)):
            raise ValueError("There must be the same number of medians as dimensions of the image of fdatagrid"
                             " and each one must have nsample_points.")

        if self.central_env.shape != (self.fdatagrid.ndim_image, 2, len(sample_points)):
            raise ValueError("There must be the same number of central envelopes"
                             "as dimensions of the image of fdatagrid"
                             " and each one must have nsample_points.")

        if self.outlying_env.shape != (self.fdatagrid.ndim_image, 2, len(sample_points)):
            raise ValueError("There must be the same number of outlying envelopes"
                             "as dimensions of the image of fdatagrid"
                             " and each one must have nsample_points.")

        if self.central_regions.shape != (self.fdatagrid.ndim_image * self.ncentral_regions, 2, len(sample_points)):
            raise ValueError("There must be the same number of central regions"
                             "as dimensions of the image of fdatagrid multiplied by the number of"
                             "central regions at each axis.")

        fig, ax = self.fdatagrid.set_figure_and_axes()

        for m in range(self.fdatagrid.ndim_image):
            for n in range(self.fdatagrid.nsamples):
                # outliers
                if self.outliers[m, n]:
                    ax[m].plot(sample_points, self.fdatagrid.data_matrix[n, :, m],
                               color=self.outliers_col, linestyle='--')
                    # central regions
            for n in range(self.ncentral_regions):
                ax[m].fill_between(sample_points, self.central_regions[m * self.ncentral_regions + n, 0],
                                   self.central_regions[m * self.ncentral_regions + n, 1],
                                   facecolor=self.central_regions_col[n])
            # outlying envelope
            ax[m].plot(sample_points, self.outlying_env[m, 0],
                       sample_points, self.outlying_env[m, 1], color=self.envelopes_col)
            # central envelope
            ax[m].plot(sample_points, self.central_env[m, 0], sample_points,
                       self.central_env[m, 1], color=self.envelopes_col)
            # vertical lines joining outlying envelope and central envelope
            index = math.ceil(self.fdatagrid.ncol / 2)
            x = sample_points[index]
            ax[m].plot([x, x], [self.outlying_env[m, 0, index], self.central_env[m, 0, index]],
                       color=self.envelopes_col)
            ax[m].plot([x, x], [self.outlying_env[m, 1, index], self.central_env[m, 1, index]],
                       color=self.envelopes_col)
            # median
            ax[m].plot(sample_points, self.median[m], color=self.median_col)

        self.fdatagrid.set_labels(fig)
        self.fdatagrid.arrange_layout(fig)

        return self

    def _visualise_surface_boxplot(self):
        x = self.fdatagrid.sample_points[0]
        lx = len(x)
        y = self.fdatagrid.sample_points[1]
        ly = len(y)
        X, Y = np.meshgrid(x, y)

        if self.median.shape != (self.fdatagrid.ndim_image, lx, ly):
            raise ValueError("There must be the same number of medians as dimensions of the image of fdatagrid"
                             " and each one must have nsample_points.")

        if self.central_env.shape != (self.fdatagrid.ndim_image, 2, lx, ly):
            raise ValueError("There must be the same number of central envelopes"
                             "as dimensions of the image of fdatagrid"
                             " and each one must have nsample_points.")

        if self.outlying_env.shape != (self.fdatagrid.ndim_image, 2, lx, ly):
            raise ValueError("There must be the same number of outlying envelopes"
                             "as dimensions of the image of fdatagrid"
                             " and each one must have nsample_points.")

        fig, ax = self.fdatagrid.set_figure_and_axes()

        for m in range(self.fdatagrid.ndim_image):
            # median
            ax[m].plot_wireframe(X, Y, np.squeeze(self.median[m]).T, rstride=ly, cstride=lx, color=self.central_regions_col[0])
            ax[m].plot_surface(X, Y, np.squeeze(self.median[m]).T, color=self.central_regions_col[0], alpha=0.8)

            # central envelope
            ax[m].plot_surface(X, Y, np.squeeze(self.central_env[m, 0]).T, color=self.central_regions_col[0], alpha=0.5)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.central_env[m, 0]).T, rstride=ly, cstride=lx,
                                 color=self.central_regions_col[0])
            ax[m].plot_surface(X, Y, np.squeeze(self.central_env[m, 1]).T, color=self.central_regions_col[0], alpha=0.5)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.central_env[m, 1]).T, rstride=ly, cstride=lx,
                                 color=self.central_regions_col[0])

            # box vertical lines
            for indices in [(0, 0), (0, ly - 1), (lx - 1, 0), (lx - 1, ly - 1)]:
                x_corner = x[indices[0]]
                y_corner = y[indices[1]]
                ax[m].plot([x_corner, x_corner], [y_corner, y_corner], [self.central_env[m, 1, indices[0], indices[1]],
                                                                        self.central_env[m, 0, indices[0], indices[1]]],
                           color=self.central_regions_col[0])

            # outlying envelope
            ax[m].plot_surface(X, Y, np.squeeze(self.outlying_env[m, 0]).T, color=self.envelopes_col, alpha=0.3)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.outlying_env[m, 0]).T, rstride=ly, cstride=lx,
                                 color=self.envelopes_col)
            ax[m].plot_surface(X, Y, np.squeeze(self.outlying_env[m, 1]).T, color=self.envelopes_col, alpha=0.3)
            ax[m].plot_wireframe(X, Y, np.squeeze(self.outlying_env[m, 1]).T, rstride=ly, cstride=lx,
                                 color=self.envelopes_col)

            # vertical lines from central to outlying envelope
            x_index = math.floor(lx / 2)
            x_central = x[x_index]
            y_index = math.floor(ly / 2)
            y_central = y[y_index]
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self.outlying_env[m, 1, x_index, y_index], self.central_env[m, 1, x_index, y_index]],
                       color=self.central_regions_col[0])
            ax[m].plot([x_central, x_central], [y_central, y_central],
                       [self.outlying_env[m, 0, x_index, y_index], self.central_env[m, 0, x_index, y_index]],
                       color=self.central_regions_col[0])

        self.fdatagrid.set_labels(fig)
        self.fdatagrid.arrange_layout(fig)

        return self

    def visualise(self):
        if self.fdatagrid.ndim_domain > 2:
            raise ValueError("Function only implemented if the dimension of the domain is 1 or 2.")
        if self.fdatagrid.ndim_domain == 1:
            self._visualise_fdboxplot()
        else:
            self._visualise_surface_boxplot()


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


def fdboxplot(fdatagrid, fig=None, method=modified_band_depth, prob=[0.5], fullout=False, factor=1.5, barcol="blue",
              colormap=plt.cm.get_cmap('RdPu'), outliercol="red", mediancol="black"):
    """Implementation of the functional boxplot.

    Args:
        fdatagrid (FDataGrid): Object to be visualized.
		fig (figure object, optional): figure where the graphs are plotted.
			Defaults to matplotlib.pyplot.figure.
        method (depth function, optional): Method used to order the data (Fraiman_Muniz_depth, band_depth, modified_band_depth).
			Defaults to modified_band_depth.
        prob (list of float, optional): List with float numbers (in the range from 1 to 0) that indicate which central regions to
			represent. Defaults to [0.5] which represents the 0.5 central region.
		fullout (boolean): If true, the entire curve of the outlier samples is shown.
		factor (double): Number used to calculate the outlying envelope.
        colormap (matplotlib.colors.LinearSegmentedColormap): Colormap from which the colors to represent the central regions are selected.
        outliercol (string): Color of the outliers.
        barcol (string): Color of the envelopes and vertical lines.

    Returns:
        _plot: List of mpl_toolkits.mplot3d.art3d.Poly3DCollection, mpl_toolkits.mplot3d.art3d.Line3DCollection and
            mpl_toolkits.mplot3d.art3d.Line3D.
        outliers: List of lenght number of samples containing 1 if the sample is an outlier, 0 otherwise.

    """
    if fdatagrid.ndim_domain > 1:
        raise ValueError("Function only supports FDataGrid with domain dimension 1.")

    if ((method != Fraiman_Muniz_depth) & (method != band_depth) & (method != modified_band_depth)):
        raise ValueError("Undefined function.")

    if sorted(prob, reverse=True) != prob:
        raise ValueError("Probabilities required to be in descending order.")

    if min(prob) < 0 or max(prob) > 1:
        raise ValueError("Probabilities must be between 0 and 1.")

    if not isinstance(colormap, matplotlib.colors.LinearSegmentedColormap):
        raise ValueError("colormap must be of type matplotlib.colors.LinearSegmentedColormap")

    # Obtaining the necessary colors of the colormap.
    tones = np.linspace(0.1, 1.0, len(prob) + 1, endpoint=False)[1:]
    color = colormap(tones)

    if fullout:
        var_zorder = 4
    else:
        var_zorder = 1

    depth = method(fdatagrid)
    indices_descencing_depth = (-depth).argsort(axis=0)

    if fig == None:
        fig, ax = fdatagrid.set_figure_and_axes()

    _plot = FDataBoxplotInfo(fdatagrid, color, barcol, outliercol, mediancol)

    for m in range(fdatagrid.ndim_image):

        for i in range(len(prob)):

            indices_samples = indices_descencing_depth[:, m][:math.ceil(fdatagrid.nsamples * prob[i])]
            samples_used = fdatagrid.data_matrix[indices_samples, :, m]
            max_samples_used = np.amax(samples_used, axis=0)
            min_samples_used = np.amin(samples_used, axis=0)

            if prob[i] == 0.5:
                # central envelope
                ax[m].plot(fdatagrid.sample_points[0], max_samples_used.T, fdatagrid.sample_points[0],
                           min_samples_used.T, color=barcol, zorder=4)
                _plot.add_central_env([[max_samples_used.T, min_samples_used.T]])

                # outlying envelope
                max_value = np.amax(fdatagrid.data_matrix[:, :, m], axis=0)
                min_value = np.amin(fdatagrid.data_matrix[:, :, m], axis=0)
                iqr = np.absolute(max_samples_used - min_samples_used)
                outlying_max_envelope = np.minimum(max_samples_used + iqr * factor, max_value)
                outlying_min_envelope = np.maximum(min_samples_used - iqr * factor, min_value)
                ax[m].plot(fdatagrid.sample_points[0], outlying_max_envelope.flatten(),
                           fdatagrid.sample_points[0], outlying_min_envelope.flatten(), color=barcol, zorder=4)
                _plot.add_outlying_env([[outlying_max_envelope.flatten(), outlying_min_envelope.flatten()]])

                # vertical lines
                index = math.ceil(fdatagrid.ncol / 2)
                x = fdatagrid.sample_points[0][index]
                ax[m].plot([x, x], [outlying_max_envelope[index], max_samples_used[index]], color=barcol, zorder=4)
                ax[m].plot([x, x], [outlying_min_envelope[index], min_samples_used[index]], color=barcol, zorder=4)

                # outliers
                _plot.outliers = np.zeros((fdatagrid.ndim_image, fdatagrid.nsamples))
                for j in list(range(fdatagrid.nsamples)):
                    outliers_above = (outlying_max_envelope < fdatagrid.data_matrix[j, :, m])
                    outliers_below = (outlying_min_envelope > fdatagrid.data_matrix[j, :, m])
                    if (outliers_above.sum() > 0 or outliers_below.sum() > 0):
                        _plot.outliers[m, j] = 1
                        ax[m].plot(fdatagrid.sample_points[0], fdatagrid.data_matrix[j, :, m], color=outliercol,
                                   linestyle='--', zorder=1)

                        # central regions
            ax[m].fill_between(fdatagrid.sample_points[0], max_samples_used.flatten(),
                               min_samples_used.flatten(), facecolor=color[i], zorder=var_zorder)
            _plot.add_central_region([[max_samples_used.flatten(), min_samples_used.flatten()]])

        # mean sample
        ax[m].plot(fdatagrid.sample_points[0], fdatagrid.data_matrix[indices_descencing_depth[0, m], :, m].T,
                   color=mediancol,
                   zorder=5)
        _plot.add_median([fdatagrid.data_matrix[indices_descencing_depth[0, m], :, m].T])

    fdatagrid.set_labels(fig)
    fdatagrid.arrange_layout(fig)

    return _plot

def surface_boxplot(fdatagrid, fig=None, method=modified_band_depth, factor=1.5, boxcol="black", outcol="grey"):
    """Implementation of the surface boxplot.

    Args:
        fdatagrid (FDataGrid): Object to be visualized.
		fig (figure object, optional): aigure where the graphs are plotted.
			Defaults to matplotlib.pyplot.figure.
        method (depth function, optional): Method used to order the data (Fraiman_Muniz_depth, band_depth, modified_band_depth).
			Defaults to modified_band_depth.
		factor (double): Number used to calculate the outlying envelope.
        boxcol (string): Color of the box: mean, central envelopes and vertical lines.
        outboxcol (string): Color of the outlying envelopes.

    Returns:
        _plot: List of mpl_toolkits.mplot3d.art3d.Poly3DCollection, mpl_toolkits.mplot3d.art3d.Line3DCollection and
                mpl_toolkits.mplot3d.art3d.Line3D.

    """
    if fdatagrid.ndim_domain != 2:
        raise ValueError("Function only supports FDataGrid with domain dimension 2.")

    if ((method != Fraiman_Muniz_depth) & (method != band_depth) & (method != modified_band_depth)):
        raise ValueError("Undefined function.")

    depth = method(fdatagrid)
    indices_descencing_depth = (-depth).argsort(axis=0)

    if fig == None:
        fig, ax = fdatagrid.set_figure_and_axes()

    x = fdatagrid.sample_points[0]
    lx = len(x)
    y = fdatagrid.sample_points[1]
    ly = len(y)
    X, Y = np.meshgrid(x, y)

    _plot = FDataBoxplotInfo(fdatagrid, central_regions_col = [boxcol], envelopes_col=outcol)

    for m in range(fdatagrid.ndim_image):

        indices_samples = indices_descencing_depth[:, m][:math.ceil(fdatagrid.nsamples * 0.5)]
        samples_used = fdatagrid.data_matrix[indices_samples, :, :, m]
        max_samples_used = np.amax(samples_used, axis=0)
        min_samples_used = np.amin(samples_used, axis=0)

        # mean sample
        ax[m].plot_wireframe(X, Y, np.squeeze(fdatagrid.data_matrix[indices_descencing_depth[0, m], :, :, m]).T,
                             rstride=ly, cstride=lx, color=boxcol)
        ax[m].plot_surface(X, Y, np.squeeze(fdatagrid.data_matrix[indices_descencing_depth[0, m], :, :, m]).T,
                           color=boxcol, alpha=0.8)
        _plot.add_median([fdatagrid.data_matrix[indices_descencing_depth[0, m], :, :, m]])

        # central envelope
        ax[m].plot_surface(X, Y, np.squeeze(max_samples_used).T, color=boxcol, alpha=0.5)
        ax[m].plot_wireframe(X, Y, np.squeeze(max_samples_used).T, rstride=ly, cstride=lx, color=boxcol)
        ax[m].plot_surface(X, Y, np.squeeze(min_samples_used).T, color=boxcol, alpha=0.5)
        ax[m].plot_wireframe(X, Y, np.squeeze(min_samples_used).T, rstride=ly, cstride=lx, color=boxcol)
        _plot.add_central_env([[max_samples_used, min_samples_used]])

        # box vertical lines
        for indices in [(0, 0), (0, ly - 1), (lx - 1, 0), (lx - 1, ly - 1)]:
            x_corner = x[indices[0]]
            y_corner = y[indices[1]]
            ax[m].plot([x_corner, x_corner], [y_corner, y_corner],[min_samples_used[indices[0], indices[1]],
                                                                   max_samples_used[indices[0], indices[1]]], color=boxcol)

        # outlying envelope
        max_value = np.amax(fdatagrid.data_matrix[:, :, :, m], axis=0)
        min_value = np.amin(fdatagrid.data_matrix[:, :, :, m], axis=0)
        iqr = np.absolute(max_samples_used - min_samples_used)
        oulying_max_envelope = np.minimum(max_samples_used + iqr * factor, max_value)
        oulying_min_envelope = np.maximum(min_samples_used - iqr * factor, min_value)
        ax[m].plot_surface(X, Y, np.squeeze(oulying_max_envelope).T, color=outcol, alpha=0.3)
        ax[m].plot_wireframe(X, Y, np.squeeze(oulying_max_envelope).T, rstride=ly, cstride=lx, color=outcol)
        ax[m].plot_surface(X, Y, np.squeeze(oulying_min_envelope).T, color=outcol, alpha=0.3)
        ax[m].plot_wireframe(X, Y, np.squeeze(oulying_min_envelope).T, rstride=ly, cstride=lx, color=outcol)
        _plot.add_outlying_env([[oulying_max_envelope, oulying_min_envelope]])

        # vertical lines from central to outlying envelope
        x_index = math.floor(lx / 2)
        x_central = x[x_index]
        y_index = math.floor(ly / 2)
        y_central = y[y_index]
        ax[m].plot([x_central, x_central], [y_central, y_central],
                                [oulying_max_envelope[x_index, y_index], max_samples_used[x_index, y_index]],
                                color=boxcol)
        ax[m].plot([x_central, x_central], [y_central, y_central],
                                [oulying_min_envelope[x_index, y_index], min_samples_used[x_index, y_index]],
                                color=boxcol)

    fdatagrid.set_labels(fig)
    fdatagrid.arrange_layout(fig)

    return _plot

def magnitude_shape_plot(fdatagrid, ax=None, depth_method=modified_band_depth, dim_weights=None, pointwise_weights=None):
    """Implementation of the magnitude shape plot which is based on the
    calculation of the directional outlyingness of each of the samples.

    Args:
        fdatagrid (FDataGrid): Object to be visualized.
    	ax (axis object, optional): axis over which the graph is plotted.
                Defaults to matplotlib current axis.
        depth_method (depth function, optional): Method used to order the data (Fraiman_Muniz_depth,
        band_depth, modified_band_depth). Defaults to modified_band_depth.
    	dim_weights (array_like, optional): an array containing the weights of each of
    	the dimensions of the image.
        pointwise_weights (array_like, optional): an array containing the weights of each
            point of discretisation where values have been recorded.

    Returns:
        points(numpy.ndarray): 2 dimension matrix where each row
            contains the points plotted in the graph.

    """
    if fdatagrid.ndim_image > 1:
        raise NotImplementedError("Only support 1 dimension on the image.")

    # The depths of the samples are calculated giving them an ordering.
    mean_dir_outl, variation_dir_outl = directional_outlyingness(fdatagrid, depth_method, dim_weights, pointwise_weights)
    points = np.array(list(zip(mean_dir_outl, variation_dir_outl)))

    cov = MinCovDet(store_precision=True).fit(points)
    rbd_2 = cov.mahalanobis(points)
    s_jj = np.diag(cov.covariance_)
    c = np.mean(s_jj)
    m = 2 / np.square(variation(s_jj))
    p = fdatagrid.ndim_image
    dfn = p + 1
    dfd = m - p
    q = 0.993
    cutoff_value = stats.f.ppf(q, dfn, dfd, loc=0, scale=1)
    scaling = c * (m - p) / m / (p + 1)
    outliers = (scaling * rbd_2 > cutoff_value)
    outliers = outliers.astype(str)
    outliers[np.where(outliers == 'True')] = 'red'
    outliers[np.where(outliers == 'False')] = 'blue'

    if ax is None:
        ax = matplotlib.pyplot.gca()
    ax.scatter(mean_dir_outl, variation_dir_outl, color=outliers)
    ax.set_xlabel('MO')
    ax.set_ylabel('VO')
    ax.set_title('MS-Plot')

    return points


def clustering(fdatagrid, ax=None, num_clusters=2):
    if fdatagrid.ndim_image > 1 and fdatagrid.ndim_domain:
        raise NotImplementedError("Only support 1 dimension on the image and on the domain.")

    if num_clusters < 2 or num_clusters > 10:
        raise ValueError("The number of clusters must be between 2 and 10, both included.")

    repetitions = 0
    distances_to_centers = np.empty((fdatagrid.nsamples, num_clusters))
    centers = np.empty((num_clusters, len(fdatagrid.sample_points[0])))
    centers_aux = np.empty((num_clusters, len(fdatagrid.sample_points[0])))

    for i in range(num_clusters):
        centers[i] = fdatagrid.data_matrix[math.floor(i * fdatagrid.nsamples / num_clusters)].flatten()

    while not np.array_equal(centers, centers_aux) and repetitions < 100:
        centers_aux = centers
        for i in range(fdatagrid.nsamples):
            for j in range(num_clusters):
                #pairwise distance
                distances_to_centers[i, j] = scipy.spatial.distance.euclidean(fdatagrid.data_matrix[i], centers[j])
        clustering_values = np.argmin(distances_to_centers, axis=1)
        for i in range(num_clusters):
            indices = np.where(clustering_values == i)
            centers[i] = np.average(fdatagrid.data_matrix[indices, :, :].flatten()
                                    .reshape((len(indices[0]), len(fdatagrid.sample_points[0]))), axis=0)
        repetitions += 1

    colors_samples = np.empty(fdatagrid.nsamples).astype(str)
    for i in range(num_clusters):
        colors_samples[np.where(clustering_values == i)] = "C{}".format(i)

    if ax is None:
        ax = matplotlib.pyplot.gca()

    for i in range(fdatagrid.nsamples):
        ax.plot(fdatagrid.sample_points[0], fdatagrid.data_matrix[i, :, 0].T, color=colors_samples[i])

    return clustering_values
"""Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from abc import ABC, abstractmethod

import numpy as np

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import mpl_toolkits.mplot3d

from skfda.representation.extrapolation import _parse_extrapolation
from .._utils import _coordinate_list, _list_of_arrays


class FData(ABC):
    """Defines the structure of a functional data object.

    Attributes:
        nsamples (int): Number of samples.
        ndim_domain (int): Dimension of the domain.
        ndim_image (int): Dimension of the image.
        extrapolation (Extrapolation): Default extrapolation mode.
        dataset_label (str): name of the dataset.
        axes_labels (list): list containing the labels of the different
            axis. The first element is the x label, the second the y label
            and so on.
        keepdims (bool): Default value of argument keepdims in
            :func:`evaluate`.

    """

    def __init__(self, extrapolation, dataset_label, axes_labels, keepdims):

        self.extrapolation = extrapolation
        self.dataset_label = dataset_label
        self.axes_labels = axes_labels
        self.keepdims = keepdims

    @property
    @abstractmethod
    def nsamples(self):
        """Return the number of samples.

        Returns:
            int: Number of samples of the FData object.

        """
        pass

    @property
    @abstractmethod
    def ndim_domain(self):
        """Return number of dimensions of the domain.

        Returns:
            int: Number of dimensions of the domain.

        """
        pass

    @property
    @abstractmethod
    def ndim_image(self):
        """Return number of dimensions of the image.

        Returns:
            int: Number of dimensions of the image.

        """
        pass

    @property
    def ndim_codomain(self):
        """Return number of dimensions of the codomain.

        Returns:
            int: Number of dimensions of the codomain.

        """
        return self.ndim_image

    @property
    def extrapolation(self):
        """Return default type of extrapolation."""

        return self._extrapolation

    @extrapolation.setter
    def extrapolation(self, value):
        """Sets the type of extrapolation."""

        if value is None:
            self._extrapolation = None
        else:
            self._extrapolation = _parse_extrapolation(value)

        self._extrapolator_evaluator = None

    @property
    def extrapolator_evaluator(self):
        """Return the evaluator constructed by the extrapolator."""
        if self.extrapolation is None:
            return None

        elif self._extrapolator_evaluator is None:
            self._extrapolator_evaluator = self._extrapolation.evaluator(self)

        return self._extrapolator_evaluator

    @property
    @abstractmethod
    def domain_range(self):
        """Return the domain range of the object

        Returns:
            List of tuples with the ranges for each domain dimension.
        """
        pass

    def _reshape_eval_points(self, eval_points, evaluation_aligned):
        """Convert and reshape the eval_points to ndarray with the corresponding
        shape.

        Args:
            eval_points (array_like): Evaluation points to be reshaped.
            evaluation_aligned (bool): Boolean flag. True if all the samples
                will be evaluated at the same evaluation_points.

        Returns:
            (np.ndarray): Numpy array with the eval_points, if
            evaluation_aligned is True with shape `number of evaluation points`
            x `ndim_domain`. If the points are not aligned the shape of the
            points will be `nsamples` x `number of evaluation points`
            x `ndim_domain`.

        """

        # Case evaluation of a scalar value, i.e., f(0)
        if np.isscalar(eval_points):
            eval_points = [eval_points]

        # Creates a copy of the eval points, and convert to np.array
        eval_points = np.array(eval_points, dtype=float)

        if evaluation_aligned: # Samples evaluated at same eval points

            eval_points = eval_points.reshape((eval_points.shape[0],
                                               self.ndim_domain))

        else: # Different eval_points for each sample

            if eval_points.ndim < 2 or eval_points.shape[0] != self.nsamples:

                raise ValueError(f"eval_points should be a list "
                                 f"of length {self.nsamples} with the "
                                 f"evaluation points for each sample.")

            eval_points = eval_points.reshape((eval_points.shape[0],
                                               eval_points.shape[1],
                                               self.ndim_domain))

        return eval_points

    def _extrapolation_index(self, eval_points):
        """Checks the points that need to be extrapolated.

        Args:
            eval_points (np.ndarray): Array with shape `n_eval_points` x
                `ndim_domain` with the evaluation points, or shape ´nsamples´ x
                `n_eval_points` x `ndim_domain` with different evaluation points
                for each sample.

        Returns:

            (np.ndarray): Array with boolean index. The positions with True
                in the index are outside the domain range and extrapolation
                should be applied.

        """
        index = np.zeros(eval_points.shape[:-1], dtype=np.bool)

        # Checks bounds in each domain dimension
        for i, bounds in enumerate(self.domain_range):
            np.logical_or(index, eval_points[..., i] < bounds[0], out=index)
            np.logical_or(index, eval_points[..., i] > bounds[1], out=index)

        return index


    def _evaluate_grid(self, axes, *, derivative=0, extrapolation=None,
                       aligned_evaluation=True, keepdims=None):
        """Evaluate the functional object in the cartesian grid.

        This method is called internally by :meth:`evaluate` when the argument
        `grid` is True.

        Evaluates the functional object in the grid generated by the cartesian
        product of the axes. The length of the list of axes should be equal than
        the domain dimension of the object.

        If the list of axes has lengths :math:`n_1, n_2, ..., n_m`, where
        :math:`m` is equal than the dimension of the domain, the result of the
        evaluation in the grid will be a matrix with :math:`m+1` dimensions and
        shape :math:`n_{samples} x n_1 x n_2 x ... x n_m`.

        If `aligned_evaluation` is false each sample is evaluated in a different
        grid, and the list of axes should contain a list of axes for each
        sample.

        If the domain dimension is 1, the result of the behaviour of the
        evaluation will be the same than :meth:`evaluate` without the grid
        option, but with worst performance.

        Args:
            axes (array_like): List of axes to generated the grid where the
                object will be evaluated.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            aligned_evaluation (bool, optional): If False evaluates each sample
                in a different grid.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.
                By default is used the value given during the instance of the
                object.

        Returns:
            (numpy.darray): Numpy array with ndim_domain + 1 dimensions with the
             result of the evaluation.

        Raises:
            ValueError: If there are a different number of axes than the domain
                dimension.

        """
        axes = _list_of_arrays(axes)

        if aligned_evaluation:

            lengths = [len(ax) for ax in axes]

            if len(axes) != self.ndim_domain:
                raise ValueError(f"Length of axes should be {self.ndim_domain}")

            eval_points = _coordinate_list(axes)

            res = self.evaluate(eval_points, derivative=derivative,
                                extrapolation=extrapolation, keepdims=True)

        elif self.ndim_domain == 1:

            eval_points = [ax.squeeze(0) for ax in axes]

            return self.evaluate(eval_points, derivative=derivative,
                                 extrapolation=extrapolation, keepdims=keepdims,
                                 aligned_evaluation=False)
        else:

            if len(axes) != self.nsamples:
                raise ValueError("Should be provided a list of axis per sample")
            elif len(axes[0]) != self.ndim_domain:
                raise ValueError(f"Incorrect length of axes. "
                                 f"({self.ndim_domain}) != {len(axes[0])}")

            lengths = [len(ax) for ax in axes[0]]
            eval_points = np.empty((self.nsamples,
                                       np.prod(lengths),
                                       self.ndim_domain))

            for i in range(self.nsamples):
                eval_points[i] = _coordinate_list(axes[i])

            res = self.evaluate(eval_points, derivative=derivative,
                                extrapolation=extrapolation,
                                keepdims=True, aligned_evaluation=False)

        shape = [self.nsamples] + lengths

        if keepdims is None:
            keepdims = self.keepdims

        if self.ndim_image != 1 or keepdims:
            shape += [self.ndim_image]

        # Roll the list of result in a list
        return res.reshape(shape)


    def _join_evaluation(self, index_matrix, index_ext, index_ev,
                         res_extrapolation, res_evaluation):
        """Join the points evaluated.

        This method is used internally by :func:`evaluate` to join the result of
        the evaluation and the result of the extrapolation.

        Args:
            index_matrix (ndarray): Boolean index with the points extrapolated.
            index_ext (ndarray): Boolean index with the columns that contains
                points extrapolated.
            index_ev (ndarray): Boolean index with the columns that contains
                points evaluated.
            res_extrapolation (ndarray): Result of the extrapolation.
            res_evaluation (ndarray): Result of the evaluation.

        Returns:
            (ndarray): Matrix with the points evaluated with shape
            `nsamples` x `number of points evaluated` x `ndim_image`.

        """
        res = np.empty((self.nsamples, index_matrix.shape[-1],
                           self.ndim_image))

        # Case aligned evaluation
        if index_matrix.ndim == 1:
            res[:, index_ev, :] = res_evaluation
            res[:, index_ext, :] = res_extrapolation

        else:

            res[:, index_ev] = res_evaluation
            res[index_matrix] = res_extrapolation[index_matrix[:, index_ext]]


        return res

    @abstractmethod
    def _evaluate(self, eval_points, *, derivative=0):
        """Internal evaluation method, defines the evaluation of the FData.

        Evaluates the samples of an FData object at the same eval_points.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is True.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(len(eval_points), ndim_domain)` with the evaluation points.
                Each entry represents the coordinate of a point.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Numpy 3d array with shape `(n_samples,
                len(eval_points), ndim_image)` with the result of the evaluation.
                The entry (i,j,k) will contain the value k-th image dimension of
                the i-th sample, at the j-th evaluation point.

        """
        pass

    @abstractmethod
    def _evaluate_composed(self, eval_points, *, derivative=0):
        """Internal evaluation method, defines the evaluation of a FData.

        Evaluates the samples of an FData object at different eval_points.

        This method is called internally by :meth:`evaluate` when the argument
        `aligned_evaluation` is False.

        Args:
            eval_points (numpy.ndarray): Numpy array with shape
                `(n_samples, len(eval_points), ndim_domain)` with the evaluation
                 points for each sample.
            derivative (int, optional): Order of the derivative. Defaults to 0.

        Returns:
            (numpy.darray): Numpy 3d array with shape `(n_samples,
                len(eval_points), ndim_image)` with the result of the evaluation.
                The entry (i,j,k) will contain the value k-th image dimension of
                the i-th sample, at the j-th evaluation point.

        """
        pass


    def evaluate(self, eval_points, *, derivative=0, extrapolation=None,
                 grid=False, aligned_evaluation=True, keepdims=None):
        """Evaluate the object or its derivatives at a list of values or a grid.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            grid (bool, optional): Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the input
                arrays. If true the eval_points should be a list of size
                ndim_domain with the corresponding times for each axis. The
                return matrix has shape nsamples x len(t1) x len(t2) x ... x
                len(t_ndim_domain) x ndim_image. If the domain dimension is 1
                the parameter has no efect. Defaults to False.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.
                By default is used the value given during the instance of the
                object.

        Returns:
            (np.darray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        if extrapolation is None:
            extrapolation = self.extrapolation
            extrapolator_evaluator = self.extrapolator_evaluator
        else:
            # Gets the function to perform extrapolation or None
            extrapolation = _parse_extrapolation(extrapolation)
            extrapolator_evaluator  = None

        if grid: # Evaluation of a grid performed in auxiliar function
            return self._evaluate_grid(eval_points, derivative=derivative,
                                       extrapolation=extrapolation,
                                       aligned_evaluation=aligned_evaluation,
                                       keepdims=keepdims)

        # Convert to array and check dimensions of eval points
        eval_points = self._reshape_eval_points(eval_points, aligned_evaluation)

        # Check if extrapolation should be applied
        if extrapolation is not None:
            index_matrix = self._extrapolation_index(eval_points)
            extrapolate = index_matrix.any()

        else:
            extrapolate = False

        if not extrapolate: # Direct evaluation

            if aligned_evaluation:
                res = self._evaluate(eval_points, derivative=derivative)
            else:
                res = self._evaluate_composed(eval_points,
                                              derivative=derivative)

        else:
            # Evaluation using extrapolation
            if extrapolator_evaluator is None:
                extrapolator_evaluator = extrapolation.evaluator(self)

            # Partition of eval points
            if aligned_evaluation:

                index_ext = index_matrix
                index_ev = ~index_matrix

                eval_points_extrapolation = eval_points[index_ext]
                eval_points_evaluation = eval_points[index_ev]

                # Direct evaluation
                res_evaluation = self._evaluate(eval_points_evaluation,
                                                derivative=derivative)
                res_extrapolation = extrapolator_evaluator.evaluate(
                    eval_points_extrapolation,
                    derivative=derivative)

            else:
                index_ext = np.logical_or.reduce(index_matrix, axis=0)
                eval_points_extrapolation = eval_points[:, index_ext]

                index_ev = np.logical_or.reduce(~index_matrix, axis=0)
                eval_points_evaluation = eval_points[:, index_ev]

                # Direct evaluation
                res_evaluation = self._evaluate_composed(eval_points_evaluation,
                                                         derivative=derivative)

                res_extrapolation = extrapolator_evaluator.evaluate_composed(
                    eval_points_extrapolation,
                    derivative=derivative)

            res = self._join_evaluation(index_matrix, index_ext, index_ev,
                                        res_extrapolation, res_evaluation)

        # If not provided gets default value of keepdims
        if keepdims is None:
            keepdims = self.keepdims

        # Delete last axis if not keepdims and
        if self.ndim_image == 1 and not keepdims:
            res = res.reshape(res.shape[:-1])

        return res


    def __call__(self, eval_points, *, derivative=0, extrapolation=None,
                 grid=False, aligned_evaluation=True, keepdims=None):
        """Evaluate the object or its derivatives at a list of values or a grid.
        This method is a wrapper of :meth:`evaluate`.

        Args:
            eval_points (array_like): List of points where the functions are
                evaluated. If a matrix of shape nsample x eval_points is given
                each sample is evaluated at the values in the corresponding row
                in eval_points.
            derivative (int, optional): Order of the derivative. Defaults to 0.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range. By
                default it is used the mode defined during the instance of the
                object.
            grid (bool, optional): Whether to evaluate the results on a grid
                spanned by the input arrays, or at points specified by the input
                arrays. If true the eval_points should be a list of size
                ndim_domain with the corresponding times for each axis. The
                return matrix has shape nsamples x len(t1) x len(t2) x ... x
                len(t_ndim_domain) x ndim_image. If the domain dimension is 1
                the parameter has no efect. Defaults to False.
            keepdims (bool, optional): If the image dimension is equal to 1 and
                keepdims is True the return matrix has shape
                nsamples x eval_points x 1 else nsamples x eval_points.
                By default is used the value given during the instance of the
                object.

        Returns:
            (np.ndarray): Matrix whose rows are the values of the each
            function at the values specified in eval_points.

        """
        return self.evaluate(eval_points, derivative=derivative,
                             extrapolation=extrapolation, grid=grid,
                             aligned_evaluation=aligned_evaluation,
                             keepdims=keepdims)

    @abstractmethod
    def derivative(self, order=1):
        """Differentiate a FData object.


        Args:
            order (int, optional): Order of the derivative. Defaults to one.

        Returns:
            :class:`FData`: Functional object containg the derivative.
        """
        pass

    @abstractmethod
    def shift(self, shifts, *, restrict_domain=False, extrapolation=None,
              discretization_points=None, **kwargs):
        """Perform a shift of the curves.

        Args:
            shifts (array_like or numeric): List with the shift corresponding
                for each sample or numeric with the shift to apply to all
                samples.
            restrict_domain (bool, optional): If True restricts the domain to
                avoid evaluate points outside the domain using extrapolation.
                Defaults uses extrapolation.
            extrapolation (str or Extrapolation, optional): Controls the
                extrapolation mode for elements outside the domain range.
                By default uses the method defined in fd. See extrapolation to
                more information.
            discretization_points (array_like, optional): Set of points where
                the functions are evaluated to obtain the discrete
                representation of the object to operate. If an empty list is
                passed it calls np.linspace with bounds equal to the ones
                defined in fd.domain_range and the number of points the maximum
                between 201 and 10 times the number of basis plus 1.

        Returns:
            :class:`FData` with the shifted functional data.
        """
        pass

    def set_figure_and_axes(self, nrows, ncols):
        """Set figure and its axes.

        Args:
            nrows(int, optional): designates the number of rows of the figure to
                plot the different dimensions of the image. ncols must be also
                be customized in the same call.
            ncols(int, optional): designates the number of columns of the figure
                to plot the different dimensions of the image. nrows must be
                also be customized in the same call.

        Returns:
            fig (figure object): figure object initialiazed.
            ax (axes object): axes of the initialized figure.

        """

        if self.ndim_domain == 1:
            projection = None
        else:
            projection = '3d'

        if ncols is None and nrows is None:
            ncols = int(np.ceil(np.sqrt(self.ndim_image)))
            nrows = int(np.ceil(self.ndim_image / ncols))
        elif ncols is None and nrows is not None:
            nrows = int(np.ceil(self.ndim_image / nrows))
        elif ncols is not None and nrows is None:
            nrows = int(np.ceil(self.ndim_image / ncols))

        fig = plt.gcf()
        axes = fig.get_axes()

        # If it is not empty
        if len(axes) != 0:
            # Gets geometry of current fig
            geometry = (fig.axes[0]
                        .get_subplotspec()
                        .get_topmost_subplotspec()
                        .get_gridspec().get_geometry())

            # Check if the projection of the axes is the same
            if projection == '3d':
                same_projection = all(a.name == '3d' for a in axes)
            else:
                same_projection = all(a.name == 'rectilinear' for a in axes)

            # If compatible uses the same figure
            if (same_projection and geometry == (nrows, ncols) and
                len(axes) == self.ndim_image):
                 return fig, axes

            else: # Create new figure if it is not compatible
                fig = plt.figure()


        for i in range(self.ndim_image):
            fig.add_subplot(nrows, ncols, i + 1, projection=projection)

        if ncols > 1 and self.axes_labels is not None and self.ndim_image > 1:
            plt.subplots_adjust(wspace=0.4)

        if nrows > 1 and self.axes_labels is not None and self.ndim_image > 1:
            plt.subplots_adjust(hspace=0.4)

        ax = fig.get_axes()

        return fig, ax

    def set_labels(self, fig=None, ax=None, patches=None):
        """Set labels if any.

        Args:
            fig (figure object): figure object containing the axes that
                implement set_xlabel and set_ylabel, and set_zlabel in case
                of a 3d projection.
            ax (list of axes): axes objects that implement set_xlabel and
                set_ylabel, and set_zlabel in case of a 3d projection; used if
                fig is None.
            patches (list of mpatches.Patch); objects used to generate each
                entry in the legend.

        """
        if fig is not None:
            if self.dataset_label is not None:
                fig.suptitle(self.dataset_label)
            ax = fig.get_axes()
            if patches is not None and self.ndim_image > 1:
                fig.legend(handles=patches)
            elif patches is not None:
                ax[0].legend(handles=patches)
        elif len(ax) == 1:
            if self.dataset_label is not None:
                ax[0].set_title(self.dataset_label)
            if patches is not None:
                ax[0].legend(handles=patches)


        if self.axes_labels is not None:
            if ax[0].name == '3d':
                for i in range(self.ndim_image):
                    ax[i].set_xlabel(self.axes_labels[0])
                    ax[i].set_ylabel(self.axes_labels[1])
                    ax[i].set_zlabel(self.axes_labels[i + 2])
            else:
                for i in range(self.ndim_image):
                    ax[i].set_xlabel(self.axes_labels[0])
                    ax[i].set_ylabel(self.axes_labels[i + 1])

    def generic_plotting_checks(self, fig=None, ax=None, nrows=None,
                                 ncols=None):
        """Check the arguments passed to both :func:`plot
        <fda.functional_data.plot>` and :func:`scatter <fda.grid.scatter>`
         methods of the FDataGrid object.

        Args:
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            ax (list of axis objects, optional): axis over where the graphs are
                plotted. If None, see param fig.
            nrows(int, optional): designates the number of rows of the figure to
                plot the different dimensions of the image. Only specified if
                fig and ax are None. ncols must be also be customized in the
                same call.
            ncols(int, optional): designates the number of columns of the figure
                to plot the different dimensions of the image. Only specified if
                fig and ax are None. nrows must be also be customized in the
                same call.

        Returns:
            fig (figure object): figure object in which the graphs are plotted
                in case ax is None.
            ax (axes object): axes in which the graphs are plotted.

        """
        if self.ndim_domain > 2:
            raise NotImplementedError("Plot only supported for functional data"
                                      "modeled in at most 3 dimensions.")

        if fig is not None and ax is not None:
            raise ValueError("fig and axes parameters cannot be passed as "
                             "arguments at the same time.")

        if fig is not None and len(fig.get_axes()) != self.ndim_image:
            raise ValueError("Number of axes of the figure must be equal to"
                             "the dimension of the image.")

        if ax is not None and len(ax)!= self.ndim_image:
            raise ValueError("Number of axes must be equal to the dimension of "
                             "the image.")

        if ((ax is not None or fig is not None) and
            (nrows is not None or ncols is not None)):
            raise ValueError("The number of columns and/or number of rows of "
                             "the figure, in which each dimension of the image "
                             "is plotted, can only be customized in case fig is"
                             " None and ax is None.")

        if ((nrows is not None and ncols is not None) and
            nrows*ncols < self.ndim_image):
            raise ValueError("The number of columns and the number of rows "
                             "specified is incorrect.")

        if fig is None and ax is None:
            fig, ax = self.set_figure_and_axes(nrows, ncols)

        elif fig is not None:
            ax = fig.get_axes()

        else:
            fig = ax[0].get_figure()

        return fig, ax

    def plot(self, chart=None, *, derivative=0, fig=None, ax=None, nrows=None,
             ncols=None, npoints=None, domain_range=None, sample_labels=None,
             label_colors=None, label_names=None, **kwargs):
        """Plot the FDatGrid object.

        Args:
            chart (figure object, axe or list of axes, optional): figure over
                with the graphs are plotted or axis over where the graphs are
                    plotted. If None and ax is also None, the figure is
                    initialized.
            derivative (int or tuple, optional): Order of derivative to be
                plotted. In case of surfaces a tuple with the order of
                derivation in each direction can be passed. See :func:`evaluate`
                to obtain more information. Defaults 0.
            fig (figure object, optional): figure over with the graphs are
                plotted in case ax is not specified. If None and ax is also
                None, the figure is initialized.
            ax (list of axis objects, optional): axis over where the graphs are
                plotted. If None, see param fig.
            nrows(int, optional): designates the number of rows of the figure to
                plot the different dimensions of the image. Only specified if
                fig and ax are None.
            ncols(int, optional): designates the number of columns of the figure
                to plot the different dimensions of the image. Only specified if
                fig and ax are None.
            npoints (int or tuple, optional): Number of points to evaluate in
                the plot. In case of surfaces a tuple of length 2 can be pased
                with the number of points to plot in each axis, otherwise the
                same number of points will be used in the two axes. By default
                in unidimensional plots will be used 501 points; in surfaces
                will be used 30 points per axis, wich makes a grid with 900
                points.
            domain_range (tuple or list of tuples, optional): Range where the
                function will be plotted. In objects with unidimensional domain
                the domain range should be a tuple with the bounds of the
                interval; in the case of surfaces a list with 2 tuples with
                the ranges for each dimension. Default uses the domain range
                of the functional object.
            sample_labels (list of int): contains integers from [0 to number of
                labels) indicating to which group each sample belongs to. Then,
                the samples with the same label are plotted in the same color.
                If None, the default value, each sample is plotted in the color
                assigned by matplotlib.pyplot.rcParams['axes.prop_cycle'].
            label_colors (list of colors): colors in which groups are represented,
                there must be one for each group. If None, each group is shown
                with distict colors in the "Greys" colormap.
            label_names (list of str): name of each of the groups which appear
                in a legend, there must be one for each one. Defaults to None
                and the legend is not shown.
            **kwargs: if ndim_domain is 1, keyword arguments to be passed to the
                matplotlib.pyplot.plot function; if ndim_domain is 2, keyword
                arguments to be passed to the matplotlib.pyplot.plot_surface
                function.

        Returns:
            fig (figure object): figure object in which the graphs are plotted.
            ax (axes object): axes in which the graphs are plotted.

        """

        # Parse chart argument
        if chart is not None:
            if fig is not None or ax is not None:
                raise ValueError("fig, axes and chart parameters cannot "
                                 "be passed as arguments at the same time.")
            if isinstance(chart, plt.Figure):
                fig = chart
            elif isinstance(chart, Axes):
                ax = [chart]
            else:
                ax = chart

        if domain_range is None:
            domain_range = self.domain_range
        else:
            domain_range = _list_of_arrays(domain_range)

        fig, ax = self.generic_plotting_checks(fig, ax, nrows, ncols)

        patches = None
        next_color = False

        if sample_labels is not None:

            nlabels = np.max(sample_labels) + 1

            if np.any((sample_labels < 0) | (sample_labels >= nlabels)) or \
                    not np.all(np.isin(range(nlabels), sample_labels)):
                raise ValueError("sample_labels must contain at least an "
                                 "occurence of numbers between 0 and number "
                                 "of distint sample labels.")

            if label_colors is not None:
                if len(label_colors) != nlabels:
                    raise ValueError("There must be a color in label_colors "
                                     "for each of the labels that appear in "
                                     "sample_labels.")
                sample_colors = np.asarray(label_colors)[sample_labels]

            else:
                colormap = plt.cm.get_cmap('Greys')
                sample_colors = colormap(sample_labels / (nlabels - 1))


            if label_names is not None:
                if len(label_names) != nlabels:
                    raise ValueError("There must be a name in  label_names "
                                     "for each of the labels that appear in "
                                     "sample_labels.")

                if label_colors is None:
                    label_colors = colormap(
                        np.arange(nlabels) / (nlabels - 1))

                patches = []
                for i in range(nlabels):
                    patches.append(
                        mpatches.Patch(color=label_colors[i],
                                       label=label_names[i]))

        else:

            if 'color' in kwargs:
                sample_colors = self.nsamples * [kwargs.get("color")]
                kwargs.pop('color')

            elif 'c' in kwargs:
                sample_colors = self.nsamples * [kwargs.get("color")]
                kwargs.pop('c')

            else:
                sample_colors = np.empty((self.nsamples,)).astype(str)
                next_color = True

        if self.ndim_domain == 1:

            if npoints is None:
                npoints = 501

            # Evaluates the object in a linspace
            eval_points = np.linspace(*domain_range[0], npoints)
            mat = self(eval_points, derivative=derivative, keepdims=True)

            for i in range(self.ndim_image):
                for j in range(self.nsamples):
                    if sample_labels is None and next_color:
                        sample_colors[j] = ax[i]._get_lines.get_next_color()
                    ax[i].plot(eval_points, mat[j,..., i].T,
                               c=sample_colors[j], **kwargs)

        else:

            # Selects the number of points
            if npoints is None:
                npoints = (30, 30)
            elif np.isscalar(npoints):
                npoints = (npoints, npoints)
            elif len(npoints) != 2:
                raise ValueError("npoints should be a number or a tuple of "
                                 "length 2.")

            # Axes where will be evaluated
            x = np.linspace(*domain_range[0], npoints[0])
            y = np.linspace(*domain_range[1], npoints[1])

            # Evaluation of the functional object
            Z =  self((x,y), derivative=derivative, grid=True, keepdims=True)

            X, Y = np.meshgrid(x, y, indexing='ij')

            for i in range(self.ndim_image):
                for j in range(self.nsamples):
                    if sample_labels is None and next_color:
                        sample_colors[j] = ax[i]._get_lines.get_next_color()
                    ax[i].plot_surface(X, Y, Z[j,...,i],
                                       color=sample_colors[j], **kwargs)

        self.set_labels(fig, ax, patches)

        return fig, ax

    @abstractmethod
    def copy(self, **kwargs):
        """Make a copy of the object.

        Args:
            **kwargs: named args with attributes to be changed in the new copy.

        Returns:
            FData: A copy of the FData object with the arguments specified
            in **kwargs changed.

        """
        pass

    @abstractmethod
    def mean(self):
        """Compute the mean of all the samples.

        Returns:
            FData : A FData object with just one sample representing
            the mean of all the samples in the original object.

        """
        pass

    @abstractmethod
    def to_grid(self, eval_points=None):
        """Return the discrete representation of the object.

        Args:
            eval_points (array_like, optional): Set of points where the
                functions are evaluated.

        Returns:
              FDataGrid: Discrete representation of the functional data
              object.
        """

        pass

    @abstractmethod
    def to_basis(self, basis, eval_points=None, **kwargs):
        """Return the basis representation of the object.

        Args:
            basis(Basis): basis object in which the functional data are
                going to be represented.
            **kwargs: keyword arguments to be passed to
                FDataBasis.from_data().

        Returns:
            FDataBasis: Basis representation of the funtional data
            object.
        """

        pass

    @abstractmethod
    def concatenate(self, other):
        """Join samples from a similar FData object.

        Joins samples from another FData object if it has the same
        dimensions and has compatible representations.

        Args:
            other (:class:`FData`): another FData object.

        Returns:
            :class:`FData`: FData object with the samples from the two
            original objects.
        """

        pass

    @abstractmethod
    def compose(self, fd, *, eval_points=None, **kwargs):
        """Composition of functions.

        Performs the composition of functions.

        Args:
            fd (:class:`FData`): FData object to make the composition. Should
                have the same number of samples and image dimension equal to
                the domain dimension of the object composed.
            eval_points (array_like): Points to perform the evaluation.
            **kwargs: Named arguments to be passed to the composition method of
                the specific functional object.

        """
        pass

    @abstractmethod
    def __getitem__(self, key):
        """Return self[key]."""

        pass

    @abstractmethod
    def __add__(self, other):
        """Addition for FData object."""

        pass

    @abstractmethod
    def __radd__(self, other):
        """Addition for FData object."""

        pass

    @abstractmethod
    def __sub__(self, other):
        """Subtraction for FData object."""

        pass

    @abstractmethod
    def __rsub__(self, other):
        """Right subtraction for FData object."""

        pass

    @abstractmethod
    def __mul__(self, other):
        """Multiplication for FData object."""

        pass

    @abstractmethod
    def __rmul__(self, other):
        """Multiplication for FData object."""

        pass

    @abstractmethod
    def __truediv__(self, other):
        """Division for FData object."""

        pass

    @abstractmethod
    def __rtruediv__(self, other):
        """Right division for FData object."""

        pass


    def __iter__(self):
        """Iterate over the samples"""

        for i in range(self.nsamples):
            yield self[i]

    def __len__(self):
        """Returns the number of samples of the FData object."""

        return self.nsamples

Magnitude-Shape Plot
====================

The Magnitude-Shape Plot is implemented in the 
:class:`~skfda.exploratory.visualization.MagnitudeShapePlot` class.

The :class:`~skfda.exploratory.visualization.MagnitudeShapePlot` needs both the mean 
and the variation of the directional outlyingness of the samples, which is calculated using
:func:`~skfda.exploratory.outliers.directional_outlyingness_stats`.

Once the points assigned to each of the samples are obtained from the above
function, an outlier detection method is implemented. The results can be shown
calling the :meth:`~skfda.magnitude_shape_plot.MagnitudeShapePlot.plot`
method of the class.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.visualization.MagnitudeShapePlot

Magnitude-Shape Plot
====================

The Magnitude-Shape Plot is implemented in the :class:`MagnitudeShapePlot` class.

The :class:`MagnitudeShapePlot` needs both the mean and the variation of the
directional outlyingness of the samples, which is calculated in the below function.

.. autosummary::
   :toctree: autosummary

   fda.magnitude_shape_plot.directional_outlyingness

Once the points assigned to each of the samples are obtained from the above
function, an outlier detection method is implemented. The results can be shown
calling the :func:`plot method <fda.magnitude_shape_plot.MagnitudeShapePlot.plot>`
of the class.

.. autosummary::
   :toctree: autosummary

   fda.magnitude_shape_plot.MagnitudeShapePlot

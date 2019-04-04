Magnitude-Shape Plot
====================

Function to construct the Magnitude-Shape Plot.

First the directional outlyingness of the samples is needed, which is calculated with the below function
although it remains for internal use since the next function already returns the values.

.. autosummary::
   :toctree: autosummary

   fda.magnitude_shape_plot.directional_outlyingness


Then, an outlier detection method is implemented and the plot is shown.

.. autosummary::
   :toctree: autosummary

   fda.magnitude_shape_plot.magnitude_shape_plot
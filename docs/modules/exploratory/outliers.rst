Outlier detection
=================

Functional outlier detection is the identification of functions that do not seem to behave like the others in the
dataset. There are several ways in which a function may be different from the others. For example, a function may
have a different shape than the others, or its values could be more extreme. Thus, outlyingness is difficult to
categorize exactly as each outlier detection method looks at different features of the functions in order to
identify the outliers.

Each of the outlier detection methods in scikit-fda has the same API as the outlier detection methods of
`scikit-learn <https://scikit-learn.org/stable/modules/outlier_detection.html>`_.

Boxplot Outlier Detector
------------------------

One of the most common ways of outlier detection is given by the functional data boxplot. An observation is marked
as an outlier if it has points :math:`1.5 \cdot IQR` times outside the region containing the deepest 50% of the curves
(the central region), where :math:`IQR` is the interquartile range.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.outliers.BoxplotOutlierDetector


DirectionalOutlierDetector
--------------------------

Other more novel way of outlier detection takes into account the magnitude and shape of the curves. Curves which have
a very different shape or magnitude are considered outliers.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.outliers.DirectionalOutlierDetector

For this method, it is necessary to compute the mean and variation of the directional outlyingness, which can be done
with the following function.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.outliers.directional_outlyingness_stats

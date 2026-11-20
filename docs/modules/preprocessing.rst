Preprocessing
=============

Sometimes we need to preprocess the data prior to analyze it. The modules in
this category deal with this problem.

.. toctree::
   :titlesonly:
   :maxdepth: 4
   :caption: Modules:
   :hidden:

   preprocessing/missing
   preprocessing/smoothing
   preprocessing/registration
   preprocessing/dim_reduction
   preprocessing/feature_construction
   preprocessing/scaling

Missing data
------------

When the observations contains missing data, it is necessary to reconstruct
the invalid information before processing it further.
:doc:`Here <preprocessing/missing>` you can learn more about this procedure.

Smoothing
---------

If the functional data observations are noisy, *smoothing* the data allows a
better representation of the true underlying functions. You can learn more
about the smoothing methods provided by scikit-fda
:doc:`here <preprocessing/smoothing>`.

Registration
------------

Sometimes, the functional data may be misaligned, or the phase variation
should be ignored in the analysis. To align the data and eliminate the phase
variation, we need to use *registration* methods. 
:doc:`Here <preprocessing/registration>` you can learn more about the
registration methods available in the library.

Dimensionality Reduction
------------------------

The functional data may have too many features so we cannot analyse
the data with clarity. To better understand the data, we need to use
*dimensionality reduction* methods that can reduce the number of features
while still preserving the most relevant information.
:doc:`Here <preprocessing/dim_reduction>` you can learn more about the
dimension reduction methods available in the library.

Feature construction
--------------------

When dealing with functional data we might want to construct new features
that can be used as additional inputs to the machine learning algorithms.
The expectation is that these features make explicit characteristics that
facilitate the learning process. To construct new features from the curves,
*feature construction* methods are available.
:doc:`Here <preprocessing/feature_construction>` you can learn more about the
feature construction methods available in the library.

Scaling
-------

In the context of vector-valued functional data and mixed datasets, *scaling*
and *centering* become essential preprocessing steps before applying further
analysis. In classical functional data analysis, these operations are not
typically necessary, as functions are often observed over the same domain and
measured in the same units. In such cases, distances between functions are
already meaningful.

However, when working with data that includes multiple functional components
measured in different units or on different scales, or when combining scalar
and functional features, it is important to normalize the data to ensure fair
comparisons. Scaling helps prevent components with larger magnitude from
dominating the analysis.

:doc:`Here <preprocessing/scaling>` you can learn more about the scaling
methods available in the library.
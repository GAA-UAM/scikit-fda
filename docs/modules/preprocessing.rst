Preprocessing
=============

Sometimes we need to preprocess the data prior to analyze it. The modules in
this category deal with this problem.

.. toctree::
   :titlesonly:
   :maxdepth: 4
   :caption: Modules:
   :hidden:

   preprocessing/smoothing
   preprocessing/registration

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
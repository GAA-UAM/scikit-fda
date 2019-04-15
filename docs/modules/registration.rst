Registration
============


We see often that variation in functional observations involves phase and
amplitude variation, which may hinder further analysis. That problem is treated
during the registration process. This module contains procedures for the
registration of the data.

Shift Registration
-------------------

Many of the issues involved in registration can be solved by considering
the simplest case, a simple shift in the time scale. This often happens because
the time at which the recording process begins is arbitrary, and is unrelated
to the beginning of the interesting segment of the data. In the
`Shift Registration Example <../auto_examples/plot_shift_registration_basis.html>`_
it is shown the basic usage of this methods applied to periodic data.

.. autosummary::
   :toctree: autosummary

   fda.registration.shift_registration
   fda.registration.shift_registration_deltas


Landmark Registration
----------------------

Landmark registration aligns features applying a transformation of the time that
takes all the times of a given feature into a common value.

The simplest case in which each sample presents a unique landmark can be solved
by performing a translation in the time scale. See the
`Landmark Shift Example <../auto_examples/plot_landmark_shift.html>`_.

.. autosummary::
   :toctree: autosummary

   fda.registration.landmark_shift
   fda.registration.landmark_shift_deltas


The general case of landmark registration may present multiple landmarks for
each sample and a non-linear transformation in the time scale should be applied.
See the `Landmark Registration Example
<../auto_examples/plot_landmark_registration.html>`_

.. autosummary::
   :toctree: autosummary

   fda.registration.landmark_registration
   fda.registration.landmark_registration_warping


Elastic Registration
--------------------

The elastic registration is a novel approach to this problem that uses the
properties of the Fisher-Rao metric to perform the alignment of the curves.
In the examples of `pairwise alignment
<../auto_examples/plot_pairwise_alignment.html>`_ and `elastic registration
<../auto_examples/plot_elastic_registration.html>`_ is shown a brief
introduction to this topic along the usage of the corresponding functions.

.. autosummary::
   :toctree: autosummary

   fda.registration.elastic_registration
   fda.registration.elastic_registration_warping


The module contains some routines related with the elastic registration, making
a transformation of the sampling, computing different means or distances based
on the elastic framework.

.. autosummary::
  :toctree: autosummary

   fda.registration.elastic_mean
   fda.registration.warping_mean
   fda.registration.to_srsf
   fda.registration.from_srsf



Amplitude and Phase Decomposition
---------------------------------

The amplitude and phase variation may be quantified by comparing a sample before
and after registration. The package contains an implementation of the
decomposition procedure developed by *Kneip and Ramsay (2008)*.

.. autosummary::
   :toctree: autosummary

   fda.registration.mse_decomposition


Utility functions
-----------------

There are some other method related with the registration problem in this
module.

.. autosummary::
   :toctree: autosummary

   fda.registration.invert_warping
   fda.registration.normalize_warping

References
----------

* Ramsay, J., Silverman, B. W. (2005). Functional Data Analysis. Springer.

* Kneip, Alois & Ramsay, James. (2008).  Quantifying amplitude and phase
  variation. Journal of the American Statistical Association.

* Ramsay, J., Hooker, G. & Graves S. (2009). Functional Data Analysis with
  R and Matlab. Springer.

* Srivastava, Anuj & Klassen, Eric P. (2016). Functional and shape data
  analysis. Springer.

* J. S. Marron, James O. Ramsay, Laura M. Sangalli and Anuj Srivastava (2015).
  Functional Data Analysis of Amplitude and Phase Variation. Statistical Science
  2015, Vol. 30, No. 4

"""
Landmark shift
==============

This example shows how to shift functional data objects to align
its samples with a particular reference point.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 2


import matplotlib.pyplot as plt
import numpy as np

import skfda

##############################################################################
# We will use an example dataset synthetically generated by
# :func:`~skfda.datasets.make_multimodal_samples`, which in this case will be
# used to generate gaussian-like samples with a mode near to 0.
# Each sample will be shifted to align their modes to a reference point using
# the function
# :func:`~skfda.preprocessing.registration.landmark_shift_registration`.

fd = skfda.datasets.make_multimodal_samples(random_state=1)
fd.extrapolation = 'bounds'  #  See extrapolation for a detailed explanation.

fd.plot()

##############################################################################
# A landmark or a feature of a curve is some characteristic that one can
# associate with a specific argument value t. These are typically maxima,
# minima, or zero crossings of curves, and may be identified at the level of
# some derivatives as well as at the level of the curves themselves
# [RaSi2005-2]_.
#
# For alignment we need to know in advance the location of the landmark of
# each of the samples, in our case it will correspond to the maxima of each
# sample. Because our dataset has been generated synthetically we can obtain
# the value of the landmarks using the function
# :func:`~skfda.datasets.make_multimodal_landmarks`, which is used by
# :func:`~skfda.datasets.make_multimodal_samples` to set the location of the
# modes.
#
# In general it will be necessary to use numerical or other methods to
# determine the location of the landmarks.

landmarks = skfda.datasets.make_multimodal_landmarks(random_state=1).squeeze()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(landmarks, np.repeat(1, fd.n_samples))
fd.plot(fig=fig)

##############################################################################
# Location of the landmarks:

print(landmarks)

##############################################################################
# The following figure shows the result of shifting the curves to align their
# landmarks at 0.

fd_registered = skfda.preprocessing.registration.landmark_shift_registration(
    fd,
    landmarks,
    location=0,
)

fig = fd_registered.plot()
fig.axes[0].scatter(0, 1)

##############################################################################
# In many circumstances it is possible that we could not apply extrapolation,
# in these cases it is possible to restrict the domain to avoid evaluating
# points outside where our curves are defined.
#
# If the location of the new reference point is not specified it is choosen
# the point that minimizes the maximum amount of shift.

# Curves aligned restricting the domain
fd_restricted = skfda.preprocessing.registration.landmark_shift_registration(
    fd,
    landmarks,
    restrict_domain=True,
)

# Curves aligned to default point without restrict domain
fd_extrapolated = skfda.preprocessing.registration.landmark_shift_registration(
    fd,
    landmarks,
)


fig = fd_extrapolated.plot(linestyle='dashed', label='Extrapolated samples')
fd_restricted.plot(fig=fig, label="Restricted samples")

##############################################################################
# The previous method is also applicable for multidimensional objects,
# without limitation of the domain or image dimension. As an example we are
# going to create a datset with surfaces, in a similar way to the previous
# case.

fd = skfda.datasets.make_multimodal_samples(
    n_samples=3,
    points_per_dim=30,
    dim_domain=2,
    random_state=1,
)

fd.plot()

##############################################################################
# In this case the landmarks will be defined by tuples with 2 coordinates.

landmarks = skfda.datasets.make_multimodal_landmarks(
    n_samples=3,
    dim_domain=2,
    random_state=1,
).squeeze()
print(landmarks)

##############################################################################
# As in the previous case, we can align the curves to a specific point,
# or by default will be chosen the point that minimizes the maximum amount
# of displacement.

fd_registered = skfda.preprocessing.registration.landmark_shift_registration(
    fd,
    landmarks,
)

fd_registered.plot()

plt.show()

###############################################################################
# .. [RaSi2005-2] Ramsay, J., Silverman, B. W. (2005).
#               Functional Data Analysis. Springer.

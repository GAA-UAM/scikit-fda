"""
Outlier detection with Local Outlier Factor
===========================================

Shows the use of the Local Outlier Factor to detect outliers in the octane
dataset.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from skfda.datasets import fetch_octane
from skfda.exploratory.outliers import LocalOutlierFactor


##############################################################################
# First, we load the *octane dataset* consisting of 39 near infrared
# (NIR) spectra of gasoline samples, with wavelengths ranging from 1102nm to
# 1552nm with measurements every two nm.
#
# This dataset contains six outliers, studied in [RDEH2006]_ and [HuRS2015]_,
# to which ethanol was added. This different
# composition has an effect on the shape of the spectra of gasoline samples.
#

fd, labels = fetch_octane(return_X_y=True)
fd.plot()


##############################################################################
# :class:`~skfda.exploratory.outliers.LocalOutlierFactor`
# (`LOF <https://en.wikipedia.org/wiki/Local_outlier_factor>`_), based on
# the local density of the curves as described in [BKNS2000]_, may be used to
# detect these outliers. In order to get the results the
# :meth:`~skfda.exploratory.outliers.LocalOutlierFactor.fit_predict`
# method is used.
#

lof = LocalOutlierFactor()
is_outlier = lof.fit_predict(fd)

print(is_outlier) # 1 for inliners / -1 for outliers

##############################################################################
# The curves detected as outliers correspond to the samples to which
# ethanol was added.
#

# TODO: Use one hot encoding internally to allow arbitrary sample_labels
is_outlier[is_outlier == -1] = 0

fd.plot(sample_labels=is_outlier, label_colors=['darkorange', 'lightgrey'],
        label_names=["outlier", "inliner"])


##############################################################################
# .. rubric:: References
# ..  [RDEH2006] Rousseeuw, Peter & Debruyne, Michiel & Engelen, Sanne &
#     Hubert, Mia. (2006). Robustness and Outlier Detection in
#     Chemometrics. Critical Reviews in Analytical Chemistry. 36.
#     221-242. 10.1080/10408340600969403.
# ..  [HuRS2015] Hubert, Mia & Rousseeuw, Peter & Segaert, Pieter. (2015).
#     Multivariate functional outlier detection. Statistical Methods and
#     Applications. 24. 177-202. 10.1007/s10260-015-0297-8.
# .. [BKNS2000] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander,
#    J. (2000, May). LOF: identifying density-based local outliers. In ACM
#    sigmod record.

plt.show()

"""
Exploring data
==============

Explores the Tecator data set by plotting the functional data and calculating
means and derivatives.
"""

# Author: Miguel Carbajo Berrocal
# License: MIT

import skfda
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# In this example we are going to explore the functional properties of the
# :func:`Tecator <skfda.datasets.fetch_tecator>` dataset. This dataset measures
# the infrared absorbance spectrum of meat samples. The objective is to predict
# the fat, water, and protein content of the samples.
#
# In this example we only want to discriminate between meat with less than 20%
# of fat, and meat with a higher fat content.
dataset = skfda.datasets.fetch_tecator()
fd = dataset['data']
y = dataset['target']
target_feature_names = dataset['target_feature_names']
fat = y[:, np.asarray(target_feature_names) == 'Fat'].ravel()

###############################################################################
# We will now plot in red samples containing less than 20% of fat and in blue
# the rest.

low_fat = fat < 20

fd[low_fat].plot(c='r', linewidth=0.5)
fd[~low_fat].plot(c='b', linewidth=0.5, alpha=0.7)

###############################################################################
# The means of each group are the following ones.

skfda.mean(fd[low_fat]).plot(c='r', linewidth=0.5)
skfda.mean(fd[~low_fat]).plot(c='b', linewidth=0.5, alpha=0.7)
fd.dataset_label = fd.dataset_label + ' - means'

###############################################################################
# In this dataset, the vertical shift in the original trajectories is not very
# significative for predicting the fat content. However, the shape of the curve
# is very relevant. We can observe that looking at the first and second
# derivatives.

fdd = fd.derivative(1)
fdd[low_fat].plot(c='r', linewidth=0.5)
fdd[~low_fat].plot(c='b', linewidth=0.5, alpha=0.7)

plt.figure()
fdd = fd.derivative(2)
fdd[low_fat].plot(c='r', linewidth=0.5)
fdd[~low_fat].plot(c='b', linewidth=0.5, alpha=0.7)

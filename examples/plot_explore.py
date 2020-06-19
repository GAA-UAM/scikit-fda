"""
Exploring data
==============

Explores the Tecator data set by plotting the functional data and calculating
means and derivatives.
"""

# Author: Miguel Carbajo Berrocal
# License: MIT

import skfda

import numpy as np


##############################################################################
# In this example we are going to explore the functional properties of the
# :func:`Tecator <skfda.datasets.fetch_tecator>` dataset. This dataset
# measures the infrared absorbance spectrum of meat samples. The objective is
# to predict the fat, water, and protein content of the samples.
#
# In this example we only want to discriminate between meat with less than 20%
# of fat, and meat with a higher fat content.
dataset = skfda.datasets.fetch_tecator()
fd = dataset['data']
y = dataset['target']
target_feature_names = dataset['target_feature_names']
fat = y[:, np.asarray(target_feature_names) == 'Fat'].ravel()

##############################################################################
# We will now plot in red samples containing less than 20% of fat and in blue
# the rest.

low_fat = fat < 20
labels = np.full(fd.n_samples, 'high fat')
labels[low_fat] = 'low fat'
colors = {'high fat': 'red',
          'low fat': 'blue'}

fig = fd.plot(group=labels, group_colors=colors,
              linewidth=0.5, alpha=0.7, legend=True)

##############################################################################
# The means of each group are the following ones.

mean_low = skfda.exploratory.stats.mean(fd[low_fat])
mean_high = skfda.exploratory.stats.mean(fd[~low_fat])

means = mean_high.concatenate(mean_low)

means.dataset_label = fd.dataset_label + ' - means'
means.plot(group=['high fat', 'low fat'], group_colors=colors,
           linewidth=0.5, legend=True)

##############################################################################
# In this dataset, the vertical shift in the original trajectories is not
# very significative for predicting the fat content. However, the shape of the
# curve is very relevant. We can observe that looking at the first and second
# derivatives.
#
# The first derivative is shown below:

fdd = fd.derivative()
fig = fdd.plot(group=labels, group_colors=colors,
               linewidth=0.5, alpha=0.7, legend=True)

##############################################################################
# We now show the second derivative:
fdd = fd.derivative(order=2)
fig = fdd.plot(group=labels, group_colors=colors,
               linewidth=0.5, alpha=0.7, legend=True)

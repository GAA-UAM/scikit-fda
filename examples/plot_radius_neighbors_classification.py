"""
Radius nearest neighbors classification
=======================================

Shows the usage of the k-nearest neighbors classifier.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 1


import skfda
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skfda.ml.classification import RadiusNeighborsClassifier
from skfda.misc.metrics import pairwise_distance, lp_distance



################################################################################
#
#
#
# Text


fd1 = skfda.datasets.make_sinusoidal_process(error_std=.0, phase_std=.35, random_state=0)
fd2 = skfda.datasets.make_sinusoidal_process(phase_mean=1.9, error_std=.0, random_state=1)

fd1.plot(color='C0')
fd2.plot(color='C1')


################################################################################
#
#
#
# Text


X = fd1.concatenate(fd2)
y = 15*[0] + 15*[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


################################################################################
#
#
#
# Text


plt.figure()

sample = X_test[0]


X_train.plot(color='C0')
sample.plot(color='red', linewidth=3)

lower = sample - 0.3
upper = sample + 0.3

plt.fill_between(sample.sample_points[0], lower.data_matrix.flatten(),
                 upper.data_matrix[0].flatten(),  alpha=.25, color='C1')


################################################################################
#
#
#
# Text



# Creation of pairwise distance
l_inf = pairwise_distance(lp_distance, p=np.inf)
distances = l_inf(sample, X_train)[0] # L_inf distances to 'sample'


plt.figure()
X_train[distances > .3].plot(color='C0')
X_train[distances <= .3].plot(color='C1')
sample.plot(color='red', linewidth=3)

plt.fill_between(sample.sample_points[0], lower.data_matrix.flatten(),
                 upper.data_matrix[0].flatten(),  alpha=.25, color='C1')


################################################################################
#
#
#
# Text

radius_nn = RadiusNeighborsClassifier(radius=.3,  weights='distance')
radius_nn.fit(X_train, y_train)


################################################################################
#
#
#
# Text

pred = radius_nn.predict(X_test)
print(pred)

################################################################################
#
#
#
# Text

test_score = radius_nn.score(X_test, y_test)
print(test_score)

################################################################################
#
#
#
# Text

radius_nn = RadiusNeighborsClassifier(radius=3, metric='euclidean',
                                      weights='distance', sklearn_metric=True)


radius_nn.fit(X_train, y_train)

test_score = radius_nn.score(X_test, y_test)
print(test_score)


################################################################################
#
#
#
# Text

radius_nn.set_params(radius=.5)
radius_nn.fit(X_train, y_train)

try:
    radius_nn.predict(X_test)
except ValueError as e:
    print(e)

################################################################################
#
#
#
# Text

radius_nn.set_params(outlier_label=2)
radius_nn.fit(X_train, y_train)
pred = radius_nn.predict(X_test)

print(pred)

################################################################################
#
#
#
# Text


plt.show()

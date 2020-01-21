from skfda import FDataGrid
import numpy as np
from skfda.inference.anova.anova_oneway import func_oneway, func_oneway_usc
from skfda.datasets import make_gaussian_process
from matplotlib import pyplot as plt

m = 25
n = 1

process = make_gaussian_process(n, n_features=m)
print(process.mean().data_matrix)

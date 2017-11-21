"""This module defines the most commonly used kernels.

"""
import math
from scipy import stats
import numpy


__author__ = "Miguel Carbajo Berrocal"
__email__ = "miguel.carbajo@estudiante.uam.es"


def normal(u):
    return stats.norm.pdf(u)


def cosine(u):
    if isinstance(u, numpy.ndarray):
        res = numpy.zeros(u.shape)
        res[abs(u) <= 1] = math.pi/4 * (math.cos(math.pi*u[abs(u) <= 1]/2))
        return res
    if abs(u) <= 1:
        return math.pi/4 * (math.cos(math.pi*u/2))
    else:
        return 0


def epanechnikov(u):
    if isinstance(u, numpy.ndarray):
        res = numpy.zeros(u.shape)
        res[abs(u) <= 1] = 0.75*(1-u[abs(u) <= 1]**2)
        return res
    if abs(u) <= 1:
        return 0.75*(1-u**2)
    else:
        return 0


def tri_weight(u):
    if isinstance(u, numpy.ndarray):
        res = numpy.zeros(u.shape)
        res[abs(u) <= 1] = 35/32*(1-u[abs(u) <= 1]**2)**3
        return res
    if abs(u) <= 1:
        return 35/32*(1-u**2)**3
    else:
        return 0


def quartic(u):
    if isinstance(u, numpy.ndarray):
        res = numpy.zeros(u.shape)
        res[abs(u) <= 1] = 15/16*(1-u[abs(u) <= 1]**2)**2
        return res
    if abs(u) <= 1:
        return 15/16*(1-u**2)**2
    else:
        return 0


def uniform(u):
    if isinstance(u, numpy.ndarray):
        res = numpy.zeros(u.shape)
        res[abs(u) <= 1] = 0.5
        return res
    if abs(u) <= 1:
        return 0.5
    else:
        return 0

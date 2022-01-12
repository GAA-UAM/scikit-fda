from setuptools import setup
from setuptools import Extension
import numpy as np

setup(
    ext_modules=[
        Extension('sdtw_fast',
                sources=['sdtw_fast.pyx'],
                language='c',
                include_dirs=[np.get_include()]
                )],
    zip_safe=False
)
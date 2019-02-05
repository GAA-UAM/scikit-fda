import os
import sys

import numpy

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open(os.path.join(os.path.dirname(__file__),
                       'VERSION'), 'r') as version_file:
    version = version_file.read().strip()

deps_path = 'deps'
fdasrsf_path = os.path.join(deps_path, 'fdasrsf')


extensions = [
    Extension(name='optimum_reparamN',
              sources=[
                  os.path.join(fdasrsf_path, 'optimum_reparamN.pyx'),
                  os.path.join(fdasrsf_path, 'DynamicProgrammingQ2.c'),
                  os.path.join(fdasrsf_path, 'dp_grid.c')
              ],
              include_dirs=[numpy.get_include()],
              language='c',
              ),
    Extension(name='optimum_reparam_fN',
              sources=[
                  os.path.join(fdasrsf_path, 'optimum_reparam_fN.pyx'),
                  os.path.join(fdasrsf_path, 'DP.c')
              ],
              include_dirs=[numpy.get_include()],
              language='c',
              ),
]

setup(name='fda',
      version=version,
      description='Functional Data Analysis Python package',
      long_description="",  # TODO
      url='https://fda.readthedocs.io',
      author='Miguel Carbajo Berrocal',
      author_email='miguel.carbajo@estudiante.uam.com',
      ext_modules=cythonize(extensions),
      include_package_data=True,
      platforms=['any'],
      license='GPL3',
      packages=find_packages(),
      python_requires='>=3.5, <4',
      classifiers=[
        'Development Status :: 1',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=['numpy', 'scikit-learn',
                        'scikit-datasets[cran]>=0.1.24', 'rdata'],
      setup_requires=pytest_runner,
      tests_require=['pytest', 'numpy>=1.14'],
      test_suite='tests',
      zip_safe=False)

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
    Extension(name='optimum_reparam',
              sources=[
                  os.path.join(fdasrsf_path, 'optimum_reparam.pyx'),
                  os.path.join(fdasrsf_path, 'dp_grid.c')
              ],
              include_dirs=[numpy.get_include()],
              language='c',
              ),
]

setup(name='scikit-fda',
      version=version,
      description='Functional Data Analysis Python package',
      long_description="",  # TODO
      url='https://fda.readthedocs.io',
      maintainer='Carlos Ramos Carreño',
      maintainer_email='vnmabus@gmail.com',
      ext_modules=cythonize(extensions),
      include_package_data=True,
      platforms=['any'],
      license='BSD',
      packages=find_packages(),
      python_requires='>=3.6, <4',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=['numpy', 'scikit-learn', 'matplotlib',
                        'scikit-datasets[cran]>=0.1.24', 'rdata', 'mpldatacursor'],
      setup_requires=pytest_runner,
      tests_require=['pytest', 'numpy>=1.14'],
      test_suite='tests',
      zip_safe=False)

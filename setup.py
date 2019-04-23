import os
import sys

import numpy

from setuptools import setup, find_packages
from setuptools.extension import Extension

have_cython = False
try:
    from Cython.Distutils import build_ext as _build_ext
    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext


needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open(os.path.join(os.path.dirname(__file__),
                       'VERSION'), 'r') as version_file:
    version = version_file.read().strip()

# Cython extensions
deps_path = 'deps'
fdasrsf_path = os.path.join(deps_path, 'fdasrsf')

if have_cython:
    fdasrsf_sources = [
        os.path.join(fdasrsf_path, 'optimum_reparam.pyx'),
        os.path.join(fdasrsf_path, 'dp_grid.c')
    ]
else:
    fdasrsf_sources = [
        os.path.join(fdasrsf_path, 'optimum_reparam.c'),
        os.path.join(fdasrsf_path, 'dp_grid.c')
    ]

ext_modules = [
    Extension(name='optimum_reparam',
              sources=fdasrsf_sources,
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
      ext_modules=ext_modules,
      cmdclass={'build_ext': _build_ext},
      include_package_data=True,
      platforms=['any'],
      license='GPL3',
      packages=find_packages(),
      python_requires='>=3.6, <4',
      classifiers=[
        'Development Status :: 1',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=['numpy', 'scikit-learn', 'matplotlib',
                        'scikit-datasets[cran]>=0.1.24', 'rdata'],
      setup_requires=pytest_runner,
      tests_require=['pytest', 'numpy>=1.14'],
      test_suite='tests',
      zip_safe=False)

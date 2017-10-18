from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys

import fda


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(name='fda',
      version=fda.__version__,
      description="",  # TODO
      long_description="",  # TODO
      url='https://fda.readthedocs.io',
      author='Miguel Carbajo Berrocal',
      author_email='miguel.carbajo@estudiante.uam.com',
      include_package_data=True,
      platforms=['any'],
      license='GPL3',
      packages=find_packages(),
      python_requires='>=2.7, >=3.5, <4',
      classifiers=[
        'Development Status :: 1',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=['numpy',
                        'scipy',
                        'setuptools'],
      tests_require=['pytest'],
      test_suite='tests',
      zip_safe=False)

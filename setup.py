import os
import sys

from setuptools import setup, find_packages

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open(os.path.join(os.path.dirname(__file__),
                       'VERSION'), 'r') as version_file:
    version = version_file.read().strip()

setup(name='fda',
      version=version,
      description="",  # TODO
      long_description="",  # TODO
      url='https://fda.readthedocs.io',
      author='Miguel Carbajo Berrocal',
      author_email='miguel.carbajo@estudiante.uam.com',
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
                        'scikit-datasets[cran]>=0.1.16', 'rdata'],
      setup_requires=pytest_runner,
      tests_require=['pytest', 'numpy>=1.14'],
      test_suite='tests',
      zip_safe=False)

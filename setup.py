from setuptools import setup, find_packages

import fda

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
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=['numpy',
                        'scipy',
                        'setuptools'],
      tests_require=['pytest'],
      test_suite='tests',
      zip_safe=False)

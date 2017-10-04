from setuptools import setup

setup(name='fda',
      version='0.1',
      description='Functional Data Analysis Python package',
      url='https://github.com/mcarbajo/fda',
      author='Flying Circus',
      author_email='miguel.carbajo@estudiante.uam.es',
      license='???',
      packages=['fda'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      zip_safe=False)

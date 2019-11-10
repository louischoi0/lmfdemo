import io
from setuptools import find_packages, setup

# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.rst', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

      #

setup(name='lmf',
      version='0.1',
      description='',
      url='https://louischoi.github.io',
      author='louis',
      author_email='yerang040231@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3.7',
          ],
      install_requires = [
        "pandas>=0.25.2"
      ],
      zip_safe=False)
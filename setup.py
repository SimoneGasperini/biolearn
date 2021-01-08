from setuptools import setup
from setuptools import find_packages


setup(

  name = "biolearn",
  version = "0.1.0",

  author = ["SimoneGasperini", "Nico Curti"],
  author_email = ["simone.gasperini2@studio.unibo.it", "nico.curit2@unibo.it"],

  description = "Unsupervised neural networks with biological-inspired learning rules",
  url = "https://github.com/SimoneGasperini/biolearn.git",

  packages = find_packages(),

  classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent"
  ],

  python_requires = ">=3.8",

)

#!/usr/bin/env python

from setuptools import setup
#from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cython_en1ch.pyx"),
)


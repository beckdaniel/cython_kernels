#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '0.1'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'cython_kernels',
      version = version,
      packages = ['cython_kernels','cython_kernels.tk'],
      package_dir={'GPy': 'GPy'},
      long_description=read('README.md'),
      install_requires=['nltk'],
      )

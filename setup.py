#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

from setuptools import setup

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


setup(name="strfpy",
      version="0.0.1",
      description="pure python module for estimating spectrotemporal receptive fields",
      author="Tyler Robbins",
      author_email="tdr5wc at the domain 'virginia.edu'",

      packages=['strfpy']
      )

#!/usr/bin/env python

from setuptools import setup, find_packages


name = 'image_segmentation'
version = '0.0.1'
description = 'Python Image Segmentation Project'
author = 'Tetyana Goldanova'

setup(
      name=name,
      version=version,
      description=description,
      author=author,
      packages=find_packages(where='src'),
      package_dir={'':'src'},
)
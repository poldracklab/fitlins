# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

from . import _version

__version__ = _version.get_versions()['version']

__copyright__ = 'Copyright 2019, Center for Reproducible Neuroscience, Stanford University'
__credits__ = ['Christopher J. Markiewicz', 'Chris Gorgolewski', 'Russell A. Poldrack']
__packagename__ = 'fitlins'

DOWNLOAD_URL = 'https://github.com/poldracklab/{name}/archive/{ver}.tar.gz'.format(
    name=__packagename__, ver=__version__
)

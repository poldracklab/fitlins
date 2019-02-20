# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'The CRN developers'
__copyright__ = ('Copyright 2019, Center for Reproducible Neuroscience, '
                 'Stanford University')
__credits__ = ['Christopher J. Markiewicz', 'Chris Gorgolewski',
               'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Christopher J. Markiewicz'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/poldracklab/fitlins'
__packagename__ = 'fitlins'
__description__ = 'Fit Linear Models to BIDS Datasets'

DOWNLOAD_URL = (
    'https://github.com/poldracklab/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

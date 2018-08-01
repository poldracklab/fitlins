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
__copyright__ = ('Copyright 2018, Center for Reproducible Neuroscience, '
                 'Stanford University')
__credits__ = ['Christopher J. Markiewicz', 'Chris Gorgolewski',
               'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Christopher J. Markiewicz'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/poldracklab/fitlins'
__packagename__ = 'fitlins'
__description__ = 'Fit Linear Models to fMRI Data'
__longdesc__ = ''

DOWNLOAD_URL = (
    'https://github.com/poldracklab/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))


SETUP_REQUIRES = [
    'setuptools>=27.0',
]

REQUIRES = [
    'nibabel>=2.0',
    'nipype>=1.1.0',
    'seaborn>=0.7.1',
    'numpy>=1.11',
    'nilearn>=0.4.0',
    'pandas>=0.19',
    'tables>=3.2.1',
    'nistats>=0.0.1a',
    'grabbit>=0.2.2',
    'pybids>=0.6.4',
    'jinja2',
]

LINKS_REQUIRES = [
    'git+https://github.com/INCF/pybids.git@'
    'a86a0e862e93001536bff28e4c2ba35906824d2c#egg=pybids',
    'git+https://github.com/nistats/nistats.git@'
    'ce3695e8f34c6f34323766dc96a60a53b69d2729#egg=nistats',
]

TESTS_REQUIRES = [
]

EXTRA_REQUIRES = {
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]

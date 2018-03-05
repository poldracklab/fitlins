# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

__version__ = '0.0.1'
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
    'seaborn>=0.7.1',
    'numpy>=1.11',
    'nilearn>=0.4.0',
    'pandas>=0.19',
    'nistats>=0.0.1a',
    'pybids>=0.3',
    'jinja2',
]

LINKS_REQUIRES = [
    'git+https://github.com/grabbles/grabbit.git@'
    'f317224f90f572ec35899a43f19cb82b870ff772#egg=grabbit-0.1.1-dev',
    'git+https://github.com/INCF/pybids.git@'
    'ea6cf9db042f25dfcae907bd535e9dcabcf2bd55#egg=pybids-0.4.3-dev',
    'git+https://github.com/effigies/nistats.git@'
    'd5b63e0ce1ea817bacd9117b9d729b3ca62c9352#egg=nistats-0.0.1b-dev',
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

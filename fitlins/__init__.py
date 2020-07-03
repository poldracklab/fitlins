#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings
warnings.filterwarnings('ignore', r'numpy.ufunc size changed')
warnings.filterwarnings('ignore', r'resolve package from __spec__')
del warnings

try:
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use('Agg')
    del matplotlib

from .__about__ import __version__

try:
    import bids
except ImportError:
    pass
else:
    bids.config.set_option('extension_initial_dot', True)
    del bids

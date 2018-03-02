#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import matplotlib

from .info import (
    __version__,
    __author__,
    __copyright__,
    __credits__,
    __license__,
    __maintainer__,
    __email__,
    __status__,
    __url__,
    __packagename__,
    __description__,
    __longdesc__
)

matplotlib.use('Agg')

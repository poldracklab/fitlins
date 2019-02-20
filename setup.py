#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" fitlins setup script """
from setuptools import setup


def main():
    """ Install entry-point """
    import os
    from inspect import getfile, currentframe
    from fitlins.__about__ import (
        DOWNLOAD_URL,
    )

    pkg_data = {'fitlins': ['data/fitlins.json', 'data/*.tpl']}

    root_dir = os.path.dirname(os.path.abspath(getfile(currentframe())))

    version = None
    cmdclass = {}
    if os.path.isfile(os.path.join(root_dir, 'fitlins', 'VERSION')):
        with open(os.path.join(root_dir, 'fitlins', 'VERSION')) as vfile:
            version = vfile.readline().strip()
        pkg_data['fitlins'].insert(0, 'VERSION')

    if version is None:
        import versioneer
        version = versioneer.get_version()
        cmdclass = versioneer.get_cmdclass()

    setup(
        version=version,
        cmdclass=cmdclass,
        download_url=DOWNLOAD_URL,
    )


if __name__ == '__main__':
    main()

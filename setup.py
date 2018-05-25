#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" fitlins setup script """


def main():
    """ Install entry-point """
    import os
    from setuptools import setup, find_packages
    from inspect import getfile, currentframe
    import versioneer
    from fitlins.__about__ import (
        __packagename__,
        __version__,
        __author__,
        __email__,
        __maintainer__,
        __license__,
        __description__,
        __longdesc__,
        __url__,
        DOWNLOAD_URL,
        CLASSIFIERS,
        REQUIRES,
        SETUP_REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
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

    extensions = []

    setup(
        name=__packagename__,
        version=version,
        cmdclass=cmdclass,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        license=__license__,
        classifiers=CLASSIFIERS,
        download_url=DOWNLOAD_URL,
        # Dependencies handling
        setup_requires=SETUP_REQUIRES,
        install_requires=REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        dependency_links=LINKS_REQUIRES,
        package_data=pkg_data,
        entry_points={'console_scripts': ['fitlins=fitlins.cli.run:main']},
        packages=find_packages(exclude=("tests",)),
        zip_safe=False,
        ext_modules=extensions
    )


if __name__ == '__main__':
    main()

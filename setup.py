#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" fitlins setup script """


def main():
    """ Install entry-point """
    from setuptools import setup, find_packages
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

    extensions = []

    setup(
        name=__packagename__,
        version=__version__,
        cmdclass=versioneer.get_cmdclass(),
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
        package_data={'fitlins': ['data/fitlins.json', 'data/*.tpl']},
        entry_points={'console_scripts': ['fitlins=fitlins.cli.run:main']},
        packages=find_packages(exclude=("tests",)),
        zip_safe=False,
        ext_modules=extensions
    )


if __name__ == '__main__':
    main()

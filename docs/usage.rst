.. include:: links.rst

.. _usage:

Usage
-----

Execution and the BIDS format
=============================

The FitLins workflow takes as principal inputs a :abbr:`BIDS (Brain Imaging
Data Structure)` dataset, one or more derivative datasets, and a `BIDS Stats
Models`_ file.
We recommend using `fMRIPrep` for preprocessing your dataset.

The exact command to run ``fitlins`` depends on the :ref:`Installation` method.
The common parts of the command follow the BIDS-Apps_ definition.

Example: ::

    fitlins data/bids_root/ out/ participant \
        -d data/derivatives/fmriprep/ -w work/


Command-Line Arguments
======================

.. argparse::
   :ref: fitlins.cli.run.get_parser
   :prog: fitlins
   :nodefault:
   :nodefaultconst:

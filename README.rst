FitLins - Fitting Linear Models to fMRI data
============================================

FitLins is a tool for estimating linear models, defined by the `BIDS Model`_
specification proposal, to BIDS-formatted datasets.

This software is in alpha stage, and should be considered unstable.
Users are welcome to test the software, and open issues.

The CLI follows the `BIDS-Apps`_ convention:

Usage::

    fitlins/cli/run.py <bids_root> <out_dir> <analysis_level> [--model <model_name>]

See the output of ``fitlins/cli/run.py --help`` for all valid options.

.. _"BIDS Model": https://docs.google.com/document/d/1bq5eNDHTb6Nkx3WUiOBgKvLNnaa5OMcGtD0AZ9yms2M/
.. _BIDS-Apps: http://bids-apps.neuroimaging.io

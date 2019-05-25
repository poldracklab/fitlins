FitLins - Fitting Linear Models to BIDS Datasets
================================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1306215.svg
   :target: https://doi.org/10.5281/zenodo.1306215

.. image:: https://codecov.io/gh/poldracklab/fitlins/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/poldracklab/fitlins

.. image:: https://circleci.com/gh/poldracklab/fitlins.svg?style=svg
   :target: https://circleci.com/gh/poldracklab/fitlins

FitLins is a tool for estimating linear models, defined by the
`BIDS Stats Models`_ specification proposal, to `BIDS`_-formatted datasets.

FitLins is developed against `fMRIPrep`_-preprocessed datasets, but is intended to
work with any dataset following the `BIDS Derivatives`_ draft specification.

Example models can be found in `examples/models`_ in the main repository and
`FitLins Example Models`_ in the documentation.

This pipeline is developed by the `Poldrack lab at Stanford University
<https://poldracklab.stanford.edu/>`_ for use at the `Center for Reproducible
Neuroscience (CRN) <http://reproducibility.stanford.edu/>`_, as well as for
open-source software distribution.

.. _BIDS: https://bids.neuroimaging.io/
.. _`BIDS Stats Models`: https://docs.google.com/document/d/1bq5eNDHTb6Nkx3WUiOBgKvLNnaa5OMcGtD0AZ9yms2M/
.. _`BIDS Derivatives`: https://bids-specification.readthedocs.io/en/derivatives/05-derivatives/01-introduction.html
.. _BIDS-Apps: http://bids-apps.neuroimaging.io
.. _fMRIPrep: https://fmriprep.readthedocs.io
.. _`OpenFMRI dataset ds000030`: http://datasets.datalad.org/?dir=/openfmri/ds000030/
.. _Zenodo: https://doi.org/10.5281/zenodo.1306215
.. _examples/models: https://github.com/poldracklab/fitlins/tree/master/examples/models/
.. _`FitLins Example Models`: https://fitlins.readthedocs.io/en/latest/examples/models.html

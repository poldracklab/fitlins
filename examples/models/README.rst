======================
FitLins Example Models
======================

This is a collection of example models, written for OpenNeuro (formerly OpenFMRI)
datasets.

Statistical models, described in the draft `BIDS Stats-Models`_ specification, fit
into the BIDS data structure with the following naming convention::

    <bids_root>/models/model-<label>_[desc-<description>]_smdl.json

FitLins accepts models that are present in a BIDS directory or are passed with the
``-m``/``--model`` flag.

.. _BIDS Stats-Models: https://bids-standard.github.io/stats-models/

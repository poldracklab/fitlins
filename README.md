FitLins Example Models
======================

This repository contains example BIDS models.
Its contents match the DataLad structure, and should be directly copyable.

OpenFMRI
--------
* ds000003/model.json
  * Translated from [model001](https://openfmri.org/s3-browser/?prefix=ds000003/ds000003_R1.1.0/uncompressed/models/model001/)
* ds000030/model.json
* ds000114/model.json
  * This model can be run by FitLins, but it has a second-level contrast that
    nistats cannot currently handle, so all group level stats will be `NaN`.

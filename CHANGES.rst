0.0.4 (July 05, 2018)
=====================

  * ENH: Make dataset_description optional (https://github.com/poldracklab/fitlins/pull/51)
  * ENH: Loop over preproc files, instead of raw BOLD files (https://github.com/poldracklab/fitlins/pull/50)
  * ENH: Add --n-cpus option to CLI (https://github.com/poldracklab/fitlins/pull/49)
  * ENH: Run datasinks on main thread (https://github.com/poldracklab/fitlins/pull/39)
  * ENH: Enable derivative label to tag pipelines (https://github.com/poldracklab/fitlins/pull/37)
  * ENH: Make dataset description (https://github.com/poldracklab/fitlins/pull/29)
  * ENH: Add trivial dataset_description.json (https://github.com/poldracklab/fitlins/pull/31)
  * ENH: Run auto model by default (https://github.com/poldracklab/fitlins/pull/26)
  * ENH: Rewrite first level analysis as Nipype workflow (https://github.com/poldracklab/fitlins/pull/16)
  * ENH: Add acq, rec, run and echo to output patterns (https://github.com/poldracklab/fitlins/pull/20)
  * FIX: Second level contrast computation, and dense/sparse transformation issues (https://github.com/poldracklab/fitlins/pull/48)
  * FIX: Include run in derivative names (https://github.com/poldracklab/fitlins/pull/43)
  * FIX: Force string input to snake_to_camel (https://github.com/poldracklab/fitlins/pull/41)
  * FIX: Versioneer adjustments (https://github.com/poldracklab/fitlins/pull/36)
  * FIX: Create group level results dir (https://github.com/poldracklab/fitlins/pull/22)
  * RF: Simplify entry-points, restore preproc discovery (https://github.com/poldracklab/fitlins/pull/38)
  * MAINT: Update pybids 0.6.3, grabbit 0.2.1 (https://github.com/poldracklab/fitlins/pull/52)
  * MAINT: Manage version with versioneer (https://github.com/poldracklab/fitlins/pull/35)
  * MAINT: Neuroscout changes / pybids updates (https://github.com/poldracklab/fitlins/pull/28)
  * MAINT: Add nipype dependency, remove unused code (https://github.com/poldracklab/fitlins/pull/27)


0.0.3 (March 9, 2018)
---------------------

Maintenance release

* Update grabbit (0.1.1), pybids (0.5.0) (#11)
* Incorporate nistats/nistats#165 (#13)
* Update Dockerfile, versioning (#14)


0.0.2 (March 5, 2018)
---------------------

Hotfix, addressing deployment issues.


0.0.1 (March 5, 2018)
---------------------

Initial release of FitLins, a BIDS-model fitting BIDS app.

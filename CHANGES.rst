0.2.0 (February 1, 2019)
=======================

This release marks a substantial refactoring in the wake of
[BIDS Derivatives RC1](https://docs.google.com/document/d/17ebopupQxuRwp7U7TFvS6BH03ALJOgGHufxK8ToAvyI/),
[fMRIPrep 1.2.x](https://fmriprep.readthedocs.io/en/stable/changes.html#january-17-2019) and
[pybids 0.7.0](https://github.com/bids-standard/pybids/releases/tag/0.7.0).

Reports at second level and higher are currently broken, but we're at a point where neuroscout
is depending on the current code base, the user base is increasing, and it's worth having a starting
point for considering new features.

With thanks to Alejandro de la Vega, Adina Wagner and Yaroslav Halchenko for contributions.

* FIX: Allow derivatives to be a boolean value (https://github.com/poldracklab/fitlins/pull/91)
* FIX: Restore report generation (https://github.com/poldracklab/fitlins/pull/88)
* ENH: Plotting improvements (https://github.com/poldracklab/fitlins/pull/89)
* ENH: Allow selecting for no space ``--space ''`` option (https://github.com/poldracklab/fitlins/pull/96)
* ENH: Allow selecting for desc with ``--desc`` option (https://github.com/poldracklab/fitlins/pull/95)
* MAINT: Depend on unreleased pybids commit (https://github.com/poldracklab/fitlins/pull/99)
* MAINT: Pybids 0.7.0 compatibility (https://github.com/poldracklab/fitlins/pull/84)

0.1.0 (August 24, 2018)
=======================

This release moves FitLins to a Nipype workflow and provides a set of Nipype interfaces for interacting with BIDS Models and the nistats statistical package.

* FIX: Correctly handle missing confounds (https://github.com/poldracklab/fitlins/pull/73)
* ENH: Set loop_preproc during model loading (https://github.com/poldracklab/fitlins/pull/66)
* REF: Second-level workflow (https://github.com/poldracklab/fitlins/pull/30)
* DOC: Example model (https://github.com/poldracklab/fitlins/pull/63)
* MAINT: Update nipype, grabbit and pybids dependencies (https://github.com/poldracklab/fitlins/pull/70)

0.0.6 (August 06, 2018)
=======================

Hotfix release.

* FIX: Explicitly create working directory (https://github.com/poldracklab/fitlins/pull/61)


0.0.5 (August 01, 2018)
=======================

* FIX: Limit NaN imputation and use mean non-zero value (https://github.com/poldracklab/fitlins/pull/57)


0.0.4 (July 05, 2018)
=====================

* ENH: Allow models without non-HRF variables (https://github.com/poldracklab/fitlins/pull/55)
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
=====================

Maintenance release

* Update grabbit (0.1.1), pybids (0.5.0) (#11)
* Incorporate nistats/nistats#165 (#13)
* Update Dockerfile, versioning (#14)


0.0.2 (March 5, 2018)
=====================

Hotfix, addressing deployment issues.


0.0.1 (March 5, 2018)
=====================

Initial release of FitLins, a BIDS-model fitting BIDS app.

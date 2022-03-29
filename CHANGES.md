0.10.1 (March 29, 2022)
=======================
Bug-fix release in the 0.10.x series.

Most of the fixes come in the upstream fixes of PyBIDS, which will get bundled in
the Docker image.

* ENH: Set prog name to fitlins, regardless of which CLI program was called (https://github.com/poldracklab/fitlins/pull/350)
* MNT: Update `bug_report.md` (https://github.com/poldracklab/fitlins/pull/337)
* CI: Comprehensive testing (https://github.com/poldracklab/fitlins/pull/344)
* CI: Unify Docker deployment steps (https://github.com/poldracklab/fitlins/pull/334)
* Build(deps): Bump actions/cache from 2 to 3 (https://github.com/poldracklab/fitlins/pull/347)
* Build(deps): Bump actions/checkout from 2 to 3 (https://github.com/poldracklab/fitlins/pull/343)
* Build(deps): Bump actions/setup-python from 2 to 3 (https://github.com/poldracklab/fitlins/pull/340)

0.10.0 (February 01, 2022)
==========================
New feature release in the 0.10.x series.

This release implements support for BIDS Stats Models 1.0.0-rc1.
While we have attempted not to break old-style models, we no longer support them.

* RF: Use the new bids stats model (https://github.com/poldracklab/fitlins/pull/298)
* FIX: minor typo; missed restyling in afni.py (https://github.com/poldracklab/fitlins/pull/328)
* FIX: Minor fixes (https://github.com/poldracklab/fitlins/pull/326)
* FIX: version info & detail activating venv on different OS (https://github.com/poldracklab/fitlins/pull/311)
* ENH: Migrate CircleCI to github actions  (https://github.com/poldracklab/fitlins/pull/329)
* STY: Reformat to black, address flake8 issues (https://github.com/poldracklab/fitlins/pull/316)
* MNT: Temporarily pull pybids from github maintenance branch (https://github.com/poldracklab/fitlins/pull/308)
* CI: Update ds003 tests (https://github.com/poldracklab/fitlins/pull/331)
* CI: Check for secrets before deploying, switch to python-build (https://github.com/poldracklab/fitlins/pull/332)

0.9.2 (July 21, 2021)
======================

Minor bugfix release in the 0.9.x series.

* FIX: Exclude res from report entities (https://github.com/poldracklab/fitlins/pull/301)
* MNT: Update to versioneer 0.20 (https://github.com/poldracklab/fitlins/pull/300) 
* MNT: codespell run + ignore (https://github.com/poldracklab/fitlins/pull/289) 

0.9.1 (April 21, 2021)
======================

Bugfix release in the 0.9.x series.
Upgrades pybids and nilearn to latest versions. 

* PIN: nilearn 0.7.1+ (https://github.com/poldracklab/fitlins/pull/286)
* PIN: Upgrade pybids to 0.13 (https://github.com/poldracklab/fitlins/pull/284)

0.9.0 (February 26, 2021)
========================

New feature release in the 0.9.x series.

This release added AFNI's 3dREMLfit for first-level estimation, dropped nistats
in favor of nilearn.stats, added test reference outputs, among other minor improvements.

* AFNI TSNR calculation, ADD: BlurToFWHM, ADD: testing outputs against reference (https://github.com/poldracklab/fitlins/pull/280)
* FIX: AFNI contrast bool (https://github.com/poldracklab/fitlins/pull/266)
* ENH: Writeout estimator to derivative description (https://github.com/poldracklab/fitlins/pull/278)
* ENH: disable automatic drift modeling (https://github.com/poldracklab/fitlins/pull/276)
* ENH: modify path patterns to make task optional (https://github.com/poldracklab/fitlins/pull/275)
* ENH: Add AFNI 3dREMLfit for first-level estimation (https://github.com/poldracklab/fitlins/pull/171)
* TST: Add dummy dataset for pytest/local testing (https://github.com/poldracklab/fitlins/pull/262) 
* MNT: Update PyBIDS and nilearn (https://github.com/poldracklab/fitlins/pull/268)
* CI: Make faster? (https://github.com/poldracklab/fitlins/pull/264)


0.8.0 (October 19, 2020)
========================

New feature release in the 0.8.x series.

This release includes a change where events that are not explicitly convolved
are modeled with boxcar functions instead of the nistats default.

* FIX: Sparse matrix should be in long format, and HRF should be None by default (https://github.com/poldracklab/fitlins/pull/259)

0.7.1 (October 09, 2020)
========================

Bug-fix release in the 0.7.x series.

This release requires PyBIDS 0.12.2 or higher, fixing various bugs that affected FitLins.

* Only drop modulation if needed (https://github.com/poldracklab/fitlins/pull/253)

0.7.0 (August 05, 2020)
=======================

New feature release in the 0.7.x series. This is an accumulation of changes over
several months.

This release requires PyBIDS 0.12 and includes some *quite provisional* CIFTI-2 support
as well as log-likelihood and R^2 maps for assessing goodness of fit.

* FIX: Downgrade Sphinx to <3.0 (https://github.com/poldracklab/fitlins/pull/226)
* FIX: Validate BIDS datasets except when searching for models (https://github.com/poldracklab/fitlins/pull/214)
* FIX: Blacklist coverage >=5 (https://github.com/poldracklab/fitlins/pull/209)
* ENH: Bundle Conte69 surfaces (https://github.com/poldracklab/fitlins/pull/249)
* ENH: Add CIFTI-2 support (https://github.com/poldracklab/fitlins/pull/239)
* ENH: Add logL as model output (https://github.com/poldracklab/fitlins/pull/230)
* ENH: Save model level output (r-squared) (https://github.com/poldracklab/fitlins/pull/224)
* ENH: Add FEMA as key value to stat `path_pattern` (https://github.com/poldracklab/fitlins/pull/220)
* RF: Accept nipype configurations from users (https://github.com/poldracklab/fitlins/pull/229)
* MNT: Update requirements (https://github.com/poldracklab/fitlins/pull/225)
* MNT: Bump pybids to 0.10.2 (https://github.com/poldracklab/fitlins/pull/216)
* CI: Improve outputs (https://github.com/poldracklab/fitlins/pull/218)

0.6.2 (December 13, 2019)
=========================

Hotfix release.

* FIX: FEMA input index should filter on rows, not columns. (https://github.com/poldracklab/fitlins/pull/207)


0.6.1 (December 12, 2019)
=========================

Hotfix release.

* FIX: Add desc=preproc as filter when finding preproc BOLD files (https://github.com/poldracklab/fitlins/pull/204)


0.6.0 (December 11, 2019)
=========================

New feature release in the 0.6.x series.

This release respects recent changes to the BIDS-StatsModels draft
specification to support fixed-effects meta-analysis (FEMA) contrasts,
and renames "AutoContrasts" to "DummyContrasts".

Provisional support for F-tests has been added.

Additional rearchitecting by Dylan Nielson provides significant speedups
for large datasets by caching BIDS layout information.

* SPEC: Add fixed effects (FEMA) contrasts (https://github.com/poldracklab/fitlins/pull/191)
* SPEC: Change AutoContrasts to DummyContrasts (https://github.com/poldracklab/fitlins/pull/197)
* FIX: Don't pass ignore and force index to `init_fitlins_wf` (https://github.com/poldracklab/fitlins/pull/202)
* FIX: BIDSelect confusion between Nifti and JSON (https://github.com/poldracklab/fitlins/pull/193)
* FIX: Set `smoothing_fwhm` after creating next level. (https://github.com/poldracklab/fitlins/pull/190)
* FIX: Recognize cosine columns without underscores (https://github.com/poldracklab/fitlins/pull/185)
* ENH: Update logging levels (https://github.com/poldracklab/fitlins/pull/198)
* ENH: Add option to load BIDS layouts from database file (https://github.com/poldracklab/fitlins/pull/187)
* ENH: Add memory estimate for first-level models, enable memory management (https://github.com/poldracklab/fitlins/pull/199)
* ENH: Avoid casting BOLD data to float64 if possible (https://github.com/poldracklab/fitlins/pull/196)
* ENH: Add F-tests (https://github.com/poldracklab/fitlins/pull/195)
* ENH: Drop missing model inputs (https://github.com/poldracklab/fitlins/pull/183)
* RF: Abstract interfaces to simplify swappability (https://github.com/poldracklab/fitlins/pull/188)


0.5.1 (September 23, 2019)
==========================

Bug fix release to work with PyBIDS 0.9.4+.

* FIX: Expand entity whitelist (https://github.com/poldracklab/fitlins/pull/182)
* FIX: Don't validate generated paths (https://github.com/poldracklab/fitlins/pull/180)


0.5.0 (July 03, 2019)
=====================

This release features significant improvements to reporting and documentation,
including a Jupyter notebook to demonstrate usage. Example models are now in
the main branch of the repository, and annotated in the documentation.

* FIX: Smoothing level, check length  (https://github.com/poldracklab/fitlins/pull/157)
* FIX: mask parameter to FirstLevelModel is deprecated for mask_img (https://github.com/poldracklab/fitlins/pull/158)
* ENH: Add ds000117 model (https://github.com/poldracklab/fitlins/pull/163)
* ENH: Move to single-page report (https://github.com/poldracklab/fitlins/pull/161)
* ENH: Add task-vs-baseline contrast to ds003 (https://github.com/poldracklab/fitlins/pull/160)
* ENH: Reporting cleanups (https://github.com/poldracklab/fitlins/pull/155)
* DOC: Curate reports and models (https://github.com/poldracklab/fitlins/pull/153)
* DOC: Add example running through DS003 (https://github.com/poldracklab/fitlins/pull/152)
* MAINT: BIDSLayout.get() parameter "extensions" deprecated (https://github.com/poldracklab/fitlins/pull/167)
* MAINT: Update pybids dependency, package name (https://github.com/poldracklab/fitlins/pull/166)


0.4.0 (May 10, 2019)
====================

This release produces effect, variance, statistic (t or F), Z-score, and p-value
maps at every level, and enables smoothing at higher levels if preferred.

Additionally, documentation has been added at https://fitlins.readthedocs.io and
versioning/packaging issues have been resolved.

* FIX: Do not install FitLins as editable in Docker (https://github.com/poldracklab/fitlins/pull/137)
* ENH: Save design matrix as TSV to output directory (https://github.com/poldracklab/fitlins/pull/143)
* ENH: Enable smoothing at any analysis level (https://github.com/poldracklab/fitlins/pull/135)
* ENH: Produce all available statistical maps from each analysis unit (https://github.com/poldracklab/fitlins/pull/131)
* ENH: Add version to non-release Docker images. (https://github.com/poldracklab/fitlins/pull/136)
* DOC: Flesh out documentation (https://github.com/poldracklab/fitlins/pull/147)
* DOC: Build API docs on RTD (https://github.com/poldracklab/fitlins/pull/146)
* DOC: Create Sphinx documentation with API autodocs (https://github.com/poldracklab/fitlins/pull/145)
* MAINT: Drop Python 3.5 support (https://github.com/poldracklab/fitlins/pull/140)
* CI: Run FitLins with coverage (https://github.com/poldracklab/fitlins/pull/144)
* CI: Test FitLins on OpenNeuro DS000003, preprocessed with fMRIPrep 1.3.2 (https://github.com/poldracklab/fitlins/pull/141)


0.3.0 (April 19, 2019)
======================

This release restores reports at the second level and higher, and enables isotropic
smoothing with the nistats backend. Reporting has also been refactored to reduce
clutter in the outputs.

With thanks to Karolina Finc, Rastko Ciric and Mathias Goncalves for contributions.

* FIX: Restore level 2+ reports (https://github.com/poldracklab/fitlins/pull/130)
* FIX: Remove uninformative metadata from derivative filenames (https://github.com/poldracklab/fitlins/pull/129)
* FIX: Re-enable analysis level selection (https://github.com/poldracklab/fitlins/pull/120)
* FIX: Switch plot colors to conventional blue for negative, red for positive (https://github.com/poldracklab/fitlins/pull/108)
* ENH: Save crashfiles as text in working directory (https://github.com/poldracklab/fitlins/pull/121)
* ENH: Add naive isotropic smoothing (https://github.com/poldracklab/fitlins/pull/104)
* REF: Delegate isotropic smoothing to nistats (https://github.com/poldracklab/fitlins/pull/118)
* DOC: Update README with latest help text, remove smoothing disclaimer (https://github.com/poldracklab/fitlins/pull/119)
* MAINT: Add contributors to Zenodo (https://github.com/poldracklab/fitlins/pull/122)
* MAINT: Consolidate configuration (https://github.com/poldracklab/fitlins/pull/113)
* MAINT: Pybids 0.8 compatibility (https://github.com/poldracklab/fitlins/pull/109)
* MAINT: Use numpy 1.15 to accommodate pytables (https://github.com/poldracklab/fitlins/pull/106)

0.2.0 (February 1, 2019)
========================

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

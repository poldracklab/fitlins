# Integration tests for the fitlins commandline

The tests in this directory call fitlins from the commandline in order to compare outputs with known reference values. Refernce data can be obtained with datalad via:
```
datalad install -r -s https://gin.g-node.org/shotgunosine/fitlins_tests
datalad get fitlins_tests/ds003/ds003_fmriprep/sub-0{1,2,3}/func/*_space-MNI152NLin2009cAsym_desc-*.nii.gz \
                        fitlins_tests/ds003/ds003_fmriprep/sub-0{1,2,3}/func/*_desc-confounds_*.tsv \
                        fitlins_tests/ds003/ds003_fmriprep/dataset_description.json \
                        fitlins_tests/ds003/ds003_fmriprep/sub-*/*/*.json
datalad get -r fitlins_tests/ds003/nistats_smooth/ fitlins_tests/ds003/afni_smooth/ fitlins_tests/ds003/afni_blurto/
```


Calling these tests requires defining paths to the test data, output location, and working directories, as well as specifying which of the three predefined tests you'd like to run:
```
job_name=nistats_smooth
tests_dir=[path to your fitlins_tests directory]
pytest fitlins/fitlins/tests --bids-dir=${tests_dir}ds003/ds003_fmriprep/sourcedata --output-dir=${job_name}/out --derivatives=${tests_dir}/ds003/ds003_fmriprep/ --model=${tests_dir}/ds003/models/model-001_smdl.json --work-dir=/scratch --test-name=${job_name} --database-path=${tests_dir}/ds003_database --reference-dir=${tests_dir}/ds003/
```

The three tests are:
- afni_smooth: AFNI estimator with 10mm isotropic smoothing added
- afni_blurto: AFNI estimator blured to 5mm smoothness
- nistats_smooth: Nistats estimator with 10mm isotropic smoothing added
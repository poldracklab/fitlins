#!/bin/bash

echo Running checks

source tools/ci/env.sh

set -eu

# Required variables
echo CHECK_TYPE = $CHECK_TYPE
if [ "$CHECK_TYPE" == "workflow" ]; then
  echo DATA = $DATA
  echo TEST_NAME = $TEST_NAME
  echo OUTPUT_DIR = $OUTPUT_DIR
  echo WORK_DIR = $WORK_DIR
fi

set -x

if [ "$CHECK_TYPE" == "test" ]; then
  pytest --cov fitlins --ignore-glob=fitlins/tests/* fitlins $@
elif [ "$CHECK_TYPE" == "workflow" ]; then
  pytest -sv -r s --cov fitlins fitlins/tests \
    --bids-dir $DATA/inputs/ds000003 \
    --derivatives $DATA/inputs/ds000003-fmriprep \
    --model $DATA/inputs/models/model-001_smdl.json \
    --reference-dir $DATA/outputs \
    --output-dir $OUTPUT_DIR \
    --database-path $OUTPUT_DIR/bidsdb \
    --work-dir $WORK_DIR \
    --test-name $TEST_NAME $@
fi

coverage xml

set +eux

echo Done running checks

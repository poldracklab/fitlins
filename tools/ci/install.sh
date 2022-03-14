#!/bin/bash

echo Installing fitlins

source tools/ci/env.sh

set -eu

# Required variables
echo INSTALL_TYPE = $INSTALL_TYPE
echo CHECK_TYPE = $CHECK_TYPE
echo EXTRA_PIP_FLAGS = $EXTRA_PIP_FLAGS

set -x

if [ -n "$EXTRA_PIP_FLAGS" ]; then
    EXTRA_PIP_FLAGS=${!EXTRA_PIP_FLAGS}
fi

pip install $EXTRA_PIP_FLAGS .

# Basic import check
python -c 'import fitlins; print(fitlins.__version__)'

if [ "$CHECK_TYPE" == "skiptests" ]; then
    exit 0
fi

pip install $EXTRA_PIP_FLAGS "fitlins[$CHECK_TYPE]"

set +eux

echo Done install fitlins

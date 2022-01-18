#!/bin/bash

docker run --rm -i -a stdin -a stdout \
  repronim/neurodocker:0.7.0 generate docker - < .neurodocker.json > Dockerfile

#!/bin/bash

docker run --rm -i -a stdin -a stdout \
  repronim/neurodocker:master generate docker - < .neurodocker.json > Dockerfile

#!/bin/bash

# build the image
# run it with volumes defined in the constants
# --build-arg
# docker build -t intro_ai .
docker build --build-arg PYTHON_VERSION=$1 --build-arg NOTEBOOK_NAME=$2  -t $3 .
# docker run --gpus all --rm -p 8889:8889 -v (pwd)/src:/src -v (pwd)/container_cache/torch:/root/.cache/torch/checkpoints $2
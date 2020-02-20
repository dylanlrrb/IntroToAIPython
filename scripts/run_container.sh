#!/bin/bash

VOLUMES=""

for i in "${@:4}"
do
    VOLUMES="$VOLUMES -v $(pwd)$i"
done

docker build --build-arg PYTHON_VERSION=$1 --build-arg NOTEBOOK_NAME=$2  -t $3 .
echo "Running with mounted volumes -> $VOLUMES"
docker run --gpus all --rm $VOLUMES $3

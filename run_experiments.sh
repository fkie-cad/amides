#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_RESULTS_CONTAINER="amides-results"

echo "Starting AMIDES results container '$AMIDES_RESULTS_CONTAINER' ..."
docker run --rm --name $AMIDES_RESULTS_CONTAINER --interactive --tty --mount type=bind,source="$(pwd)"/amides/models,target=/amides/models --mount type=bind,source="$(pwd)"/amides/plots,target=/amides/plots --mount type=bind,source="$(pwd)"/data,target=/data $AMIDES_IMAGE ./experiments.sh
if [ $? -eq 0 ]; then
    echo "Successfully executed AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
else
    echo "Failed to execute AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
fi

#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_RESULTS_CONTAINER="amides-results"

echo "Starting AMIDES results container '$AMIDES_RESULTS_CONTAINER' ..."
docker start -i $AMIDES_RESULTS_CONTAINER
if [ $? -eq 0 ]; then
    echo "Successfully executed AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
else
    echo "Failed to execute AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
fi

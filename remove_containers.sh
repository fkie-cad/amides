#!/bin/bash

AMIDES_RESULTS_CONTAINER="amides-results"
AMIDES_ENV_CONTAINER="amides-env"

echo "Removing AMIDES results container '$AMIDES_RESULTS_CONTAINER'..."
docker rm --force $AMIDES_RESULTS_CONTAINER
if [ $? -eq 0 ]; then 
    echo "Successfully removed AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
else
    echo "Failed to remove AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
fi


echo "Removing AMIDES env container '$AMIDES_ENV_CONTAINER'..."
docker rm --force $AMIDES_ENV_CONTAINER
if [ $? -eq 0 ]; then
    echo "Successfully removed AMIDES env container '$AMIDES_ENV_CONTAINER'"
else
    echo "Failed to remove AMIDES env container '$AMIDES_ENV_CONTAINER'"
fi

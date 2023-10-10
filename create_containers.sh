#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_RESULTS_CONTAINER="amides-results"
AMIDES_ENV_CONTAINER="amides-env"


echo "Creating AMIDES results container '$AMIDES_RESULTS_CONTAINER'..."
docker create --name $AMIDES_RESULTS_CONTAINER --interactive --tty --mount type=bind,source="$(pwd)"/amides/models,target=/amides/models --mount type=bind,source="$(pwd)"/amides/plots,target=/amides/plots --mount type=bind,source="$(pwd)"/data,target=/data $AMIDES_IMAGE ./bin/results.sh
if [ $? -eq 0 ]; then
    echo "Successfully created AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
else
    echo "Failed creating AMIDES results container '$AMIDES_RESULTS_CONTAINER'"
fi

echo "Creating AMIDES environment container '$AMIDES_ENV_CONTAINER'..."
docker create --name $AMIDES_ENV_CONTAINER --interactive --tty --mount type=bind,source="$(pwd)"/amides/models,target=/amides/models --mount type=bind,source="$(pwd)"/amides/plots,target=/amides/plots --mount type=bind,source="$(pwd)"/data,target=/data $AMIDES_IMAGE /bin/bash
if [ $? -eq 0 ]; then
    echo "Successfully created AMIDES environment container '$AMIDES_ENV_CONTAINER'"
else
    echo "Failed creating AMIDES environment container '$AMIDES_ENV_CONTAINER'"
fi

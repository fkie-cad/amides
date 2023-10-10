#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_ENV_CONTAINER="amides-env"

echo "Starting AMIDES environment container '$AMIDES_ENV_CONTAINER'..."
docker start -i $AMIDES_ENV_CONTAINER
if [ $? -eq 0 ]; then
    echo "Successfully executed AMIDES environment container '$AMIDES_ENV_CONTAINER'"
else
    echo "Failed to execute AMIDES environment container '$AMIDES_ENV_CONTAINER'"
fi

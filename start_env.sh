#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_ENV_CONTAINER="amides-env"

echo "########## Starting AMIDES environment container '$AMIDES_ENV_CONTAINER'... ##########"
docker run --name $AMIDES_ENV_CONTAINER --interactive --tty --user docker-user --mount type=bind,source="$(pwd)"/amides/models,target=/home/docker-user/amides/models --mount type=bind,source="$(pwd)"/amides/plots,target=/home/docker-user/amides/plots --mount type=bind,source="$(pwd)"/data,target=/home/docker-user/data $AMIDES_IMAGE /bin/bash
if [ $? -eq 0 ]; then
    echo "########## Successfully executed AMIDES environment container '$AMIDES_ENV_CONTAINER' ##########"
else
    echo "########## Failed to execute AMIDES environment container '$AMIDES_ENV_CONTAINER' ##########"
fi

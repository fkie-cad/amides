#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_ENV_CONTAINER="amides-env"
PWD=$(pwd)

mkdir -p $PWD/amides/models
mkdir -p $PWD/amides/plots

echo "########## Starting AMIDES environment container '$AMIDES_ENV_CONTAINER'... ##########"
docker run --name $AMIDES_ENV_CONTAINER --interactive --rm --tty --user docker-user --mount type=bind,source=$PWD/amides/models,target=/home/docker-user/amides/models --mount type=bind,source=$PWD/amides/plots,target=/home/docker-user/amides/plots --mount type=bind,source=$PWD/data,target=/home/docker-user/data $AMIDES_IMAGE /bin/bash
if [ $? -eq 0 ]; then
    echo "########## Successfully executed AMIDES environment container '$AMIDES_ENV_CONTAINER' ##########"
else
    echo "########## Failed to execute AMIDES environment container '$AMIDES_ENV_CONTAINER' ##########"
fi

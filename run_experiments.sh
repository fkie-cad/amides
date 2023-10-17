#!/usr/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_EXPERIMENTS_CONTAINER="amides-experiments"

echo "########## Starting AMIDES experiments container '$AMIDES_EXPERIMENTS_CONTAINER' ... ##########"
docker run --rm --name $AMIDES_EXPERIMENTS_CONTAINER --interactive --tty --user docker-user --mount type=bind,source="$(pwd)"/amides/models,target=/home/docker-user/amides/models --mount type=bind,source="$(pwd)"/amides/plots,target=/home/docker-user/amides/plots --mount type=bind,source="$(pwd)"/data,target=/home/docker-user/data $AMIDES_IMAGE ./experiments.sh
if [ $? -eq 0 ]; then
    echo "########## Successfully executed AMIDES experiments container '$AMIDES_EXPERIMENTS_CONTAINER' ##########"
else
    echo "########## Failed to execute AMIDES experiments container '$AMIDES_EXPERIMENTS_CONTAINER' ##########"
fi

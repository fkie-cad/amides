#!/bin/bash

AMIDES_EXPERIMENTS_CONTAINER="amides-experiments"
AMIDES_ENV_CONTAINER="amides-env"

echo "########## Removing AMIDES results container '$AMIDES_EXPERIMENTS_CONTAINER'... ##########"
docker rm --force $AMIDES_EXPERIMENTS_CONTAINER
if [ $? -eq 0 ]; then 
    echo "########## Successfully removed AMIDES results container '$AMIDES_EXPERIMENTS_CONTAINER' ##########"
else
    echo "########## Failed to remove AMIDES results container '$AMIDES_EXPERIMENTS_CONTAINER' ##########"
fi


echo "########## Removing AMIDES env container '$AMIDES_ENV_CONTAINER'... ##########"
docker rm --force $AMIDES_ENV_CONTAINER
if [ $? -eq 0 ]; then
    echo "########## Successfully removed AMIDES env container '$AMIDES_ENV_CONTAINER' ##########"
else
    echo "########## Failed to remove AMIDES env container '$AMIDES_ENV_CONTAINER' ##########"
fi

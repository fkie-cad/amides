#!/bin/bash

AMIDES_TAG="amides:base"
UID=$(id -u)
GID=$(id -g)

echo "########## Building AMIDES Docker image '$AMIDES_TAG'... ##########"
docker build --tag $AMIDES_TAG --build-arg UID=$UID --build-arg GID=$GID .
if [ $? -eq 0 ]; then
    echo "########## Successfully built AMIDES Docker image '$AMIDES_TAG' ##########"
else
    echo "########## Failed to build AMIDES Docker image '$AMIDES_TAG' ##########"
fi


#!/usr/bin/bash

AMIDES_TAG="amides:base"

echo "########## Building AMIDES Docker image '$AMIDES_TAG'... ##########"
docker build --tag $AMIDES_TAG .
if [ $? -eq 0 ]; then
    echo "########## Successfully built AMIDES Docker image '$AMIDES_TAG' ##########"
else
    echo "########## Failed to build AMIDES Docker image '$AMIDES_TAG' ##########"
fi


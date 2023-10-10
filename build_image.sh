#!/usr/bin/bash

AMIDES_TAG="amides:base"

echo "Building image '$AMIDES_TAG'..."
docker build --tag $AMIDES_TAG .


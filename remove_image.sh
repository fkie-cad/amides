#!/bin/bash

AMIDES_IMAGE="amides:base"

echo "Removing AMIDES base image '$AMIDES_IMAGE'..."
docker image rm --force $AMIDES_IMAGE
if [ $? -eq 0 ]; then
    echo "Successfully removed AMIDES base image '$AMIDES_IMAGE'"
else
    echo "Failed to remove AMIDES base image '$AMIDES_IMAGE'"
fi

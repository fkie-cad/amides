#!/bin/bash

AMIDES_IMAGE="amides:base"
AMIDES_RESULTS_CONTAINER="amides-results"
AMIDES_ENV_CONTAINER="amides-env"

echo "Removing containers..."
./remove_containers.sh
echo "Removing image '$AMIDES_IMAGE'..."
./remove_image.sh

echo "Removing generated models..."
sudo rm -r ./amides/models/*
if [ $? -eq 0 ]; then
    echo "Successfully removed generated models"
else
    echo "Failed to remove generated models"
fi

echo "Removing generated plots..."
sudo rm -r ./amides/plots/*
if [ $? -eq 0 ]; then
    echo "Successfully removed generated plots"
else
    echo "Failed to remove generated plots"
fi

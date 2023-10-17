#!/bin/bash

./remove_image.sh

echo "########## Removing generated models... ##########"
rm -r ./amides/models/*
if [ $? -eq 0 ]; then
    echo "########## Successfully removed generated models ##########"
else
    echo "########## Failed to remove generated models ##########"
fi

echo "########## Removing generated plots... ##########"
rm -r ./amides/plots/*
if [ $? -eq 0 ]; then
    echo "########## Successfully removed generated plots ##########"
else
    echo "########## Failed to remove generated plots ##########"
fi

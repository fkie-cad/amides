#!/bin/bash

echo "########## Creating results for misuse classification (C1, C2) ##########"
./classification.sh 

echo "########## Creating results for rule attribution (C3) ##########"
./rule_attribution.sh

echo "########## Creating results for tainted training data (C4) ##########"
./tainted_training.sh

echo "########## Creating results for other rule types (C5) ##########"
./classification_other_types.sh
